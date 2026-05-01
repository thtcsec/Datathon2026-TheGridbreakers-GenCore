from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import TABLE_ORDER, build_raw_audit_row, profile_table

DATE_COLUMNS = {
    "customers": ["signup_date"],
    "promotions": ["start_date", "end_date"],
    "orders": ["order_date"],
    "shipments": ["ship_date", "delivery_date"],
    "returns": ["return_date"],
    "reviews": ["review_date"],
    "sales": ["Date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
}

RESOLVED_STATUS = {"delivered", "cancelled", "returned"}
IN_FLIGHT_STATUS = {"created", "paid", "shipped"}


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (14, 6)
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.formatter.useoffset"] = False


def safe_divide(numerator: Any, denominator: Any, fill_value: float = 0.0) -> Any:
    if isinstance(numerator, pd.Series) and np.isscalar(denominator):
        den = np.nan if denominator in (0, None) or pd.isna(denominator) else denominator
        return numerator.divide(den).fillna(fill_value)
    if isinstance(denominator, pd.Series) and np.isscalar(numerator):
        den = denominator.replace(0, np.nan)
        return pd.Series(numerator, index=den.index, dtype="float64").divide(den).fillna(fill_value)
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        return numerator.divide(denominator.replace(0, np.nan)).fillna(fill_value)
    if isinstance(numerator, np.ndarray) and np.isscalar(denominator):
        den = np.nan if denominator in (0, None) or pd.isna(denominator) else denominator
        result = np.divide(numerator, den)
        return np.nan_to_num(result, nan=fill_value)
    if isinstance(numerator, np.ndarray) and isinstance(denominator, np.ndarray):
        den = denominator.astype("float64", copy=True)
        den[den == 0] = np.nan
        result = np.divide(numerator, den)
        return np.nan_to_num(result, nan=fill_value)
    if denominator in (0, None) or pd.isna(denominator):
        return fill_value
    return numerator / denominator


def load_data(data_path: str = "../data/raw") -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for file_name in TABLE_ORDER:
        key = file_name.replace(".csv", "")
        parse_dates = DATE_COLUMNS.get(key)
        path = os.path.join(data_path, file_name)
        if parse_dates:
            tables[key] = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
        else:
            tables[key] = pd.read_csv(path, low_memory=False)
    return tables


def preview_table_bundle(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "cols": df.shape[1],
                "date_min": _table_date_min(df),
                "date_max": _table_date_max(df),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows", "table"], ascending=[False, True]).reset_index(drop=True)


def _table_date_min(df: pd.DataFrame) -> Any:
    candidates = [c for c in df.columns if "date" in c.lower() or c == "Date"]
    mins = []
    for col in candidates:
        series = pd.to_datetime(df[col], errors="coerce").dropna()
        if not series.empty:
            mins.append(series.min())
    return min(mins).date().isoformat() if mins else None


def _table_date_max(df: pd.DataFrame) -> Any:
    candidates = [c for c in df.columns if "date" in c.lower() or c == "Date"]
    maxs = []
    for col in candidates:
        series = pd.to_datetime(df[col], errors="coerce").dropna()
        if not series.empty:
            maxs.append(series.max())
    return max(maxs).date().isoformat() if maxs else None


def build_quality_audit(data_path: str = "../data/raw") -> pd.DataFrame:
    rows = []
    for file_name in TABLE_ORDER:
        _df, profile = profile_table(file_name, data_path=data_path)
        row = build_raw_audit_row(profile)
        row["PK unique"] = profile["pk_unique"]
        row["Memory (MB)"] = profile["memory_mb"]
        row["Date range"] = _date_range_text(profile)
        rows.append(row)
    return pd.DataFrame(rows)


def _date_range_text(profile: dict[str, Any]) -> str:
    if not profile["date_ranges"]:
        return "n/a"
    fragments = []
    for col, stats_dict in profile["date_ranges"].items():
        fragments.append(f"{col}: {stats_dict['min']} -> {stats_dict['max']}")
    return " | ".join(fragments[:2])


def build_relation_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    relations = [
        ("orders", "order_id", "payments", "order_id", "orders -> payments"),
        ("orders", "order_id", "shipments", "order_id", "orders -> shipments"),
        ("order_items", "product_id", "products", "product_id", "order_items -> products"),
        ("returns", "order_id", "orders", "order_id", "returns -> orders"),
        ("reviews", "order_id", "orders", "order_id", "reviews -> orders"),
    ]
    rows = []
    for left_name, left_key, right_name, right_key, relation in relations:
        left_set = set(data[left_name][left_key].dropna().astype(str).unique())
        right_set = set(data[right_name][right_key].dropna().astype(str).unique())
        intersection = len(left_set & right_set)
        rows.append(
            {
                "relation": relation,
                "left_unique": len(left_set),
                "right_unique": len(right_set),
                "intersection": intersection,
                "left_only": len(left_set - right_set),
                "right_only": len(right_set - left_set),
                "coverage_from_left": safe_divide(intersection, len(left_set)),
                "coverage_from_right": safe_divide(intersection, len(right_set)),
            }
        )
    relation_df = pd.DataFrame(rows)
    relation_df["coverage_from_left"] = relation_df["coverage_from_left"].round(4)
    relation_df["coverage_from_right"] = relation_df["coverage_from_right"].round(4)
    return relation_df


def build_fact_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = data["orders"].copy()
    order_items = data["order_items"].copy()
    products = data["products"].copy()
    promotions = data["promotions"].copy()
    customers = data["customers"].copy()
    geography = data["geography"].copy()
    shipments = data["shipments"].copy()
    payments = data["payments"].copy()
    returns = data["returns"].copy()

    promotions["promo_id"] = promotions["promo_id"].astype(str)
    promotions["promo_type"] = promotions["promo_type"].fillna("no_promo")
    promotions["promo_channel"] = promotions["promo_channel"].fillna("unknown")
    promotions["applicable_category"] = promotions["applicable_category"].replace("", "all_category").fillna("all_category")

    payments_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            installments=("installments", "max"),
            payment_method_payment=("payment_method", _first_valid),
        )
    )

    returns["refund_amount"] = returns["refund_amount"].fillna(0)
    returns["return_quantity"] = returns["return_quantity"].fillna(0)
    returns_agg_line = (
        returns.groupby(["order_id", "product_id"], as_index=False)
        .agg(
            return_quantity=("return_quantity", "sum"),
            refund_amount=("refund_amount", "sum"),
            return_records=("return_id", "nunique"),
            primary_return_reason=("return_reason", _first_mode),
        )
    )
    reason_matrix = (
        returns.assign(flag=1)
        .pivot_table(
            index=["order_id", "product_id"],
            columns="return_reason",
            values="flag",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    returns_agg_line = returns_agg_line.merge(reason_matrix, on=["order_id", "product_id"], how="left")
    returns_agg_line["has_return_line"] = 1

    order_items["discount_amount"] = order_items["discount_amount"].fillna(0)
    order_items["promo_id"] = order_items["promo_id"].fillna("NO_PROMO").astype(str)
    order_items["promo_id_2"] = order_items["promo_id_2"].fillna("")
    order_items["stacked_promo_flag"] = order_items["promo_id_2"].ne("")

    fact_line = (
        order_items.merge(products, on="product_id", how="left")
        .merge(orders, on="order_id", how="left", suffixes=("", "_order"))
        .merge(geography, on="zip", how="left", suffixes=("", "_geo"))
        .merge(promotions, on="promo_id", how="left")
        .merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
        .merge(shipments, on="order_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
        .merge(returns_agg_line, on=["order_id", "product_id"], how="left")
    )

    fact_line["return_quantity"] = fact_line["return_quantity"].fillna(0)
    fact_line["refund_amount"] = fact_line["refund_amount"].fillna(0)
    fact_line["has_return_line"] = fact_line["has_return_line"].fillna(0).astype(int)

    price_validation = fact_line[["unit_price", "price", "discount_amount", "quantity"]].copy()
    valid_mask = (
        price_validation["price"].notna()
        & price_validation["quantity"].notna()
        & price_validation["quantity"].gt(0)
        & price_validation["unit_price"].notna()
    )
    pricing_mode = "pre_discount"
    if valid_mask.any():
        check = price_validation.loc[valid_mask].copy()
        check["implied_pre"] = check["unit_price"] + safe_divide(check["discount_amount"], check["quantity"])
        unit_gap = (check["unit_price"] - check["price"]).abs().median()
        implied_gap = (check["implied_pre"] - check["price"]).abs().median()
        if implied_gap + 1e-9 < unit_gap:
            pricing_mode = "post_discount"

    if pricing_mode == "post_discount":
        fact_line["gmv"] = fact_line["quantity"] * fact_line["unit_price"] + fact_line["discount_amount"]
    else:
        fact_line["gmv"] = fact_line["quantity"] * fact_line["unit_price"]

    fact_line["discount"] = fact_line["discount_amount"]
    fact_line["net_before_outcome"] = fact_line["gmv"] - fact_line["discount"]
    fact_line["cogs_total"] = fact_line["quantity"] * fact_line["cogs"]
    fact_line["promo_flag"] = fact_line["promo_id"].ne("NO_PROMO")
    fact_line["discount_rate"] = safe_divide(fact_line["discount"], fact_line["gmv"])
    fact_line["delivery_days"] = (fact_line["delivery_date"] - fact_line["ship_date"]).dt.days
    fact_line["is_cancelled_line"] = fact_line["order_status"].eq("cancelled").astype(int)
    fact_line["cancel_leakage"] = np.where(fact_line["order_status"].eq("cancelled"), fact_line["net_before_outcome"], 0)
    fact_line["return_leakage"] = fact_line["refund_amount"]
    fact_line["is_resolved"] = fact_line["order_status"].isin(RESOLVED_STATUS).astype(int)
    fact_line["is_in_flight"] = fact_line["order_status"].isin(IN_FLIGHT_STATUS).astype(int)
    fact_line["resolved_net_revenue"] = np.where(
        fact_line["is_resolved"].eq(1),
        fact_line["net_before_outcome"] - fact_line["cancel_leakage"] - fact_line["refund_amount"],
        np.nan,
    )
    fact_line["gross_margin_proxy"] = np.where(
        fact_line["is_resolved"].eq(1),
        fact_line["net_before_outcome"] - fact_line["cogs_total"] - fact_line["refund_amount"],
        np.nan,
    )
    for column in [
        "category",
        "segment",
        "size",
        "color",
        "payment_method",
        "device_type",
        "order_source",
        "gender",
        "age_group",
        "acquisition_channel",
        "region",
        "district",
        "promo_type",
        "promo_channel",
        "applicable_category",
        "primary_return_reason",
    ]:
        if column in fact_line.columns:
            fact_line[column] = fact_line[column].fillna("unknown")

    fact_line["order_month"] = fact_line["order_date"].dt.to_period("M").dt.to_timestamp()
    fact_line["order_quarter"] = fact_line["order_date"].dt.to_period("Q").dt.to_timestamp()
    fact_line["order_year"] = fact_line["order_date"].dt.year

    base_agg = (
        fact_line.groupby("order_id", as_index=False)
        .agg(
            order_date=("order_date", "first"),
            customer_id=("customer_id", "first"),
            zip=("zip", "first"),
            order_status=("order_status", "first"),
            payment_method=("payment_method", "first"),
            device_type=("device_type", "first"),
            order_source=("order_source", "first"),
            region=("region", "first"),
            district=("district", "first"),
            city=("city", "first"),
            gender=("gender", "first"),
            age_group=("age_group", "first"),
            acquisition_channel=("acquisition_channel", "first"),
            signup_date=("signup_date", "first"),
            payment_value=("payment_value", "sum"),
            installments=("installments", "max"),
            order_gmv=("gmv", "sum"),
            order_discount=("discount", "sum"),
            order_net_before_outcome=("net_before_outcome", "sum"),
            order_cogs=("cogs_total", "sum"),
            refund_amount=("refund_amount", "sum"),
            return_quantity=("return_quantity", "sum"),
            product_count=("product_id", "nunique"),
            category_breadth=("category", "nunique"),
            item_quantity=("quantity", "sum"),
            promo_flag=("promo_flag", "max"),
            stacked_promo_flag=("stacked_promo_flag", "max"),
            has_return_order=("has_return_line", "max"),
            wrong_size_return_orders=("wrong_size", "max"),
            late_delivery_return_orders=("late_delivery", "max"),
            ship_date=("ship_date", "first"),
            delivery_date=("delivery_date", "first"),
            delivery_days=("delivery_days", "first"),
            gross_margin_proxy=("gross_margin_proxy", "sum"),
            resolved_line_revenue=("resolved_net_revenue", "sum"),
        )
    )

    dominant_category = _dominant_feature(fact_line, "category", weight_col="net_before_outcome", name="dominant_category")
    dominant_segment = _dominant_feature(fact_line, "segment", weight_col="net_before_outcome", name="dominant_segment")
    dominant_size = _dominant_feature(fact_line, "size", weight_col="quantity", name="dominant_size")
    dominant_promo_type = _dominant_feature(
        fact_line.assign(promo_type=np.where(fact_line["promo_flag"], fact_line["promo_type"], "no_promo")),
        "promo_type",
        weight_col="discount",
        name="dominant_promo_type",
    )

    fact_order = (
        base_agg.merge(dominant_category, on="order_id", how="left")
        .merge(dominant_segment, on="order_id", how="left")
        .merge(dominant_size, on="order_id", how="left")
        .merge(dominant_promo_type, on="order_id", how="left")
    )
    fact_order["order_month"] = fact_order["order_date"].dt.to_period("M").dt.to_timestamp()
    fact_order["order_quarter"] = fact_order["order_date"].dt.to_period("Q").dt.to_timestamp()
    fact_order["order_year"] = fact_order["order_date"].dt.year
    fact_order["order_weekday"] = fact_order["order_date"].dt.day_name()
    fact_order["order_month_name"] = fact_order["order_date"].dt.month_name()
    fact_order["is_weekend"] = fact_order["order_date"].dt.dayofweek.isin([5, 6]).astype(int)
    fact_order["is_resolved"] = fact_order["order_status"].isin(RESOLVED_STATUS)
    fact_order["is_in_flight"] = fact_order["order_status"].isin(IN_FLIGHT_STATUS)
    fact_order["is_cancelled_order"] = fact_order["order_status"].eq("cancelled")
    fact_order["has_return_order"] = fact_order["has_return_order"].astype(bool) | fact_order["order_status"].eq("returned")
    fact_order["is_leakage_order"] = fact_order["is_cancelled_order"] | fact_order["has_return_order"]
    fact_order["cancel_leakage"] = np.where(fact_order["is_cancelled_order"], fact_order["order_net_before_outcome"], 0)
    fact_order["return_leakage"] = fact_order["refund_amount"]
    fact_order["realized_net_revenue"] = np.where(
        fact_order["is_resolved"],
        fact_order["order_net_before_outcome"] - fact_order["cancel_leakage"] - fact_order["refund_amount"],
        np.nan,
    )
    fact_order["discount_rate"] = safe_divide(fact_order["order_discount"], fact_order["order_gmv"])
    fact_order["avg_item_price"] = safe_divide(fact_order["order_gmv"], fact_order["item_quantity"])
    fact_order["days_since_signup"] = (fact_order["order_date"] - fact_order["signup_date"]).dt.days
    fact_order["gross_margin_rate"] = safe_divide(fact_order["gross_margin_proxy"], fact_order["order_net_before_outcome"])
    fact_order["wrong_size_return_orders"] = fact_order["wrong_size_return_orders"].fillna(0).astype(int)
    fact_order["late_delivery_return_orders"] = fact_order["late_delivery_return_orders"].fillna(0).astype(int)
    fact_order = add_customer_history_features(fact_order)
    return fact_line, fact_order


def _first_valid(series: pd.Series) -> Any:
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def _first_mode(series: pd.Series) -> Any:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    mode = valid.mode()
    return mode.iloc[0] if not mode.empty else valid.iloc[0]


def _dominant_feature(df: pd.DataFrame, feature_col: str, weight_col: str, name: str) -> pd.DataFrame:
    temp = df[["order_id", feature_col, weight_col]].copy()
    temp[weight_col] = temp[weight_col].fillna(0)
    ranked = (
        temp.groupby(["order_id", feature_col], as_index=False)[weight_col]
        .sum()
        .sort_values(["order_id", weight_col, feature_col], ascending=[True, False, True])
    )
    ranked = ranked.drop_duplicates(subset=["order_id"], keep="first")
    return ranked[["order_id", feature_col]].rename(columns={feature_col: name})


def add_customer_history_features(fact_order: pd.DataFrame) -> pd.DataFrame:
    fact_order = fact_order.sort_values(["customer_id", "order_date", "order_id"]).copy()
    grouped = fact_order.groupby("customer_id", sort=False)
    fact_order["prior_total_orders"] = grouped.cumcount()
    fact_order["prior_resolved_orders"] = grouped["is_resolved"].cumsum().groupby(fact_order["customer_id"]).shift(fill_value=0)
    fact_order["prior_leakage_orders"] = grouped["is_leakage_order"].cumsum().groupby(fact_order["customer_id"]).shift(fill_value=0)
    fact_order["prior_realized_revenue"] = (
        grouped["realized_net_revenue"].cumsum().groupby(fact_order["customer_id"]).shift(fill_value=0)
    )
    fact_order["prior_avg_order_value"] = safe_divide(
        fact_order["prior_realized_revenue"], fact_order["prior_resolved_orders"].replace(0, np.nan)
    )
    fact_order["prior_leakage_rate"] = safe_divide(
        fact_order["prior_leakage_orders"], fact_order["prior_resolved_orders"].replace(0, np.nan)
    )
    fact_order["days_since_prev_order"] = grouped["order_date"].diff().dt.days
    fact_order["days_since_prev_order"] = fact_order["days_since_prev_order"].fillna(-1)
    fact_order["days_since_signup"] = fact_order["days_since_signup"].fillna(-1)
    return fact_order


def build_monthly_kpis(fact_order: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        fact_order.groupby("order_month", as_index=False)
        .agg(
            gmv=("order_gmv", "sum"),
            booked_net_revenue=("order_net_before_outcome", "sum"),
            realized_net_revenue=("realized_net_revenue", "sum"),
            booked_cogs=("order_cogs", "sum"),
            resolved_orders=("is_resolved", "sum"),
            in_flight_orders=("is_in_flight", "sum"),
            leakage_orders=("is_leakage_order", "sum"),
            return_orders=("has_return_order", "sum"),
            cancelled_orders=("is_cancelled_order", "sum"),
            total_discount=("order_discount", "sum"),
            cancel_leakage=("cancel_leakage", "sum"),
            return_leakage=("return_leakage", "sum"),
        )
    )
    monthly["aov"] = safe_divide(monthly["realized_net_revenue"], monthly["resolved_orders"])
    monthly["return_rate"] = safe_divide(monthly["return_orders"], monthly["resolved_orders"])
    monthly["cancel_rate"] = safe_divide(monthly["cancelled_orders"], monthly["resolved_orders"])
    monthly["leakage_rate"] = safe_divide(monthly["leakage_orders"], monthly["resolved_orders"])
    monthly["quarter"] = monthly["order_month"].dt.to_period("Q").dt.to_timestamp()
    return monthly


def build_quarterly_kpis(monthly_kpis: pd.DataFrame) -> pd.DataFrame:
    quarterly = (
        monthly_kpis.groupby("quarter", as_index=False)
        .agg(
            gmv=("gmv", "sum"),
            booked_net_revenue=("booked_net_revenue", "sum"),
            realized_net_revenue=("realized_net_revenue", "sum"),
            booked_cogs=("booked_cogs", "sum"),
            resolved_orders=("resolved_orders", "sum"),
            in_flight_orders=("in_flight_orders", "sum"),
            leakage_orders=("leakage_orders", "sum"),
            return_orders=("return_orders", "sum"),
            cancelled_orders=("cancelled_orders", "sum"),
            total_discount=("total_discount", "sum"),
            cancel_leakage=("cancel_leakage", "sum"),
            return_leakage=("return_leakage", "sum"),
        )
    )
    quarterly["aov"] = safe_divide(quarterly["realized_net_revenue"], quarterly["resolved_orders"])
    quarterly["return_rate"] = safe_divide(quarterly["return_orders"], quarterly["resolved_orders"])
    quarterly["cancel_rate"] = safe_divide(quarterly["cancelled_orders"], quarterly["resolved_orders"])
    quarterly["leakage_rate"] = safe_divide(quarterly["leakage_orders"], quarterly["resolved_orders"])
    return quarterly


def build_waterfall_summary(fact_order: pd.DataFrame) -> pd.DataFrame:
    resolved = fact_order.loc[fact_order["is_resolved"]].copy()
    gross = resolved["order_gmv"].sum()
    discount = resolved["order_discount"].sum()
    cancel = resolved["cancel_leakage"].sum()
    refund = resolved["return_leakage"].sum()
    realized = resolved["realized_net_revenue"].sum()
    return pd.DataFrame(
        {
            "stage": [
                "GMV",
                "Discounts",
                "Cancellation leakage",
                "Return leakage",
                "Realized net revenue",
            ],
            "value": [gross, -discount, -cancel, -refund, realized],
        }
    )


def build_dimension_mix(fact_order: pd.DataFrame, dimension: str, top_n: int = 8, recent_year: int = 2022) -> pd.DataFrame:
    recent = fact_order.loc[(fact_order["order_year"] == recent_year) & fact_order["is_resolved"]].copy()
    grouped = (
        recent.groupby(dimension, as_index=False)
        .agg(
            realized_net_revenue=("realized_net_revenue", "sum"),
            resolved_orders=("order_id", "nunique"),
            leakage_orders=("is_leakage_order", "sum"),
            gmv=("order_gmv", "sum"),
        )
        .sort_values("realized_net_revenue", ascending=False)
    )
    grouped["share_of_net_revenue"] = safe_divide(grouped["realized_net_revenue"], grouped["realized_net_revenue"].sum())
    grouped["share_of_orders"] = safe_divide(grouped["resolved_orders"], grouped["resolved_orders"].sum())
    grouped["leakage_rate"] = safe_divide(grouped["leakage_orders"], grouped["resolved_orders"])
    grouped["avg_realized_order_value"] = safe_divide(grouped["realized_net_revenue"], grouped["resolved_orders"])
    return grouped.head(top_n).reset_index(drop=True)


def build_geography_snapshot(fact_order: pd.DataFrame, recent_year: int = 2022) -> pd.DataFrame:
    geo = (
        fact_order.loc[(fact_order["order_year"] == recent_year) & fact_order["is_resolved"]]
        .groupby("region", as_index=False)
        .agg(
            realized_net_revenue=("realized_net_revenue", "sum"),
            resolved_orders=("order_id", "nunique"),
            return_orders=("has_return_order", "sum"),
            cancelled_orders=("is_cancelled_order", "sum"),
            avg_delivery_days=("delivery_days", "mean"),
        )
    )
    geo["return_rate"] = safe_divide(geo["return_orders"], geo["resolved_orders"])
    geo["cancel_rate"] = safe_divide(geo["cancelled_orders"], geo["resolved_orders"])
    return geo.sort_values("realized_net_revenue", ascending=False)


def reconcile_with_sales(monthly_kpis: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    sales = sales_df.copy()
    sales["order_month"] = sales["Date"].dt.to_period("M").dt.to_timestamp()
    sales_monthly = sales.groupby("order_month", as_index=False).agg(
        sales_revenue=("Revenue", "sum"),
        sales_cogs=("COGS", "sum"),
    )
    recon = monthly_kpis.merge(sales_monthly, on="order_month", how="left")
    recon["gap_vs_sales"] = recon["sales_revenue"] - recon["booked_net_revenue"]
    recon["alignment_ratio"] = safe_divide(recon["booked_net_revenue"], recon["sales_revenue"])
    recon["cogs_gap_vs_sales"] = recon["sales_cogs"] - recon["booked_cogs"]
    recon["cogs_alignment_ratio"] = safe_divide(recon["booked_cogs"], recon["sales_cogs"])
    recon["realized_gap_vs_sales"] = recon["sales_revenue"] - recon["realized_net_revenue"]
    recon["realized_vs_sales_ratio"] = safe_divide(recon["realized_net_revenue"], recon["sales_revenue"])
    recon["outcome_leakage_ratio"] = safe_divide(recon["realized_gap_vs_sales"], recon["sales_revenue"])
    return recon


def build_descriptive_scorecard(fact_order: pd.DataFrame, recent_year: int | None = None) -> pd.DataFrame:
    if recent_year is None:
        recent_year = int(fact_order["order_year"].max())
    current = fact_order.loc[fact_order["order_year"].eq(recent_year)].copy()
    prev = fact_order.loc[fact_order["order_year"].eq(recent_year - 1)].copy()

    rows = []
    metric_map = [
        ("GMV", current["order_gmv"].sum(), prev["order_gmv"].sum()),
        ("Realized net revenue", current["realized_net_revenue"].sum(), prev["realized_net_revenue"].sum()),
        ("Cancellation leakage", current["cancel_leakage"].sum(), prev["cancel_leakage"].sum()),
        ("Return leakage", current["return_leakage"].sum(), prev["return_leakage"].sum()),
        (
            "Total leakage value",
            current["cancel_leakage"].sum() + current["return_leakage"].sum(),
            prev["cancel_leakage"].sum() + prev["return_leakage"].sum(),
        ),
    ]
    for metric, value, previous_value in metric_map:
        rows.append(
            {
                "metric": metric,
                "year": recent_year,
                "value": float(value),
                "previous_value": float(previous_value),
                "yoy_change": safe_divide(value - previous_value, previous_value),
            }
        )
    return pd.DataFrame(rows)


def _format_vnd_compact(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:,.2f} B VND"
    if abs_value >= 1e6:
        return f"{value / 1e6:,.2f} M VND"
    if abs_value >= 1e3:
        return f"{value / 1e3:,.1f} K VND"
    return f"{value:,.0f} VND"


def build_descriptive_summary(monthly_kpis: pd.DataFrame, fact_order: pd.DataFrame, geo_df: pd.DataFrame) -> list[str]:
    latest_year = int(monthly_kpis["order_month"].dt.year.max())
    current = monthly_kpis.loc[monthly_kpis["order_month"].dt.year.eq(latest_year)]
    prev = monthly_kpis.loc[monthly_kpis["order_month"].dt.year.eq(latest_year - 1)]

    current_net = current["realized_net_revenue"].sum()
    prev_net = prev["realized_net_revenue"].sum()
    yoy = safe_divide(current_net - prev_net, prev_net)
    current_leakage = current["cancel_leakage"].sum() + current["return_leakage"].sum()
    prev_leakage = prev["cancel_leakage"].sum() + prev["return_leakage"].sum()
    leakage_yoy = safe_divide(current_leakage - prev_leakage, prev_leakage)

    top_month = current.loc[current["realized_net_revenue"].idxmax()]
    worst_leakage_month = current.loc[current["leakage_rate"].idxmax()]
    top_category = build_dimension_mix(fact_order, "dominant_category", top_n=1, recent_year=latest_year).iloc[0]
    top_region = geo_df.sort_values("realized_net_revenue", ascending=False).iloc[0]
    top_region_share = safe_divide(top_region["realized_net_revenue"], geo_df["realized_net_revenue"].sum())
    resolved_share = safe_divide(
        current["resolved_orders"].sum(),
        current["resolved_orders"].sum() + current["in_flight_orders"].sum(),
    )

    return [
        (
            f"In {latest_year}, realized net revenue reached {_format_vnd_compact(float(current_net))}, "
            f"changing {yoy:+.1%} YoY, while total leakage value reached {_format_vnd_compact(float(current_leakage))} "
            f"({leakage_yoy:+.1%} YoY)."
        ),
        (
            f"The strongest month was {top_month['order_month']:%m/%Y} with "
            f"{_format_vnd_compact(float(top_month['realized_net_revenue']))} in realized net revenue, "
            f"while the month with the highest leakage pressure was {worst_leakage_month['order_month']:%m/%Y} at "
            f"{worst_leakage_month['leakage_rate']:.1%}."
        ),
        (
            f"{top_category['dominant_category']} led {top_category['share_of_net_revenue']:.1%} of realized net revenue "
            f"and {top_category['share_of_orders']:.1%} of resolved volume in {latest_year}; "
            f"{top_region['region']} contributed {top_region_share:.1%} of realized net revenue, "
            f"while {resolved_share:.1%} of the year's flow had already reached a final outcome."
        ),
    ]


__all__ = [
    "build_descriptive_scorecard",
    "build_descriptive_summary",
    "build_dimension_mix",
    "build_fact_tables",
    "build_geography_snapshot",
    "build_monthly_kpis",
    "build_quality_audit",
    "build_quarterly_kpis",
    "build_relation_summary",
    "build_waterfall_summary",
    "load_data",
    "preview_table_bundle",
    "reconcile_with_sales",
    "set_plot_theme",
]
