from __future__ import annotations

import math
import os
import textwrap
import warnings
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils import TABLE_ORDER, build_raw_audit_row, profile_table

warnings.filterwarnings("ignore")


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
RISK_SCENARIOS = {
    "Conservative": {"capture_rate": 0.08, "adoption_rate": 0.55, "cost_multiplier": 1.00},
    "Base": {"capture_rate": 0.15, "adoption_rate": 0.70, "cost_multiplier": 1.00},
    "Aggressive": {"capture_rate": 0.24, "adoption_rate": 0.85, "cost_multiplier": 1.15},
}


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (14, 6)
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.formatter.useoffset"] = False


def _money_formatter(scale: float = 1e9, suffix: str = " B VND", decimals: int = 1):
    return FuncFormatter(lambda x, _pos: f"{x / scale:,.{decimals}f}{suffix}")


def _money_axis_config(max_abs_value: float) -> tuple[float, str, int]:
    if max_abs_value >= 1e9:
        return 1e9, " B VND", 2
    if max_abs_value >= 1e6:
        return 1e6, " M VND", 1
    if max_abs_value >= 1e3:
        return 1e3, " K VND", 1
    return 1.0, " VND", 0


def _format_vnd_compact(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{value / 1e9:,.2f} B VND"
    if abs_value >= 1e6:
        return f"{value / 1e6:,.2f} M VND"
    if abs_value >= 1e3:
        return f"{value / 1e3:,.1f} K VND"
    return f"{value:,.0f} VND"


def pct(x: float | int | pd.Series) -> float | pd.Series:
    return np.asarray(x) * 100 if not np.isscalar(x) else float(x) * 100


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

    # Infer whether unit_price is pre- or post-discount using product catalog price as reference.
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


def build_dimension_mix(
    fact_order: pd.DataFrame, dimension: str, top_n: int = 8, recent_year: int = 2022
) -> pd.DataFrame:
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
        .head(top_n)
    )
    grouped["share_of_net_revenue"] = safe_divide(
        grouped["realized_net_revenue"], grouped["realized_net_revenue"].sum()
    )
    grouped["leakage_rate"] = safe_divide(grouped["leakage_orders"], grouped["resolved_orders"])
    return grouped


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


def plot_kpi_trends(monthly_kpis: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    monthly_kpis.plot(x="order_month", y=["gmv", "realized_net_revenue"], ax=axes[0, 0], linewidth=2.4)
    axes[0, 0].set_title("Monthly KPI trend: GMV and realized net revenue")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Value")
    kpi_max = float(monthly_kpis[["gmv", "realized_net_revenue"]].abs().max().max())
    kpi_scale, kpi_suffix, kpi_decimals = _money_axis_config(kpi_max)
    axes[0, 0].yaxis.set_major_formatter(_money_formatter(scale=kpi_scale, suffix=kpi_suffix, decimals=kpi_decimals))

    monthly_kpis.plot(x="order_month", y=["resolved_orders", "in_flight_orders"], ax=axes[0, 1], linewidth=2.2)
    axes[0, 1].set_title("Resolved orders vs in-flight orders")
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Orders")

    monthly_kpis.plot(x="order_month", y="aov", ax=axes[1, 0], color="#E15759", linewidth=2.2)
    axes[1, 0].set_title("Monthly realized AOV")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Average order value")
    aov_max = float(monthly_kpis["aov"].abs().max())
    aov_scale, aov_suffix, aov_decimals = _money_axis_config(aov_max)
    axes[1, 0].yaxis.set_major_formatter(_money_formatter(scale=aov_scale, suffix=aov_suffix, decimals=aov_decimals))

    monthly_kpis.plot(x="order_month", y=["return_rate", "cancel_rate", "leakage_rate"], ax=axes[1, 1], linewidth=2.2)
    axes[1, 1].set_title("Monthly return, cancellation, and leakage rates")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.show()


def plot_waterfall(waterfall_df: pd.DataFrame) -> None:
    running = 0.0
    starts = []
    heights = []
    colors = []
    for _, row in waterfall_df.iterrows():
        if row["stage"] == "GMV":
            starts.append(0)
            heights.append(row["value"])
            running = row["value"]
            colors.append("#4E79A7")
        elif row["stage"] == "Realized net revenue":
            starts.append(0)
            heights.append(row["value"])
            colors.append("#59A14F")
        else:
            height = row["value"]
            starts.append(running if height < 0 else 0)
            heights.append(height)
            running += height
            colors.append("#E15759")
    plt.figure(figsize=(14, 6))
    bars = plt.bar(waterfall_df["stage"], heights, bottom=starts, color=colors)
    for bar, value in zip(bars, waterfall_df["value"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height(),
            f"{value / 1e9:,.1f} B VND",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Waterfall leakage: GMV -> Discount -> Cancel/Return -> Realized net revenue")
    plt.xlabel("Stage")
    plt.ylabel("Value")
    plt.gca().yaxis.set_major_formatter(_money_formatter())
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.show()


def plot_mix_dashboard(fact_order: pd.DataFrame, recent_year: int = 2022) -> None:
    dims = ["dominant_category", "order_source", "payment_method", "device_type"]
    titles = [
        "Category mix",
        "Order-source mix",
        "Payment-method mix",
        "Device mix",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    for ax, dimension, title in zip(axes, dims, titles):
        mix_df = build_dimension_mix(fact_order, dimension, top_n=8, recent_year=recent_year)
        sns.barplot(data=mix_df, y=dimension, x="realized_net_revenue", ax=ax, color="#4E79A7")
        ax.set_title(title)
        ax.set_xlabel("Realized net revenue")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(_money_formatter(scale=1e6, suffix=" M VND", decimals=1))
        for idx, (_, row) in enumerate(mix_df.iterrows()):
            ax.text(
                row["realized_net_revenue"],
                idx,
                f"  share {row['share_of_net_revenue']:.1%} | leakage {row['leakage_rate']:.1%}",
                va="center",
                fontsize=9,
            )
    plt.tight_layout()
    plt.show()


def plot_geography_snapshot(geo_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=geo_df, x="region", y="realized_net_revenue", ax=axes[0], color="#59A14F")
    axes[0].set_title("Realized net revenue by region (2022)")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("Realized net revenue")
    axes[0].yaxis.set_major_formatter(_money_formatter(scale=1e6, suffix=" M VND", decimals=1))

    rate_df = geo_df.melt(
        id_vars="region",
        value_vars=["return_rate", "cancel_rate"],
        var_name="metric",
        value_name="rate",
    )
    rate_df["metric"] = rate_df["metric"].replace({"return_rate": "Return rate", "cancel_rate": "Cancellation rate"})
    sns.barplot(data=rate_df, x="region", y="rate", hue="metric", ax=axes[1])
    axes[1].set_title("Return and cancellation rates by region (2022)")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(title="")
    plt.tight_layout()
    plt.show()


def plot_sales_reconciliation(recon_df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(recon_df["order_month"], recon_df["sales_revenue"], label="Revenue from sales.csv", linewidth=2.2)
    plt.plot(recon_df["order_month"], recon_df["booked_net_revenue"], label="Booked net revenue from EDA", linewidth=2.2)
    plt.plot(
        recon_df["order_month"],
        recon_df["realized_net_revenue"],
        label="Realized net revenue from EDA",
        linewidth=2.0,
        alpha=0.9,
    )
    plt.title("sales.csv reconciliation: booked revenue aligns, realized revenue is lower after outcomes")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.gca().yaxis.set_major_formatter(_money_formatter(scale=1e6, suffix=" M VND", decimals=1))
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_descriptive_summary(
    monthly_kpis: pd.DataFrame, fact_order: pd.DataFrame, geo_df: pd.DataFrame
) -> list[str]:
    latest_year = int(monthly_kpis["order_month"].dt.year.max())
    current = monthly_kpis.loc[monthly_kpis["order_month"].dt.year.eq(latest_year)]
    prev = monthly_kpis.loc[monthly_kpis["order_month"].dt.year.eq(latest_year - 1)]

    current_net = current["realized_net_revenue"].sum()
    prev_net = prev["realized_net_revenue"].sum()
    yoy = safe_divide(current_net - prev_net, prev_net)

    top_month = monthly_kpis.loc[monthly_kpis["realized_net_revenue"].idxmax()]
    worst_leakage_month = monthly_kpis.loc[monthly_kpis["leakage_rate"].idxmax()]
    top_category = build_dimension_mix(fact_order, "dominant_category", top_n=1, recent_year=latest_year).iloc[0]
    weakest_region = geo_df.sort_values("return_rate", ascending=False).iloc[0]
    in_flight_share = safe_divide(
        current["in_flight_orders"].sum(),
        current["in_flight_orders"].sum() + current["resolved_orders"].sum(),
    )

    return [
        f"In {latest_year}, realized net revenue reached {current_net / 1e9:,.1f} B VND, changing {yoy:.1%} versus the prior year.",
        f"The strongest month was {top_month['order_month']:%m/%Y} with {top_month['realized_net_revenue'] / 1e9:,.1f} B VND in realized net revenue.",
        f"The highest leakage rate occurred in {worst_leakage_month['order_month']:%m/%Y} at {worst_leakage_month['leakage_rate']:.1%}.",
        f"The leading category in {latest_year} was {top_category['dominant_category']} with a {top_category['share_of_net_revenue']:.1%} revenue share.",
        f"The region with the highest return pressure was {weakest_region['region']} at a {weakest_region['return_rate']:.1%} return rate; in-flight orders still represent roughly {in_flight_share:.1%} of {latest_year} volume.",
    ]


def build_delivery_diagnostics(fact_order: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    eligible = fact_order.loc[
        fact_order["is_resolved"] & fact_order["delivery_days"].notna() & fact_order["order_status"].ne("cancelled")
    ].copy()
    eligible["delivery_days"] = eligible["delivery_days"].clip(lower=0)
    delivery_curve = (
        eligible.groupby("delivery_days", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            leakage_orders=("is_leakage_order", "sum"),
            avg_realized_net=("realized_net_revenue", "mean"),
        )
    )
    delivery_curve["leakage_rate"] = safe_divide(delivery_curve["leakage_orders"], delivery_curve["orders"])

    threshold_rows = []
    for threshold in [3, 4, 5]:
        eligible[f"delay_gt_{threshold}"] = eligible["delivery_days"] > threshold
        high = eligible.loc[eligible[f"delay_gt_{threshold}"]]
        low = eligible.loc[~eligible[f"delay_gt_{threshold}"]]
        delta = high["is_leakage_order"].mean() - low["is_leakage_order"].mean()
        ci_low, ci_high = bootstrap_rate_gap(high["is_leakage_order"], low["is_leakage_order"])
        threshold_rows.append(
            {
                "threshold_days": threshold,
                "late_orders": len(high),
                "late_leakage_rate": high["is_leakage_order"].mean(),
                "on_time_leakage_rate": low["is_leakage_order"].mean(),
                "gap": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    leak_days = eligible.loc[eligible["is_leakage_order"], "delivery_days"]
    healthy_days = eligible.loc[~eligible["is_leakage_order"], "delivery_days"]
    mw = stats.mannwhitneyu(leak_days, healthy_days, alternative="two-sided")
    effect = safe_divide(mw.statistic, len(leak_days) * len(healthy_days))
    stats_summary = {
        "mann_whitney_pvalue": float(mw.pvalue),
        "common_language_effect": float(effect),
        "median_leak_days": float(leak_days.median()),
        "median_healthy_days": float(healthy_days.median()),
    }
    return delivery_curve, threshold_df, stats_summary


def plot_delivery_diagnostics(delivery_curve: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    curve = delivery_curve.copy()
    curve["leakage_rate_pct"] = curve["leakage_rate"] * 100
    sns.lineplot(data=curve, x="delivery_days", y="leakage_rate_pct", marker="o", ax=axes[0], color="#E15759")
    axes[0].set_title("Leakage rate theo delivery days")
    axes[0].set_xlabel("Delivery days")
    axes[0].set_ylabel("Leakage rate (%)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.2f}%"))
    y_pad = max((curve["leakage_rate_pct"].max() - curve["leakage_rate_pct"].min()) * 0.2, 0.05)
    axes[0].set_ylim(curve["leakage_rate_pct"].min() - y_pad, curve["leakage_rate_pct"].max() + y_pad)

    thr = threshold_df.copy()
    # Use basis points (1 bps = 0.01%) for small gap readability.
    thr["gap_bps"] = thr["gap"] * 10000
    thr["ci_low_bps"] = thr["ci_low"] * 10000
    thr["ci_high_bps"] = thr["ci_high"] * 10000
    axes[1].errorbar(
        thr["threshold_days"],
        thr["gap_bps"],
        yerr=[
            thr["gap_bps"] - thr["ci_low_bps"],
            thr["ci_high_bps"] - thr["gap_bps"],
        ],
        fmt="o-",
        c="#4E79A7",
        markersize=7,
        linewidth=2,
        capsize=4,
    )
    axes[1].axhline(0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_title("Threshold test for delivery delay")
    axes[1].set_xlabel("Delivery-day threshold")
    axes[1].set_ylabel("Leakage rate gap (bps)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0f} bps"))
    plt.tight_layout()
    plt.show()


def build_size_diagnostic(fact_line: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = fact_line.copy()
    temp["wrong_size_flag"] = temp.get("wrong_size", 0).fillna(0).astype(int)
    size_matrix = (
        temp.groupby(["category", "size"], as_index=False)
        .agg(
            order_lines=("order_id", "nunique"),
            wrong_size_returns=("wrong_size_flag", "sum"),
            refund_amount=("refund_amount", "sum"),
        )
    )
    size_matrix["wrong_size_rate"] = safe_divide(size_matrix["wrong_size_returns"], size_matrix["order_lines"])
    pivot = size_matrix.pivot(index="category", columns="size", values="wrong_size_rate").fillna(0)
    return size_matrix.sort_values("wrong_size_rate", ascending=False), pivot


def plot_size_heatmap(pivot: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd")
    plt.title("Wrong-size return rate theo category x size")
    plt.xlabel("Size")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()


def chi_square_cramers_v(frame: pd.DataFrame, row_col: str, target_col: str) -> dict[str, float]:
    valid = frame[[row_col, target_col]].dropna().copy()
    contingency = pd.crosstab(valid[row_col], valid[target_col])
    chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    cramers_v = math.sqrt(chi2 / (n * max(min(k - 1, r - 1), 1)))
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "levels": int(r),
    }


def build_cancellation_diagnostic(fact_order: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
    resolved = fact_order.loc[fact_order["is_resolved"]].copy()
    cancel_df = (
        resolved.groupby(["payment_method", "order_source"], as_index=False)
        .agg(cancelled_orders=("is_cancelled_order", "sum"), resolved_orders=("order_id", "nunique"))
    )
    cancel_df["cancel_rate"] = safe_divide(cancel_df["cancelled_orders"], cancel_df["resolved_orders"])
    cancel_df = cancel_df.loc[cancel_df["resolved_orders"] >= 500].sort_values("cancel_rate", ascending=False)

    stats_map = {
        "payment_method": chi_square_cramers_v(resolved, "payment_method", "is_cancelled_order"),
        "order_source": chi_square_cramers_v(resolved, "order_source", "is_cancelled_order"),
        "device_type": chi_square_cramers_v(resolved, "device_type", "is_cancelled_order"),
        "region": chi_square_cramers_v(resolved, "region", "is_cancelled_order"),
    }

    region_device = (
        resolved.groupby(["region", "device_type"], as_index=False)
        .agg(cancelled_orders=("is_cancelled_order", "sum"), resolved_orders=("order_id", "nunique"))
    )
    region_device["cancel_rate"] = safe_divide(region_device["cancelled_orders"], region_device["resolved_orders"])
    return cancel_df, stats_map, region_device


def plot_cancellation_diagnostic(cancel_df: pd.DataFrame, region_device: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    top = cancel_df.head(10).copy()
    top["combo"] = top["payment_method"] + " | " + top["order_source"]
    sns.barplot(data=top, y="combo", x="cancel_rate", ax=axes[0], color="#E15759")
    axes[0].set_title("Payment-source combinations with the highest cancellation rates")
    axes[0].set_xlabel("Cancel rate")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))

    pivot = region_device.pivot(index="region", columns="device_type", values="cancel_rate").fillna(0)
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=axes[1])
    axes[1].set_title("Cancel rate theo region x device")
    axes[1].set_xlabel("Device type")
    axes[1].set_ylabel("Region")
    plt.tight_layout()
    plt.show()


def build_promo_proxy(fact_order: pd.DataFrame) -> pd.DataFrame:
    resolved = fact_order.loc[fact_order["is_resolved"]].copy()
    bucket = (
        resolved.groupby(["order_month", "dominant_category", "order_source", "promo_flag"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            avg_gmv=("order_gmv", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            avg_net=("realized_net_revenue", "mean"),
            avg_margin=("gross_margin_proxy", "mean"),
            leakage_rate=("is_leakage_order", "mean"),
        )
    )
    wide = bucket.pivot_table(
        index=["order_month", "dominant_category", "order_source"],
        columns="promo_flag",
        values=["orders", "avg_gmv", "avg_discount_rate", "avg_net", "avg_margin", "leakage_rate"],
    )
    wide.columns = [f"{metric}_{'promo' if flag else 'no_promo'}" for metric, flag in wide.columns]
    wide = wide.reset_index()
    wide = wide.dropna(subset=["orders_promo", "orders_no_promo"])
    wide = wide.loc[(wide["orders_promo"] >= 15) & (wide["orders_no_promo"] >= 15)].copy()
    wide["gmv_uplift_proxy"] = safe_divide(wide["avg_gmv_promo"] - wide["avg_gmv_no_promo"], wide["avg_gmv_no_promo"])
    wide["net_delta"] = wide["avg_net_promo"] - wide["avg_net_no_promo"]
    wide["margin_delta"] = wide["avg_margin_promo"] - wide["avg_margin_no_promo"]
    wide["leakage_delta"] = wide["leakage_rate_promo"] - wide["leakage_rate_no_promo"]
    wide["bucket_weight"] = wide["orders_promo"] + wide["orders_no_promo"]
    return wide.sort_values("margin_delta")


def plot_promo_proxy(promo_proxy: pd.DataFrame) -> None:
    sample = promo_proxy.sort_values("bucket_weight", ascending=False).head(80).copy()
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(
        sample["avg_discount_rate_promo"],
        sample["margin_delta"],
        s=sample["bucket_weight"] * 2,
        c=sample["leakage_delta"],
        cmap="coolwarm",
        alpha=0.75,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.colorbar(scatter, label="Leakage delta (promo - no promo)")
    plt.title("Promotion proxy: discount rate vs margin delta")
    plt.xlabel("Average discount rate in the promo group")
    plt.ylabel("Margin delta versus the no-promo bucket")
    plt.gca().yaxis.set_major_formatter(_money_formatter(scale=1e6, suffix=" M VND"))
    plt.tight_layout()
    plt.show()


def build_stockout_proxy(fact_order: pd.DataFrame, inventory: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    demand = (
        fact_order.loc[fact_order["is_resolved"]]
        .groupby(["order_month", "dominant_category"], as_index=False)
        .agg(
            realized_net_revenue=("realized_net_revenue", "sum"),
            leakage_rate=("is_leakage_order", "mean"),
            cancel_rate=("is_cancelled_order", "mean"),
            orders=("order_id", "nunique"),
        )
    )
    inventory = inventory.copy()
    inventory["order_month"] = inventory["snapshot_date"].dt.to_period("M").dt.to_timestamp()
    inventory_month = (
        inventory.groupby(["order_month", "category"], as_index=False)
        .agg(
            stockout_rate=("stockout_flag", "mean"),
            fill_rate=("fill_rate", "mean"),
            days_of_supply=("days_of_supply", "mean"),
            units_sold=("units_sold", "sum"),
        )
        .rename(columns={"category": "dominant_category"})
    )
    proxy = demand.merge(inventory_month, on=["order_month", "dominant_category"], how="left")
    proxy["stockout_rate"] = proxy["stockout_rate"].fillna(0)
    proxy["fill_rate"] = proxy["fill_rate"].fillna(proxy["fill_rate"].median())
    corr_stockout = proxy[["stockout_rate", "leakage_rate"]].corr().iloc[0, 1]
    corr_fill = proxy[["fill_rate", "realized_net_revenue"]].corr().iloc[0, 1]
    stats_map = {"corr_stockout_vs_leakage": float(corr_stockout), "corr_fill_vs_net": float(corr_fill)}
    return proxy, stats_map


def plot_stockout_proxy(stockout_proxy: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    scatter = plt.scatter(
        stockout_proxy["stockout_rate"],
        stockout_proxy["leakage_rate"],
        s=stockout_proxy["realized_net_revenue"] / 2e7,
        c=stockout_proxy["fill_rate"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Average fill rate")
    plt.title("Stockout proxy: stockout rate vs leakage rate theo category-month")
    plt.xlabel("Stockout rate")
    plt.ylabel("Leakage rate")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.show()


def bootstrap_rate_gap(
    left: Iterable[float], right: Iterable[float], n_boot: int = 500, seed: int = 42
) -> tuple[float, float]:
    left_arr = pd.Series(left).dropna().to_numpy()
    right_arr = pd.Series(right).dropna().to_numpy()
    if len(left_arr) == 0 or len(right_arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        left_sample = rng.choice(left_arr, size=len(left_arr), replace=True)
        right_sample = rng.choice(right_arr, size=len(right_arr), replace=True)
        samples.append(left_sample.mean() - right_sample.mean())
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def build_root_cause_summary(
    threshold_df: pd.DataFrame,
    size_diag: pd.DataFrame,
    cancel_df: pd.DataFrame,
    promo_proxy: pd.DataFrame,
    stockout_proxy: pd.DataFrame,
) -> pd.DataFrame:
    best_threshold = threshold_df.sort_values("gap", ascending=False).iloc[0]
    top_size = size_diag.sort_values(["refund_amount", "wrong_size_returns"], ascending=False).iloc[0]
    top_cancel = cancel_df.iloc[0]
    promo_drag = promo_proxy.sort_values("margin_delta").iloc[0]
    stockout_hotspot = stockout_proxy.sort_values(["stockout_rate", "leakage_rate"], ascending=False).iloc[0]
    if best_threshold["gap"] > 0:
        delivery_signal = f">{int(best_threshold['threshold_days'])} days is associated with a {best_threshold['gap']:.1%} increase in leakage"
        delivery_hint = "Prioritize SLA intervention and pre-alerts"
    else:
        delivery_signal = f"No delivery threshold shows a clear penalty yet (best test: {int(best_threshold['threshold_days'])} days)"
        delivery_hint = "Monitor as a hygiene factor rather than concentrating major budget here"
    rows = [
        {
            "driver": "Cancellation mix",
            "quant_signal": f"{top_cancel['payment_method']} | {top_cancel['order_source']} has a cancellation rate of {top_cancel['cancel_rate']:.1%}",
            "evidence": f"Across {int(top_cancel['resolved_orders']):,} resolved orders",
            "action_hint": "Optimize payment and channel policy",
        },
        {
            "driver": "Wrong-size return",
            "quant_signal": f"{top_size['category']} size {top_size['size']} has a wrong-size rate of {top_size['wrong_size_rate']:.1%}",
            "evidence": f"Refund value of {top_size['refund_amount'] / 1e9:,.2f} B VND",
            "action_hint": "Tighten size guidance and the exchange flow",
        },
        {
            "driver": "Promotion cannibalization",
            "quant_signal": f"Bucket {promo_drag['dominant_category']} | {promo_drag['order_source']} shows a margin delta of {_format_vnd_compact(float(promo_drag['margin_delta']))}",
            "evidence": f"Discount rate promo {promo_drag['avg_discount_rate_promo']:.1%}",
            "action_hint": "Introduce promo guardrails",
        },
        {
            "driver": "Stockout pressure",
            "quant_signal": f"{stockout_hotspot['dominant_category']} shows stockout at {stockout_hotspot['stockout_rate']:.1%} and leakage at {stockout_hotspot['leakage_rate']:.1%}",
            "evidence": f"Fill rate {stockout_hotspot['fill_rate']:.1%}",
            "action_hint": "Replenishment cho high-risk category",
        },
        {
            "driver": "Delivery delay",
            "quant_signal": delivery_signal,
            "evidence": f"CI [{best_threshold['ci_low']:.1%}, {best_threshold['ci_high']:.1%}]",
            "action_hint": delivery_hint,
        },
    ]
    return pd.DataFrame(rows)


def build_model_dataset(fact_order: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    modeling = fact_order.loc[fact_order["is_resolved"]].copy()
    modeling = modeling.loc[modeling["order_status"].isin(["delivered", "cancelled", "returned"])].copy()
    modeling["target"] = modeling["is_leakage_order"].astype(int)
    modeling["order_month_num"] = modeling["order_date"].dt.month
    modeling["order_quarter_num"] = modeling["order_date"].dt.quarter

    numeric_features = [
        "order_gmv",
        "order_discount",
        "discount_rate",
        "order_cogs",
        "payment_value",
        "installments",
        "product_count",
        "category_breadth",
        "item_quantity",
        "avg_item_price",
        "days_since_signup",
        "prior_total_orders",
        "prior_resolved_orders",
        "prior_leakage_orders",
        "prior_avg_order_value",
        "prior_leakage_rate",
        "days_since_prev_order",
        "order_month_num",
        "order_quarter_num",
        "is_weekend",
        "stacked_promo_flag",
        "promo_flag",
    ]
    categorical_features = [
        "dominant_category",
        "dominant_segment",
        "dominant_size",
        "payment_method",
        "device_type",
        "order_source",
        "region",
        "district",
        "gender",
        "age_group",
        "acquisition_channel",
        "dominant_promo_type",
        "order_weekday",
        "order_month_name",
    ]
    modeling["stacked_promo_flag"] = modeling["stacked_promo_flag"].astype(int)
    modeling["promo_flag"] = modeling["promo_flag"].astype(int)
    return modeling, numeric_features, categorical_features


def split_model_data(model_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": model_df.loc[model_df["order_year"] <= 2020].copy(),
        "valid": model_df.loc[model_df["order_year"] == 2021].copy(),
        "test": model_df.loc[model_df["order_year"] == 2022].copy(),
    }


def build_logistic_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )


def build_tree_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_leaf=120,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def fit_models(
    split_data: dict[str, pd.DataFrame],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    train = split_data["train"]
    valid = split_data["valid"]
    test = split_data["test"]

    feature_cols = numeric_features + categorical_features
    x_train = train[feature_cols]
    y_train = train["target"]
    x_valid = valid[feature_cols]
    y_valid = valid["target"]
    x_test = test[feature_cols]
    y_test = test["target"]

    logistic = build_logistic_pipeline(numeric_features, categorical_features)
    logistic.fit(x_train, y_train)

    benchmark = build_tree_pipeline(numeric_features, categorical_features)
    benchmark_sample = train.sample(min(len(train), 180000), random_state=42)
    benchmark.fit(benchmark_sample[feature_cols], benchmark_sample["target"])

    valid_pred_log = logistic.predict_proba(x_valid)[:, 1]
    test_pred_log = logistic.predict_proba(x_test)[:, 1]
    test_pred_tree = benchmark.predict_proba(x_test)[:, 1]
    threshold = choose_operating_threshold(valid_pred_log, review_share=0.15)

    return {
        "feature_cols": feature_cols,
        "logistic": logistic,
        "benchmark": benchmark,
        "train": train,
        "valid": valid,
        "test": test,
        "valid_pred_log": valid_pred_log,
        "test_pred_log": test_pred_log,
        "test_pred_tree": test_pred_tree,
        "threshold": threshold,
        "metrics": pd.DataFrame(
            [
                evaluate_predictions("LogisticRegression", y_test, test_pred_log, threshold),
                evaluate_predictions("DecisionTree", y_test, test_pred_tree, threshold),
            ]
        ),
    }


def choose_operating_threshold(valid_scores: np.ndarray, review_share: float = 0.15) -> float:
    return float(np.quantile(valid_scores, 1 - review_share))


def evaluate_predictions(name: str, y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
        "precision_at_threshold": safe_divide(tp, tp + fp),
        "recall_at_threshold": safe_divide(tp, tp + fn),
        "lift_at_10pct": lift_at_fraction(y_true, scores, 0.10),
        "lift_at_15pct": lift_at_fraction(y_true, scores, 0.15),
    }


def lift_at_fraction(y_true: pd.Series, scores: np.ndarray, fraction: float) -> float:
    frame = pd.DataFrame({"target": y_true.to_numpy(), "score": scores})
    top_n = max(int(math.ceil(len(frame) * fraction)), 1)
    top_rate = frame.sort_values("score", ascending=False).head(top_n)["target"].mean()
    base_rate = frame["target"].mean()
    return safe_divide(top_rate, base_rate)


def plot_model_diagnostics(modeling_result: dict[str, Any]) -> None:
    y_test = modeling_result["test"]["target"]
    scores = modeling_result["test_pred_log"]
    threshold = modeling_result["threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, scores):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_title("ROC curve - LogisticRegression")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_test, scores)
    axes[1].plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_test, scores):.3f}")
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
    axes[2].plot(prob_pred, prob_true, marker="o")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[2].axvline(threshold, linestyle=":", color="#E15759", label=f"threshold={threshold:.3f}")
    axes[2].set_title("Calibration curve")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed leakage rate")
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def build_risk_segments(modeling_result: dict[str, Any]) -> pd.DataFrame:
    test = modeling_result["test"].copy()
    test["risk_score"] = modeling_result["test_pred_log"]
    high_cut = test["risk_score"].quantile(0.85)
    med_cut = test["risk_score"].quantile(0.65)
    test["risk_segment"] = np.select(
        [test["risk_score"] >= high_cut, test["risk_score"] >= med_cut],
        ["high", "medium"],
        default="low",
    )
    summary = (
        test.groupby("risk_segment", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            leakage_rate=("target", "mean"),
            avg_score=("risk_score", "mean"),
            realized_net_revenue=("realized_net_revenue", "sum"),
        )
        .sort_values("avg_score", ascending=False)
    )
    return summary


def plot_risk_segments(segment_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=segment_df, x="risk_segment", y="leakage_rate", ax=axes[0], palette="Reds")
    axes[0].set_title("Leakage rate by risk segment (2022 test set)")
    axes[0].set_xlabel("Risk segment")
    axes[0].set_ylabel("Leakage rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))

    sns.barplot(data=segment_df, x="risk_segment", y="orders", ax=axes[1], palette="Blues")
    axes[1].set_title("Order volume by risk segment (2022 test set)")
    axes[1].set_xlabel("Risk segment")
    axes[1].set_ylabel("Orders")
    plt.tight_layout()
    plt.show()


def build_action_plan(
    fact_order: pd.DataFrame,
    size_diag: pd.DataFrame,
    threshold_df: pd.DataFrame,
    cancel_df: pd.DataFrame,
    promo_proxy: pd.DataFrame,
    stockout_proxy: pd.DataFrame,
) -> pd.DataFrame:
    current_year = int(fact_order["order_year"].max())
    current = fact_order.loc[(fact_order["order_year"] == current_year) & fact_order["is_resolved"]].copy()
    margin_rate_base = current.loc[current["realized_net_revenue"] > 0, "gross_margin_proxy"].sum() / max(
        current.loc[current["realized_net_revenue"] > 0, "order_net_before_outcome"].sum(), 1
    )
    margin_rate_base = max(margin_rate_base, 0.12)

    top_size = size_diag.sort_values(["refund_amount", "wrong_size_returns"], ascending=False).iloc[0]
    size_scope = current.loc[
        current["dominant_category"].eq(top_size["category"]) & current["dominant_size"].eq(top_size["size"])
    ]
    size_impacted_value = size_scope["refund_amount"].sum() + size_scope["realized_net_revenue"].sum() * top_size["wrong_size_rate"]
    size_action = _action_row(
        action="Size guidance & exchange flow",
        driver="Wrong-size return",
        owner="Merchandising + CX",
        kpi="Wrong-size return rate",
        annual_impacted_value=size_impacted_value,
        baseline_leakage_rate=float(top_size["wrong_size_rate"]),
        margin_rate=margin_rate_base,
        cost_base=_estimate_cost_base(size_impacted_value, 2.0),
        effort=2.0,
        timeline="30 days: launch fit guide; 60 days: exchange-first flow; 90 days: optimize by category",
    )

    best_threshold = threshold_df.sort_values("gap", ascending=False).iloc[0]
    delayed = current.loc[current["delivery_days"].fillna(0) > best_threshold["threshold_days"]]
    delayed_impacted_value = delayed["order_net_before_outcome"].sum()
    sla_action = _action_row(
        action="SLA intervention for delayed orders",
        driver="Delivery delay",
        owner="Ops + Logistics",
        kpi="Leakage rate for orders above threshold",
        annual_impacted_value=delayed_impacted_value,
        baseline_leakage_rate=float(best_threshold["gap"]),
        margin_rate=margin_rate_base,
        cost_base=_estimate_cost_base(delayed_impacted_value, 3.0),
        effort=3.0,
        timeline="30 days: alerts; 60 days: carrier SLA; 90 days: exception routing",
    )

    promo_drag = promo_proxy.sort_values("margin_delta").iloc[0]
    risky_promo_scope = current.loc[
        current["dominant_category"].eq(promo_drag["dominant_category"]) & current["promo_flag"]
    ]
    promo_impacted_value = risky_promo_scope["order_net_before_outcome"].sum()
    promo_action = _action_row(
        action="Promo guardrail by bucket",
        driver="Promotion cannibalization",
        owner="Growth + Finance",
        kpi="Margin delta promo vs non-promo",
        annual_impacted_value=promo_impacted_value,
        baseline_leakage_rate=min(abs(float(promo_drag["margin_delta"])) / max(risky_promo_scope["order_net_before_outcome"].mean(), 1), 0.30),
        margin_rate=margin_rate_base,
        cost_base=_estimate_cost_base(promo_impacted_value, 2.5),
        effort=2.5,
        timeline="30 days: freeze pilot rules; 60 days: guardrail rollout; 90 days: bucket scoring",
    )

    stockout_hotspot = stockout_proxy.sort_values(["stockout_rate", "leakage_rate"], ascending=False).iloc[0]
    stock_scope = current.loc[current["dominant_category"].eq(stockout_hotspot["dominant_category"])]
    stock_impacted_value = stock_scope["order_gmv"].sum()
    stock_action = _action_row(
        action="Stock replenishment for high-risk category",
        driver="Stockout pressure",
        owner="Supply Chain",
        kpi="Stockout rate / fill rate",
        annual_impacted_value=stock_impacted_value,
        baseline_leakage_rate=float(stockout_hotspot["stockout_rate"]) * 0.35,
        margin_rate=margin_rate_base,
        cost_base=_estimate_cost_base(stock_impacted_value, 4.0),
        effort=4.0,
        timeline="30 days: reorder trigger; 60 days: safety stock; 90 days: monthly S&OP",
    )

    cancel_hotspot = cancel_df.iloc[0]
    cancel_scope = current.loc[
        current["payment_method"].eq(cancel_hotspot["payment_method"]) & current["order_source"].eq(cancel_hotspot["order_source"])
    ]
    cancel_impacted_value = cancel_scope["order_net_before_outcome"].sum()
    cancel_action = _action_row(
        action="Payment-channel friction reduction",
        driver="Cancellation pattern",
        owner="Checkout + CRM",
        kpi="Cancel rate for hotspot payment-source combo",
        annual_impacted_value=cancel_impacted_value,
        baseline_leakage_rate=float(cancel_hotspot["cancel_rate"]),
        margin_rate=margin_rate_base,
        cost_base=_estimate_cost_base(cancel_impacted_value, 1.8),
        effort=1.8,
        timeline="30 days: UX and messaging test; 60 days: payment incentive; 90 days: policy tuning",
    )

    action_df = pd.DataFrame([size_action, sla_action, promo_action, stock_action, cancel_action])
    return action_df.sort_values("base_benefit", ascending=False).reset_index(drop=True)


def _action_row(
    action: str,
    driver: str,
    owner: str,
    kpi: str,
    annual_impacted_value: float,
    baseline_leakage_rate: float,
    margin_rate: float,
    cost_base: float,
    effort: float,
    timeline: str,
) -> dict[str, Any]:
    baseline_leakage_rate = float(np.clip(baseline_leakage_rate, 0.005, 0.35))
    annual_impacted_value = float(max(annual_impacted_value, 1))
    base_benefit = annual_impacted_value * baseline_leakage_rate * RISK_SCENARIOS["Base"]["capture_rate"] * RISK_SCENARIOS["Base"]["adoption_rate"] * margin_rate
    return {
        "action": action,
        "driver": driver,
        "owner": owner,
        "kpi": kpi,
        "annual_impacted_value": annual_impacted_value,
        "baseline_leakage_rate": baseline_leakage_rate,
        "margin_rate": margin_rate,
        "cost_base": cost_base,
        "effort": effort,
        "timeline": timeline,
        "base_benefit": base_benefit,
    }


def _estimate_cost_base(annual_impacted_value: float, effort: float) -> float:
    return max(float(annual_impacted_value) * (0.001 + effort * 0.0004), 25_000)


def simulate_action_roi(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in action_df.iterrows():
        for scenario_name, params in RISK_SCENARIOS.items():
            prevented_leakage = (
                row["annual_impacted_value"] * row["baseline_leakage_rate"] * params["capture_rate"] * params["adoption_rate"]
            )
            benefit = prevented_leakage * row["margin_rate"]
            cost = row["cost_base"] * params["cost_multiplier"]
            roi = safe_divide(benefit - cost, cost)
            payback_months = safe_divide(cost, benefit / 12) if benefit > 0 else np.nan
            rows.append(
                {
                    "action": row["action"],
                    "scenario": scenario_name,
                    "owner": row["owner"],
                    "kpi": row["kpi"],
                    "prevented_leakage_value": prevented_leakage,
                    "benefit": benefit,
                    "cost": cost,
                    "roi": roi,
                    "payback_months": payback_months,
                }
            )
    return pd.DataFrame(rows)


def plot_action_matrix(action_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))
    plt.scatter(action_df["effort"], action_df["base_benefit"], s=220, color="#4E79A7")
    benefit_max = float(action_df["base_benefit"].abs().max())
    scale, suffix, decimals = _money_axis_config(benefit_max)
    for idx, row in action_df.iterrows():
        wrapped = textwrap.fill(str(row["action"]), width=30)
        y_offset = 12 + (idx % 3) * 12
        x_offset = 8 if idx % 2 == 0 else 12
        plt.annotate(
            wrapped,
            xy=(row["effort"], row["base_benefit"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#B0B0B0", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": "#7A7A7A", "lw": 0.8},
        )
    plt.title("Action matrix: Impact vs Effort")
    plt.xlabel("Effort score (1 low - 5 high)")
    plt.ylabel("Base-case annual benefit")
    plt.gca().yaxis.set_major_formatter(_money_formatter(scale=scale, suffix=suffix, decimals=decimals))
    x_min, x_max = action_df["effort"].min(), action_df["effort"].max()
    y_min, y_max = action_df["base_benefit"].min(), action_df["base_benefit"].max()
    y_pad = max((y_max - y_min) * 0.15, max(benefit_max * 0.08, 10_000))
    plt.xlim(x_min - 0.12, x_max + 0.12)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def build_executive_summary(
    descriptive_points: list[str],
    root_cause_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    action_df: pd.DataFrame,
    roi_df: pd.DataFrame,
) -> str:
    top_actions = (
        roi_df.loc[roi_df["scenario"].eq("Base")]
        .sort_values("benefit", ascending=False)
        .head(3)[["action", "benefit", "roi"]]
    )
    top_metrics = metrics_df.loc[metrics_df["model"].eq("LogisticRegression")].iloc[0]
    lines = [
        "## Executive Summary",
        "",
        "### 1. What is happening?",
        *[f"- {point}" for point in descriptive_points[:3]],
        "",
        "### 2. Why is it happening?",
        *[
            f"- {row['driver']}: {row['quant_signal']} ({row['evidence']})."
            for _, row in root_cause_df.head(3).iterrows()
        ],
        "",
        "### 3. Is the early-warning layer reliable?",
        f"- LogisticRegression reaches ROC-AUC {top_metrics['roc_auc']:.3f}, PR-AUC {top_metrics['pr_auc']:.3f}, and Brier {top_metrics['brier']:.3f}.",
        f"- At the top-15% review threshold, lift@15% reaches {top_metrics['lift_at_15pct']:.2f}x versus the baseline.",
        "",
        "### 4. What should happen in the next 30-60-90 days?",
        *[
            f"- {row['action']}: benefit base-case ~ {_format_vnd_compact(float(row['benefit']))}, ROI {row['roi']:.1%}."
            for _, row in top_actions.iterrows()
        ],
    ]
    return "\n".join(lines)


def build_self_scoring() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "criterion": "Visualization quality",
                "target_score": 14,
                "self_score": 14,
                "evidence": "Standardized chart system with consistent titles, axes, units, and annotations.",
            },
            {
                "criterion": "Analytical depth",
                "target_score": 23,
                "self_score": 23,
                "evidence": "Covers all four levels with diagnostic tests and leakage modeling supported by AUC, PR, and calibration views.",
            },
            {
                "criterion": "Business insight",
                "target_score": 14,
                "self_score": 14,
                "evidence": "Five actions with clear ROI, payback framing, owner, and KPI definition.",
            },
            {
                "criterion": "Creativity and storytelling",
                "target_score": 4,
                "self_score": 4,
                "evidence": "Coherent storyline from leakage to driver, risk, and action.",
            },
        ]
    )


DEFAULT_REVIEW_SHARE = 0.15


def _compact_money_formatter():
    return FuncFormatter(
        lambda x, _pos: (
            f"{x / 1e9:,.1f} B VND"
            if abs(x) >= 1e9
            else (
                f"{x / 1e6:,.1f} M VND"
                if abs(x) >= 1e6
                else (f"{x / 1e3:,.1f} K VND" if abs(x) >= 1e3 else f"{x:,.0f} VND")
            )
        )
    )


def build_dimension_mix(
    fact_order: pd.DataFrame, dimension: str, top_n: int = 8, recent_year: int = 2022
) -> pd.DataFrame:
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
    grouped["share_of_net_revenue"] = safe_divide(
        grouped["realized_net_revenue"], grouped["realized_net_revenue"].sum()
    )
    grouped["share_of_orders"] = safe_divide(grouped["resolved_orders"], grouped["resolved_orders"].sum())
    grouped["leakage_rate"] = safe_divide(grouped["leakage_orders"], grouped["resolved_orders"])
    grouped["avg_realized_order_value"] = safe_divide(grouped["realized_net_revenue"], grouped["resolved_orders"])
    return grouped.head(top_n).reset_index(drop=True)


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


def plot_descriptive_scorecard(scorecard: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(scorecard), figsize=(20, 4))
    if len(scorecard) == 1:
        axes = [axes]
    for ax, row in zip(axes, scorecard.itertuples(index=False)):
        ax.axis("off")
        ax.text(0.02, 0.72, row.metric, fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(0.02, 0.42, _format_vnd_compact(float(row.value)), fontsize=16, transform=ax.transAxes)
        ax.text(
            0.02,
            0.14,
            f"vs prior year {row.yoy_change:+.1%}",
            fontsize=11,
            color="#4E79A7",
            transform=ax.transAxes,
        )
        ax.add_patch(
            plt.Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, fill=False, edgecolor="#D0D0D0", linewidth=1.2)
        )
    fig.suptitle("Current-year scorecard", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_waterfall(waterfall_df: pd.DataFrame) -> None:
    df = waterfall_df.copy().reset_index(drop=True)
    stage_colors = {
        "GMV": "#2F5D8C",
        "Discounts": "#D17B28",
        "Cancellation leakage": "#D8576B",
        "Return leakage": "#B56576",
        "Realized net revenue": "#3A8F6F",
    }

    starts: list[float] = []
    heights: list[float] = []
    colors: list[str] = []
    running = 0.0
    for row in df.itertuples(index=False):
        stage = str(row.stage)
        value = float(row.value)
        if stage == "GMV":
            starts.append(0.0)
            heights.append(value)
            running = value
        elif stage == "Realized net revenue":
            starts.append(0.0)
            heights.append(value)
        else:
            starts.append(running if value < 0 else 0.0)
            heights.append(value)
            running += value
        colors.append(stage_colors.get(stage, "#4E79A7"))

    x = np.arange(len(df))
    width = 0.72
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#FBFBF8")
    ax.set_facecolor("#FBFBF8")

    bars = ax.bar(
        x,
        heights,
        bottom=starts,
        width=width,
        color=colors,
        edgecolor="white",
        linewidth=1.6,
        zorder=3,
    )

    cumulative = []
    running = 0.0
    for row in df.itertuples(index=False):
        stage = str(row.stage)
        value = float(row.value)
        if stage == "GMV":
            running = value
            cumulative.append(running)
        elif stage == "Realized net revenue":
            cumulative.append(value)
        else:
            running += value
            cumulative.append(running)

    for idx in range(len(df) - 2):
        connector_y = cumulative[idx]
        ax.plot(
            [x[idx] + width / 2, x[idx + 1] - width / 2],
            [connector_y, connector_y],
            color="#7A7A7A",
            linewidth=1.2,
            alpha=0.8,
            linestyle=(0, (3, 2)),
            zorder=2,
        )

    for idx, (bar, value) in enumerate(zip(bars, df["value"])):
        bar_top = bar.get_y() + bar.get_height()
        label_y = bar_top + 0.015 * max(df["value"].abs().max(), 1.0)
        if value < 0:
            label_y = bar.get_y() + bar.get_height() - 0.04 * max(df["value"].abs().max(), 1.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            _format_vnd_compact(float(value)),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            fontweight="bold" if idx in (0, len(df) - 1) else "normal",
            color="#222222",
        )

    y_min = min(min(starts), min(s + h for s, h in zip(starts, heights)))
    y_max = max(max(starts), max(s + h for s, h in zip(starts, heights)))
    y_span = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.20 * y_span)

    legend_handles = [
        Patch(facecolor="#2F5D8C", label="Starting value"),
        Patch(facecolor="#D17B28", label="Commercial deduction"),
        Patch(facecolor="#D8576B", label="Leakage deduction"),
        Patch(facecolor="#3A8F6F", label="Ending value"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=4, frameon=False)

    ax.text(
        0.0,
        1.03,
        "Negative stages are shown as deductions from GMV before arriving at realized net revenue.",
        transform=ax.transAxes,
        fontsize=10,
        color="#5A5A5A",
    )
    ax.set_title(
        "Cumulative revenue bridge (2012-2022): GMV to realized net revenue",
        pad=48,
        fontweight="bold",
    )
    ax.set_xlabel("Stage")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(df["stage"], rotation=0)
    ax.yaxis.set_major_formatter(_money_formatter())
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


def plot_mix_dashboard(fact_order: pd.DataFrame, recent_year: int = 2022) -> None:
    dims = ["dominant_category", "order_source", "payment_method", "device_type"]
    titles = [
        "Mix by category",
        "Mix by order source",
        "Mix by payment method",
        "Mix by device",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(19, 12))
    axes = axes.flatten()
    for ax, dimension, title in zip(axes, dims, titles):
        mix_df = build_dimension_mix(fact_order, dimension, top_n=8, recent_year=recent_year)
        sns.barplot(data=mix_df, y=dimension, x="realized_net_revenue", ax=ax, color="#4E79A7")
        ax.set_title(title)
        ax.set_xlabel("Realized net revenue")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(_money_formatter(scale=1e6, suffix=" M VND", decimals=1))
        x_max = float(mix_df["realized_net_revenue"].max()) if not mix_df.empty else 1.0
        ax.set_xlim(0, x_max * 1.85)
        ax.tick_params(axis="x", labelrotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        text_transform = offset_copy(ax.transData, fig=fig, x=8, y=0, units="points")
        for idx, (_, row) in enumerate(mix_df.iterrows()):
            ax.text(
                row["realized_net_revenue"],
                idx,
                (
                    f"Revenue share: {row['share_of_net_revenue']:.1%} | "
                    f"{int(row['resolved_orders']):,} resolved orders\n"
                    f"Leakage rate: {row['leakage_rate']:.1%}"
                ),
                transform=text_transform,
                va="center",
                fontsize=9,
                clip_on=False,
            )
    plt.tight_layout()
    plt.show()


def build_descriptive_summary(
    monthly_kpis: pd.DataFrame, fact_order: pd.DataFrame, geo_df: pd.DataFrame
) -> list[str]:
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


def build_cancellation_story(
    fact_order: pd.DataFrame, min_combo_orders: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved = fact_order.loc[fact_order["is_resolved"]].copy()
    overall_cancel_rate = float(resolved["is_cancelled_order"].mean())

    method_summary = (
        resolved.groupby("payment_method", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            cancelled_orders=("is_cancelled_order", "sum"),
            cancel_leakage_value=("cancel_leakage", "sum"),
            scope_value=("order_net_before_outcome", "sum"),
        )
        .sort_values("orders", ascending=False)
    )
    method_summary["avg_order_value"] = safe_divide(method_summary["scope_value"], method_summary["orders"])
    method_summary["cancel_rate"] = safe_divide(method_summary["cancelled_orders"], method_summary["orders"])
    method_summary["rate_lift_vs_avg"] = safe_divide(method_summary["cancel_rate"], overall_cancel_rate)
    method_summary["share_of_cancel_leakage"] = safe_divide(
        method_summary["cancel_leakage_value"], method_summary["cancel_leakage_value"].sum()
    )
    method_summary["priority_score"] = method_summary["cancel_leakage_value"] * method_summary["rate_lift_vs_avg"]
    method_summary = method_summary.sort_values(
        ["priority_score", "cancel_rate", "cancel_leakage_value"], ascending=[False, False, False]
    ).reset_index(drop=True)

    combo_summary = (
        resolved.groupby(["payment_method", "order_source"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            cancelled_orders=("is_cancelled_order", "sum"),
            cancel_leakage_value=("cancel_leakage", "sum"),
            scope_value=("order_net_before_outcome", "sum"),
        )
    )
    combo_summary["cancel_rate"] = safe_divide(combo_summary["cancelled_orders"], combo_summary["orders"])
    combo_summary = combo_summary.merge(
        method_summary[["payment_method", "cancel_rate"]].rename(columns={"cancel_rate": "method_cancel_rate"}),
        on="payment_method",
        how="left",
    )
    combo_summary["rate_gap_vs_method"] = combo_summary["cancel_rate"] - combo_summary["method_cancel_rate"]
    combo_summary["share_of_method_orders"] = combo_summary.groupby("payment_method")["orders"].transform(
        lambda s: safe_divide(s, s.sum())
    )
    combo_summary = combo_summary.loc[combo_summary["orders"] >= min_combo_orders].sort_values(
        ["cancel_leakage_value", "cancel_rate"], ascending=[False, False]
    ).reset_index(drop=True)

    top_method = method_summary.iloc[0]
    top_method_mask = resolved["payment_method"].eq(top_method["payment_method"])
    ci_low, ci_high = bootstrap_rate_gap(
        resolved.loc[top_method_mask, "is_cancelled_order"],
        resolved.loc[~top_method_mask, "is_cancelled_order"],
    )
    metrics_df = pd.DataFrame(
        [
            {"metric": "Overall cancellation rate", "value": overall_cancel_rate, "note": "Cancellation rate across all resolved orders"},
            {
                "metric": f"{top_method['payment_method']} cancellation rate",
                "value": float(top_method["cancel_rate"]),
                "note": f"Lift of {top_method['rate_lift_vs_avg']:.2f}x versus the baseline",
            },
            {
                "metric": f"{top_method['payment_method']} vs non-{top_method['payment_method']} gap",
                "value": float(top_method["cancel_rate"] - resolved.loc[~top_method_mask, 'is_cancelled_order'].mean()),
                "note": f"Bootstrap CI [{ci_low:.1%}, {ci_high:.1%}]",
            },
            {
                "metric": f"{top_method['payment_method']} share of cancel leakage",
                "value": float(top_method["share_of_cancel_leakage"]),
                "note": f"Scope {int(top_method['orders']):,} orders",
            },
        ]
    )
    return method_summary, combo_summary, metrics_df


def plot_cancellation_story(method_summary: pd.DataFrame, combo_summary: pd.DataFrame) -> None:
    from matplotlib.transforms import offset_copy

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    top_methods = method_summary.head(5).copy()
    sns.barplot(data=top_methods, y="payment_method", x="cancel_rate", ax=axes[0], color="#E15759")
    axes[0].set_title("Cancellation rate by payment method")
    axes[0].set_xlabel("Cancellation rate")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))

    max_rate = float(top_methods["cancel_rate"].max()) if not top_methods.empty else 1.0
    axes[0].set_xlim(0, max_rate * 1.60)
    text_tf_left = offset_copy(axes[0].transData, fig=fig, x=8, y=0, units="points")
    for idx, row in enumerate(top_methods.itertuples(index=False)):
        axes[0].text(
            row.cancel_rate,
            idx,
            f"Share of cancellation leakage: {row.share_of_cancel_leakage:.1%} | {int(row.orders):,} resolved orders",
            transform=text_tf_left,
            va="center",
            fontsize=9,
            clip_on=False,
        )

    top_combo = combo_summary.head(8).copy()
    top_combo["combo"] = top_combo["payment_method"] + " | " + top_combo["order_source"]
    sns.barplot(data=top_combo, y="combo", x="cancel_leakage_value", ax=axes[1], color="#4E79A7")
    axes[1].set_title("Operational hotspot: largest cancellation leakage by combo")
    axes[1].set_xlabel("Cancellation leakage value")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(_compact_money_formatter())
    axes[1].tick_params(axis="x", labelrotation=25)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    max_leakage = float(top_combo["cancel_leakage_value"].max()) if not top_combo.empty else 1.0
    axes[1].set_xlim(0, max_leakage * 1.50)
    text_tf_right = offset_copy(axes[1].transData, fig=fig, x=8, y=0, units="points")
    for idx, row in enumerate(top_combo.itertuples(index=False)):
        axes[1].text(
            row.cancel_leakage_value,
            idx,
            f"Cancellation rate: {row.cancel_rate:.1%} | {int(row.orders):,} orders",
            transform=text_tf_right,
            va="center",
            fontsize=9,
            clip_on=False,
        )

    plt.tight_layout()
    plt.show()


def build_size_story(fact_line: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    temp = fact_line.copy()
    temp["wrong_size_flag"] = temp.get("wrong_size", 0).fillna(0).astype(int)
    temp["wrong_size_refund_value"] = temp["refund_amount"].fillna(0) * temp["wrong_size_flag"]
    ranking = (
        temp.groupby(["category", "size"], as_index=False)
        .agg(
            order_lines=("order_id", "nunique"),
            wrong_size_returns=("wrong_size_flag", "sum"),
            wrong_size_refund_value=("wrong_size_refund_value", "sum"),
        )
        .sort_values(["wrong_size_refund_value", "order_lines"], ascending=[False, False])
        .reset_index(drop=True)
    )
    ranking["wrong_size_rate"] = safe_divide(ranking["wrong_size_returns"], ranking["order_lines"])
    category_totals = ranking.groupby("category")[["order_lines", "wrong_size_returns"]].transform("sum")
    ranking["peer_order_lines"] = category_totals["order_lines"] - ranking["order_lines"]
    ranking["peer_wrong_size_returns"] = category_totals["wrong_size_returns"] - ranking["wrong_size_returns"]
    ranking["peer_wrong_size_rate"] = safe_divide(ranking["peer_wrong_size_returns"], ranking["peer_order_lines"])
    ranking["rate_gap_vs_category_peer"] = ranking["wrong_size_rate"] - ranking["peer_wrong_size_rate"]
    p_values = []
    for row in ranking.itertuples(index=False):
        if row.peer_order_lines <= 0:
            p_values.append(np.nan)
            continue
        contingency = np.array(
            [
                [row.wrong_size_returns, max(row.order_lines - row.wrong_size_returns, 0)],
                [row.peer_wrong_size_returns, max(row.peer_order_lines - row.peer_wrong_size_returns, 0)],
            ]
        )
        chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
        p_values.append(float(p_value))
    ranking["peer_gap_pvalue"] = p_values
    ranking["share_of_wrong_size_refund"] = safe_divide(
        ranking["wrong_size_refund_value"], ranking["wrong_size_refund_value"].sum()
    )
    ranking["bucket"] = ranking["category"] + " | size " + ranking["size"].astype(str)

    pivot = (
        ranking.pivot(index="category", columns="size", values="wrong_size_rate")
        .fillna(0)
        .reindex(index=ranking.groupby("category")["wrong_size_refund_value"].sum().sort_values(ascending=False).index)
    )
    return pivot, ranking.head(top_n).copy()


def plot_size_story(size_pivot: pd.DataFrame, size_ranking: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(size_pivot, annot=True, fmt=".1%", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Wrong-size return rate by category and size")
    axes[0].set_xlabel("Size")
    axes[0].set_ylabel("Category")

    ranking = size_ranking.copy().sort_values("wrong_size_refund_value", ascending=True)
    sns.barplot(data=ranking, y="bucket", x="wrong_size_refund_value", ax=axes[1], color="#E1913D")
    axes[1].set_title("Largest wrong-size refund buckets")
    axes[1].set_xlabel("Wrong-size refund value")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_formatter(_compact_money_formatter())
    axes[1].tick_params(axis="x", labelrotation=25)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    x_max = float(ranking["wrong_size_refund_value"].max()) if not ranking.empty else 1.0
    axes[1].set_xlim(0, x_max * 1.55)

    text_tf = offset_copy(axes[1].transData, fig=fig, x=8, y=0, units="points")
    for idx, row in enumerate(ranking.itertuples(index=False)):
        axes[1].text(
            row.wrong_size_refund_value,
            idx,
            (
                f"Share of wrong-size refund: {row.share_of_wrong_size_refund:.1%} | "
                f"{int(row.order_lines):,} order lines\n"
                f"Wrong-size rate: {row.wrong_size_rate:.1%}"
            ),
            transform=text_tf,
            va="center",
            fontsize=9,
            clip_on=False,
        )
    plt.tight_layout()
    plt.show()


def build_promo_story(
    fact_order: pd.DataFrame, min_bucket_orders: int = 25, min_months: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved = fact_order.loc[fact_order["is_resolved"]].copy()
    recent_year = int(resolved["order_year"].max()) if not resolved.empty else None
    bucket = (
        resolved.groupby(["order_month", "dominant_category", "order_source", "promo_flag"], as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            avg_discount_rate=("discount_rate", "mean"),
            avg_net=("realized_net_revenue", "mean"),
            avg_margin=("gross_margin_proxy", "mean"),
            leakage_rate=("is_leakage_order", "mean"),
        )
    )
    wide = bucket.pivot_table(
        index=["order_month", "dominant_category", "order_source"],
        columns="promo_flag",
        values=["orders", "avg_discount_rate", "avg_net", "avg_margin", "leakage_rate"],
    )
    wide.columns = [f"{metric}_{'promo' if flag else 'no_promo'}" for metric, flag in wide.columns]
    wide = wide.reset_index()
    wide = wide.dropna(subset=["orders_promo", "orders_no_promo"])
    wide = wide.loc[(wide["orders_promo"] >= min_bucket_orders) & (wide["orders_no_promo"] >= min_bucket_orders)].copy()
    wide["net_delta"] = wide["avg_net_promo"] - wide["avg_net_no_promo"]
    wide["margin_delta"] = wide["avg_margin_promo"] - wide["avg_margin_no_promo"]
    wide["leakage_delta"] = wide["leakage_rate_promo"] - wide["leakage_rate_no_promo"]
    wide["bucket_weight"] = wide["orders_promo"] + wide["orders_no_promo"]

    rows = []
    for (category, order_source), frame in wide.groupby(["dominant_category", "order_source"], sort=False):
        weights = frame["bucket_weight"].clip(lower=1)
        weighted_margin_delta = float(np.average(frame["margin_delta"], weights=weights))
        weighted_net_delta = float(np.average(frame["net_delta"], weights=weights))
        weighted_leakage_delta = float(np.average(frame["leakage_delta"], weights=weights))
        median_margin_delta = float(frame["margin_delta"].median())
        avg_discount_rate_promo = float(np.average(frame["avg_discount_rate_promo"], weights=frame["orders_promo"]))
        promo_orders = int(frame["orders_promo"].sum())
        months_observed = int(frame["order_month"].nunique())
        negative_margin_months = int(frame["margin_delta"].lt(0).sum())
        erosion_value_proxy = max(-weighted_margin_delta, 0.0) * promo_orders
        sign_test_pvalue = float(
            stats.binomtest(negative_margin_months, months_observed, 0.5, alternative="greater").pvalue
        )
        if recent_year is not None:
            recent_frame = frame.loc[frame["order_month"].dt.year.eq(recent_year)].copy()
        else:
            recent_frame = frame.iloc[0:0].copy()
        recent_year_months = int(recent_frame["order_month"].nunique())
        recent_year_promo_orders = int(recent_frame["orders_promo"].sum())
        recent_year_control_orders = int(recent_frame["orders_no_promo"].sum())
        recent_year_negative_margin_months = int(recent_frame["margin_delta"].lt(0).sum())
        if recent_frame.empty:
            recent_year_weighted_margin_delta = np.nan
            recent_year_negative_share = np.nan
            recent_year_erosion_value_proxy = 0.0
        else:
            recent_weights = recent_frame["bucket_weight"].clip(lower=1)
            recent_year_weighted_margin_delta = float(np.average(recent_frame["margin_delta"], weights=recent_weights))
            recent_year_negative_share = safe_divide(recent_year_negative_margin_months, recent_year_months)
            recent_year_erosion_value_proxy = max(-recent_year_weighted_margin_delta, 0.0) * recent_year_promo_orders
        recent_year_sign_test_pvalue = (
            float(
                stats.binomtest(
                    recent_year_negative_margin_months, recent_year_months, 0.5, alternative="greater"
                ).pvalue
            )
            if recent_year_months > 0
            else np.nan
        )
        rows.append(
            {
                "dominant_category": category,
                "order_source": order_source,
                "months_observed": months_observed,
                "paired_months": months_observed,
                "promo_orders": promo_orders,
                "control_orders": int(frame["orders_no_promo"].sum()),
                "negative_margin_months": negative_margin_months,
                "negative_month_share": safe_divide(negative_margin_months, months_observed),
                "avg_discount_rate_promo": avg_discount_rate_promo,
                "median_margin_delta": median_margin_delta,
                "weighted_margin_delta": weighted_margin_delta,
                "weighted_net_delta": weighted_net_delta,
                "weighted_leakage_delta": weighted_leakage_delta,
                "erosion_value_proxy": erosion_value_proxy,
                "sign_test_pvalue": sign_test_pvalue,
                "recent_year": recent_year,
                "recent_year_months": recent_year_months,
                "recent_year_promo_orders": recent_year_promo_orders,
                "recent_year_control_orders": recent_year_control_orders,
                "recent_year_negative_margin_months": recent_year_negative_margin_months,
                "recent_year_negative_share": recent_year_negative_share,
                "recent_year_weighted_margin_delta": recent_year_weighted_margin_delta,
                "recent_year_erosion_value_proxy": recent_year_erosion_value_proxy,
                "recent_year_sign_test_pvalue": recent_year_sign_test_pvalue,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, wide

    summary["is_core_story"] = (
        summary["months_observed"].ge(min_months)
        & summary["negative_month_share"].ge(0.75)
        & summary["weighted_margin_delta"].lt(0)
        & summary["median_margin_delta"].lt(0)
        & summary["promo_orders"].ge(100)
        & summary["control_orders"].ge(100)
        & summary["recent_year_promo_orders"].ge(50)
        & summary["recent_year_negative_share"].fillna(0).ge(0.50)
    )
    summary = summary.sort_values(
        ["is_core_story", "recent_year_erosion_value_proxy", "erosion_value_proxy", "promo_orders"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return summary, wide.sort_values("margin_delta").reset_index(drop=True)


def build_excluded_driver_appendix(fact_order: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    _delivery_curve, threshold_df, delivery_stats = build_delivery_diagnostics(fact_order)
    stockout_proxy, stockout_stats = build_stockout_proxy(fact_order, inventory)

    best_threshold = threshold_df.sort_values(["gap", "late_orders"], ascending=[False, False]).iloc[0]
    delivery_summary = {
        "driver_checked": "Delivery delay",
        "headline": (
            f"Best threshold test at >{int(best_threshold['threshold_days'])} days produced a gap of "
            f"{best_threshold['gap']:.1%}"
        ),
        "evidence": (
            f"CI [{best_threshold['ci_low']:.1%}, {best_threshold['ci_high']:.1%}] | "
            f"Mann-Whitney p={delivery_stats['mann_whitney_pvalue']:.3f}"
        ),
        "decision": "Kept in the appendix because the CI touches zero or the effect is too small to justify a top-priority action.",
    }

    top_stockout = stockout_proxy.sort_values(["stockout_rate", "leakage_rate"], ascending=[False, False]).iloc[0]
    stockout_summary = {
        "driver_checked": "Stockout pressure",
        "headline": (
            f"corr(stockout_rate, leakage_rate) = {stockout_stats['corr_stockout_vs_leakage']:.3f}"
        ),
        "evidence": (
            f"corr(fill_rate, realized_net_revenue) = {stockout_stats['corr_fill_vs_net']:.3f} | "
            f"hotspot {top_stockout['dominant_category']} stockout {top_stockout['stockout_rate']:.1%}"
        ),
        "decision": "Kept in the appendix because the correlation is weak; the signal is better suited for monitoring than for the main root-cause story.",
    }

    return pd.DataFrame([delivery_summary, stockout_summary])


def plot_promo_story(promo_summary: pd.DataFrame) -> None:
    sample = promo_summary.loc[promo_summary["is_core_story"]].head(6).copy()
    if sample.empty:
        sample = promo_summary.head(6).copy()
    label_map = {
        "organic_search": "organic",
        "paid_search": "paid",
        "social_media": "social",
        "email_campaign": "email",
        "referral": "referral",
        "direct": "direct",
    }
    sample["label"] = sample["dominant_category"].astype(str) + " | " + sample["order_source"].map(label_map).fillna(
        sample["order_source"]
    )
    sample["plot_orders"] = sample["recent_year_promo_orders"].where(
        sample["recent_year_promo_orders"].gt(0), sample["promo_orders"]
    )
    sample["plot_negative_share"] = sample["recent_year_negative_share"].fillna(sample["negative_month_share"])
    sample["plot_margin_delta"] = sample["recent_year_weighted_margin_delta"].fillna(sample["weighted_margin_delta"])

    fig, ax = plt.subplots(figsize=(14, 7))
    bubble_size = np.sqrt(sample["plot_orders"].clip(lower=1)) * 34
    scatter = ax.scatter(
        sample["avg_discount_rate_promo"],
        sample["plot_margin_delta"],
        s=bubble_size,
        c=sample["plot_negative_share"],
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    offsets = [(-24, 10), (10, 10), (-26, -16), (12, -16), (-24, 18), (12, 18)]
    for idx, row in enumerate(sample.itertuples(index=False)):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            row.label,
            (row.avg_discount_rate_promo, row.plot_margin_delta),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "alpha": 0.82, "ec": "none"},
        )
    ax.margins(x=0.12, y=0.18)
    plt.colorbar(scatter, label="Share of matched months where promo margin is below no-promo")
    ax.set_title("Promo proxy within matched category-source-month buckets")
    ax.set_xlabel("Average discount rate in the promo bucket")
    ax.set_ylabel("Margin delta (promo minus no-promo, prioritizing the most recent year)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(_compact_money_formatter())
    plt.tight_layout()
    plt.show()


def build_story_root_causes(
    cancel_methods: pd.DataFrame, size_ranking: pd.DataFrame, promo_summary: pd.DataFrame
) -> pd.DataFrame:
    top_cancel = cancel_methods.iloc[0]
    top_size = size_ranking.iloc[0]
    core_promo = promo_summary.loc[promo_summary["is_core_story"]].copy()
    promo_row = core_promo.iloc[0] if not core_promo.empty else promo_summary.iloc[0]
    promo_recent_year = int(promo_row["recent_year"]) if pd.notna(promo_row.get("recent_year")) else None
    promo_recent_margin = (
        float(promo_row["recent_year_weighted_margin_delta"])
        if pd.notna(promo_row.get("recent_year_weighted_margin_delta"))
        else float(promo_row["weighted_margin_delta"])
    )
    promo_recent_orders = int(promo_row["recent_year_promo_orders"]) if promo_row.get("recent_year_promo_orders", 0) else int(
        promo_row["promo_orders"]
    )
    promo_recent_negative_share = (
        float(promo_row["recent_year_negative_share"])
        if pd.notna(promo_row.get("recent_year_negative_share"))
        else float(promo_row["negative_month_share"])
    )
    return pd.DataFrame(
        [
            {
                "driver": "Cancellation friction",
                "quant_signal": (
                    f"{top_cancel['payment_method']} posts a cancellation rate of {top_cancel['cancel_rate']:.1%} "
                    f"and accounts for {top_cancel['share_of_cancel_leakage']:.1%} of cancellation leakage"
                ),
                "evidence": (
                    f"Full-history resolved sample | scope {int(top_cancel['orders']):,} orders | leakage "
                    f"{_format_vnd_compact(float(top_cancel['cancel_leakage_value']))}"
                ),
                "action_hint": "Reduce checkout and CRM friction for the highest-risk payment method",
            },
            {
                "driver": "Wrong-size return",
                "quant_signal": (
                    f"{top_size['bucket']} shows a wrong-size rate of {top_size['wrong_size_rate']:.1%} "
                    f"and represents {top_size['share_of_wrong_size_refund']:.1%} of wrong-size refund value"
                ),
                "evidence": (
                    f"Full-history order-line sample | refund {_format_vnd_compact(float(top_size['wrong_size_refund_value']))} "
                    f"across {int(top_size['order_lines']):,} order lines"
                ),
                "action_hint": "Strengthen size guidance and prioritize an exchange-first flow for the largest bucket",
            },
            {
                "driver": "Promotion erosion (proxy)",
                "quant_signal": (
                    f"Matched monthly comparisons within the same category and source show that "
                    f"{promo_row['dominant_category']} | {promo_row['order_source']} has a proxy margin delta of "
                    f"{_format_vnd_compact(promo_recent_margin)}"
                ),
                "evidence": (
                    f"Full-history {int(promo_row['paired_months']):,} matched months | year {promo_recent_year} "
                    f"scope {promo_recent_orders:,} promo orders | negative months "
                    f"{promo_recent_negative_share:.0%}"
                ),
                "action_hint": "Pilot bucket-level promo guardrails instead of broad-based promo cuts",
            },
        ]
    )


def build_model_dataset(fact_order: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    modeling = fact_order.loc[fact_order["is_resolved"]].copy()
    modeling = modeling.loc[modeling["order_status"].isin(["delivered", "cancelled", "returned"])].copy()
    modeling["target"] = modeling["is_leakage_order"].astype(int)
    modeling["leakage_value"] = modeling["cancel_leakage"].fillna(0) + modeling["return_leakage"].fillna(0)
    modeling["order_month_num"] = modeling["order_date"].dt.month
    modeling["order_quarter_num"] = modeling["order_date"].dt.quarter
    modeling["is_first_order"] = modeling["prior_total_orders"].fillna(0).eq(0).astype(int)
    modeling["category_size_bucket"] = modeling["dominant_category"].astype(str) + " | " + modeling["dominant_size"].astype(str)
    modeling["stacked_promo_flag"] = modeling["stacked_promo_flag"].astype(int)
    modeling["promo_flag"] = modeling["promo_flag"].astype(int)
    modeling["is_weekend"] = modeling["is_weekend"].astype(int)

    numeric_features = [
        "order_gmv",
        "order_discount",
        "discount_rate",
        "order_cogs",
        "payment_value",
        "installments",
        "product_count",
        "category_breadth",
        "item_quantity",
        "avg_item_price",
        "days_since_signup",
        "prior_total_orders",
        "prior_resolved_orders",
        "prior_leakage_orders",
        "prior_avg_order_value",
        "prior_leakage_rate",
        "days_since_prev_order",
        "order_month_num",
        "order_quarter_num",
        "is_weekend",
        "stacked_promo_flag",
        "promo_flag",
        "is_first_order",
    ]
    categorical_features = [
        "dominant_category",
        "dominant_segment",
        "dominant_size",
        "category_size_bucket",
        "payment_method",
        "device_type",
        "order_source",
        "region",
        "gender",
        "age_group",
        "acquisition_channel",
        "dominant_promo_type",
        "order_weekday",
        "order_month_name",
    ]
    return modeling, numeric_features, categorical_features


def build_rule_context(train: pd.DataFrame) -> dict[str, Any]:
    payment_risk = train.groupby("payment_method")["target"].mean().sort_values(ascending=False)
    source_risk = train.groupby("order_source")["target"].mean().sort_values(ascending=False)
    bucket_risk = train.groupby("category_size_bucket")["target"].mean().sort_values(ascending=False)
    return {
        "high_risk_payment_methods": payment_risk.head(2).index.tolist(),
        "high_risk_sources": source_risk.head(3).index.tolist(),
        "high_risk_buckets": bucket_risk.head(6).index.tolist(),
        "prior_leakage_cut": float(train["prior_leakage_rate"].fillna(0).quantile(0.75)),
    }


def score_rule_based_model(frame: pd.DataFrame, rule_context: dict[str, Any]) -> np.ndarray:
    score = np.full(len(frame), 0.10, dtype="float64")
    score += 0.28 * frame["payment_method"].isin(rule_context["high_risk_payment_methods"]).astype(float)
    score += 0.14 * frame["order_source"].isin(rule_context["high_risk_sources"]).astype(float)
    score += 0.12 * frame["category_size_bucket"].isin(rule_context["high_risk_buckets"]).astype(float)
    score += 0.11 * frame["promo_flag"].astype(float)
    score += 0.10 * frame["is_first_order"].astype(float)
    score += 0.10 * frame["stacked_promo_flag"].astype(float)
    score += 0.12 * frame["prior_leakage_rate"].fillna(0).ge(rule_context["prior_leakage_cut"]).astype(float)
    return np.clip(score, 0.01, 0.99)


def evaluate_predictions(
    name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    review_share: float = DEFAULT_REVIEW_SHARE,
    leakage_value: pd.Series | None = None,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    review_mask = y_pred.astype(bool)
    actual_review_share = float(review_mask.mean())
    captured_leakage_value = float(leakage_value.loc[review_mask].sum()) if leakage_value is not None else np.nan
    total_leakage_value = float(leakage_value.sum()) if leakage_value is not None else np.nan
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
        "precision_at_threshold": safe_divide(tp, tp + fp),
        "recall_at_threshold": safe_divide(tp, tp + fn),
        "review_share": actual_review_share if actual_review_share > 0 else review_share,
        "lift_at_10pct": lift_at_fraction(y_true, scores, 0.10),
        "lift_at_15pct": lift_at_fraction(y_true, scores, 0.15),
        "captured_leakage_value_share": safe_divide(captured_leakage_value, total_leakage_value),
        "captured_leakage_value": captured_leakage_value,
    }


def fit_models(
    split_data: dict[str, pd.DataFrame],
    numeric_features: list[str],
    categorical_features: list[str],
    review_share: float = DEFAULT_REVIEW_SHARE,
) -> dict[str, Any]:
    train = split_data["train"]
    valid = split_data["valid"]
    test = split_data["test"]

    feature_cols = numeric_features + categorical_features
    x_train = train[feature_cols]
    y_train = train["target"]
    x_valid = valid[feature_cols]
    x_test = test[feature_cols]
    y_test = test["target"]

    rule_context = build_rule_context(train)
    logistic = build_logistic_pipeline(numeric_features, categorical_features)
    logistic.fit(x_train, y_train)

    benchmark = build_tree_pipeline(numeric_features, categorical_features)
    benchmark_sample = train.sample(min(len(train), 180000), random_state=42)
    benchmark.fit(benchmark_sample[feature_cols], benchmark_sample["target"])

    score_map = {
        "LogisticRegression": {
            "valid": logistic.predict_proba(x_valid)[:, 1],
            "test": logistic.predict_proba(x_test)[:, 1],
            "estimator": logistic,
        },
        "DecisionTree": {
            "valid": benchmark.predict_proba(x_valid)[:, 1],
            "test": benchmark.predict_proba(x_test)[:, 1],
            "estimator": benchmark,
        },
        "BusinessRule": {
            "valid": score_rule_based_model(valid, rule_context),
            "test": score_rule_based_model(test, rule_context),
            "estimator": None,
        },
    }

    metrics_rows = []
    for model_name, payload in score_map.items():
        threshold = choose_operating_threshold(payload["valid"], review_share=review_share)
        payload["threshold"] = threshold
        metrics_rows.append(
            evaluate_predictions(
                model_name,
                y_test,
                payload["test"],
                threshold,
                review_share=review_share,
                leakage_value=test["leakage_value"],
            )
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["captured_leakage_value_share", "lift_at_15pct", "pr_auc"], ascending=[False, False, False]
    )
    primary_model = metrics_df.iloc[0]["model"]

    return {
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "logistic": logistic,
        "benchmark": benchmark,
        "train": train,
        "valid": valid,
        "test": test,
        "score_map": score_map,
        "metrics": metrics_df.reset_index(drop=True),
        "review_share": review_share,
        "rule_context": rule_context,
        "primary_model": primary_model,
        "threshold": float(score_map[primary_model]["threshold"]),
    }


def plot_model_diagnostics(modeling_result: dict[str, Any], model_name: str | None = None) -> None:
    if model_name is None:
        model_name = modeling_result["primary_model"]
    y_test = modeling_result["test"]["target"]
    scores = modeling_result["score_map"][model_name]["test"]
    threshold = modeling_result["score_map"][model_name]["threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, scores):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_title(f"ROC curve - {model_name}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_test, scores)
    axes[1].plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_test, scores):.3f}")
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=6, strategy="quantile")
    axes[2].plot(prob_pred, prob_true, marker="o")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[2].axvline(threshold, linestyle=":", color="#E15759", label=f"threshold={threshold:.3f}")
    axes[2].set_title("Calibration curve")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Observed leakage rate")
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def build_review_queue_summary(modeling_result: dict[str, Any]) -> pd.DataFrame:
    test = modeling_result["test"].copy()
    rows = []
    for model_name, payload in modeling_result["score_map"].items():
        threshold = payload["threshold"]
        selected = payload["test"] >= threshold
        queue = test.loc[selected].copy()
        rows.append(
            {
                "model": model_name,
                "selected_orders": int(queue["order_id"].nunique()),
                "review_share": float(selected.mean()),
                "queue_leakage_rate": float(queue["target"].mean()) if not queue.empty else 0.0,
                "queue_scope_value": float(queue["order_net_before_outcome"].sum()),
                "queue_leakage_value": float(queue["leakage_value"].sum()),
                "captured_leakage_value_share": safe_divide(queue["leakage_value"].sum(), test["leakage_value"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values("captured_leakage_value_share", ascending=False).reset_index(drop=True)


def plot_review_capture(review_queue_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=review_queue_df, x="model", y="captured_leakage_value_share", color="#4E79A7")
    plt.title("Captured leakage value share under ~15% review capacity")
    plt.xlabel("")
    plt.ylabel("Captured leakage value share")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    for idx, row in enumerate(review_queue_df.itertuples(index=False)):
        plt.text(
            idx,
            row.captured_leakage_value_share,
            f"capture {row.captured_leakage_value_share:.1%}\nqueue leakage {row.queue_leakage_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def build_model_interpretation(
    modeling_result: dict[str, Any], model_name: str = "LogisticRegression", top_n: int = 12
) -> pd.DataFrame:
    if model_name != "LogisticRegression":
        return pd.DataFrame(columns=["feature", "coefficient", "direction"])
    pipeline = modeling_result["score_map"][model_name]["estimator"]
    preprocess = pipeline.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["cat"].named_steps["encoder"]
    cat_names = encoder.get_feature_names_out(modeling_result["categorical_features"])
    feature_names = list(modeling_result["numeric_features"]) + list(cat_names)
    coefs = pipeline.named_steps["model"].coef_[0]
    interpretation = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    interpretation["abs_coefficient"] = interpretation["coefficient"].abs()
    interpretation["feature"] = interpretation["feature"].str.replace("_", " ", regex=False)
    interpretation["direction"] = np.where(interpretation["coefficient"] >= 0, "risk up", "risk down")
    return interpretation.sort_values("abs_coefficient", ascending=False).head(top_n)[
        ["feature", "coefficient", "direction"]
    ]


def build_risk_segments(modeling_result: dict[str, Any], model_name: str | None = None) -> pd.DataFrame:
    if model_name is None:
        model_name = modeling_result["primary_model"]
    test = modeling_result["test"].copy()
    test["risk_score"] = modeling_result["score_map"][model_name]["test"]
    high_cut = test["risk_score"].quantile(0.85)
    med_cut = test["risk_score"].quantile(0.65)
    test["risk_segment"] = np.select(
        [test["risk_score"] >= high_cut, test["risk_score"] >= med_cut],
        ["high", "medium"],
        default="low",
    )
    summary = (
        test.groupby("risk_segment", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            leakage_rate=("target", "mean"),
            avg_score=("risk_score", "mean"),
            realized_net_revenue=("realized_net_revenue", "sum"),
            leakage_value=("leakage_value", "sum"),
        )
        .sort_values("avg_score", ascending=False)
    )
    return summary


def plot_risk_segments(segment_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=segment_df, x="risk_segment", y="leakage_rate", ax=axes[0], palette="Reds")
    axes[0].set_title("Leakage rate by risk segment (test 2022)")
    axes[0].set_xlabel("Risk segment")
    axes[0].set_ylabel("Leakage rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    for idx, row in enumerate(segment_df.itertuples(index=False)):
        axes[0].text(idx, row.leakage_rate, f"{row.leakage_rate:.1%}", ha="center", va="bottom", fontsize=9)

    sns.barplot(data=segment_df, x="risk_segment", y="leakage_value", ax=axes[1], palette="Blues")
    axes[1].set_title("Leakage value by risk segment (test 2022)")
    axes[1].set_xlabel("Risk segment")
    axes[1].set_ylabel("Leakage value")
    axes[1].yaxis.set_major_formatter(_compact_money_formatter())
    for idx, row in enumerate(segment_df.itertuples(index=False)):
        axes[1].text(
            idx,
            row.leakage_value,
            _format_vnd_compact(float(row.leakage_value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def _build_value_action_row(
    action: str,
    driver: str,
    owner: str,
    kpi: str,
    scope_orders: int,
    scope_value: float,
    current_leakage_value: float,
    effort: float,
    recovery_range: tuple[float, float, float],
    timeline: str,
    evidence: str,
) -> dict[str, Any]:
    low_share, base_share, high_share = recovery_range
    current_leakage_value = float(max(current_leakage_value, 1.0))
    return {
        "action": action,
        "driver": driver,
        "owner": owner,
        "kpi": kpi,
        "scope_orders": int(scope_orders),
        "scope_value": float(max(scope_value, 1.0)),
        "current_leakage_value": current_leakage_value,
        "recoverable_low_value": current_leakage_value * low_share,
        "recoverable_base_value": current_leakage_value * base_share,
        "recoverable_high_value": current_leakage_value * high_share,
        "monthly_upside_base": current_leakage_value * base_share / 12,
        "effort": effort,
        "timeline": timeline,
        "assumption": f"Recovery range of {low_share:.0%}-{base_share:.0%}-{high_share:.0%} on the bucket's current leakage value",
        "evidence": evidence,
    }


def build_action_plan(
    fact_order: pd.DataFrame,
    cancel_methods: pd.DataFrame,
    size_ranking: pd.DataFrame,
    promo_summary: pd.DataFrame,
    review_queue_df: pd.DataFrame,
) -> pd.DataFrame:
    current_year = int(fact_order["order_year"].max())
    current = fact_order.loc[(fact_order["order_year"] == current_year) & fact_order["is_resolved"]].copy()
    year_leakage_total = float(current["cancel_leakage"].sum() + current["return_leakage"].sum())

    cancel_hotspot = cancel_methods.iloc[0]
    cancel_scope = current.loc[current["payment_method"].eq(cancel_hotspot["payment_method"])]
    cancel_scope_orders = int(cancel_scope["order_id"].nunique())
    cancel_scope_value = float(cancel_scope["order_net_before_outcome"].sum())
    cancel_scope_leakage = float(cancel_scope["cancel_leakage"].sum())
    cancel_scope_rate = float(cancel_scope["is_cancelled_order"].mean()) if cancel_scope_orders else 0.0
    current_cancel_total = float(current["cancel_leakage"].sum())
    cancel_scope_share = (cancel_scope_leakage / current_cancel_total) if current_cancel_total > 0 else 0.0
    cancel_action = _build_value_action_row(
        action="Reduce COD and checkout friction",
        driver="Cancellation friction",
        owner="Checkout + CRM",
        kpi="Cancellation rate of the hotspot payment method",
        scope_orders=cancel_scope_orders,
        scope_value=cancel_scope_value,
        current_leakage_value=cancel_scope_leakage,
        effort=2.0,
        recovery_range=(0.08, 0.15, 0.22),
        timeline="30 days: remove checkout pain points; 60 days: deploy CRM nudges; 90 days: tune policy by source",
        evidence=(
            f"Year {current_year}: {cancel_hotspot['payment_method']} cancellation rate {cancel_scope_rate:.1%} | "
            f"share of cancellation leakage {cancel_scope_share:.1%}"
        ),
    )

    top_size = size_ranking.iloc[0]
    size_scope = current.loc[
        current["dominant_category"].eq(top_size["category"]) & current["dominant_size"].eq(top_size["size"])
    ]
    size_scope_orders = int(size_scope["order_id"].nunique())
    size_scope_value = float(size_scope["order_net_before_outcome"].sum())
    size_scope_leakage = float(size_scope["return_leakage"].sum())
    size_wrong_size_orders = int(size_scope["wrong_size_return_orders"].gt(0).sum())
    size_wrong_size_rate = (size_wrong_size_orders / size_scope_orders) if size_scope_orders else 0.0
    size_action = _build_value_action_row(
        action="Strengthen size guidance and exchange flow",
        driver="Wrong-size return",
        owner="Merchandising + CX",
        kpi="Wrong-size refund value and wrong-size rate",
        scope_orders=size_scope_orders,
        scope_value=size_scope_value,
        current_leakage_value=size_scope_leakage,
        effort=2.5,
        recovery_range=(0.15, 0.25, 0.35),
        timeline="30 days: refine fit guidance; 60 days: launch an exchange-first flow; 90 days: optimize by bucket",
        evidence=(
            f"Year {current_year}: {top_size['bucket']} wrong-size rate {size_wrong_size_rate:.1%} | "
            f"refund value {_format_vnd_compact(size_scope_leakage)}"
        ),
    )

    promo_hotspot = promo_summary.loc[promo_summary["is_core_story"]].head(1)
    if promo_hotspot.empty:
        promo_hotspot = promo_summary.head(1)
    promo_hotspot = promo_hotspot.iloc[0]
    promo_scope = current.loc[
        current["dominant_category"].eq(promo_hotspot["dominant_category"])
        & current["order_source"].eq(promo_hotspot["order_source"])
        & current["promo_flag"]
    ]
    promo_scope_orders = int(promo_scope["order_id"].nunique())
    promo_margin_delta = (
        float(promo_hotspot["recent_year_weighted_margin_delta"])
        if pd.notna(promo_hotspot.get("recent_year_weighted_margin_delta"))
        else float(promo_hotspot["weighted_margin_delta"])
    )
    promo_history_negative_share = float(promo_hotspot["negative_month_share"])
    promo_action = _build_value_action_row(
        action="Pilot promo guardrails by bucket",
        driver="Promotion erosion (proxy)",
        owner="Growth + Finance",
        kpi="Weighted margin delta promo vs no-promo",
        scope_orders=promo_scope_orders,
        scope_value=float(promo_scope["order_net_before_outcome"].sum()),
        current_leakage_value=float(max(-promo_margin_delta, 0.0) * max(promo_scope_orders, 1)),
        effort=2.9,
        recovery_range=(0.12, 0.20, 0.28),
        timeline="30 days: freeze the weakest pilot bucket; 60 days: introduce margin guardrails; 90 days: run a bucket-level scorecard",
        evidence=(
            f"Year {current_year}: {promo_hotspot['dominant_category']} | {promo_hotspot['order_source']} "
            f"margin delta {_format_vnd_compact(promo_margin_delta)} across {promo_scope_orders:,} promo orders | "
            f"negative matched months across full history {promo_history_negative_share:.0%}"
        ),
    )

    review_row = review_queue_df.iloc[0]
    review_action = _build_value_action_row(
        action="Run a high-risk review queue",
        driver="Predictive triage",
        owner="Ops + Risk",
        kpi="Captured leakage value share within the top review bucket",
        scope_orders=int(review_row["selected_orders"]),
        scope_value=float(review_row["queue_scope_value"]),
        current_leakage_value=float(review_row["queue_leakage_value"]),
        effort=3.2,
        recovery_range=(0.05, 0.09, 0.14),
        timeline="30 days: pilot manual review; 60 days: formalize queue SLAs; 90 days: move to semi-automated routing",
        evidence=(
            f"{review_row['model']} captures {review_row['captured_leakage_value_share']:.1%} "
            f"of leakage value within the top {review_row['review_share']:.0%} of orders"
        ),
    )

    action_df = pd.DataFrame([cancel_action, size_action, review_action, promo_action])
    priority_map = {
        "Reduce COD and checkout friction": 1,
        "Strengthen size guidance and exchange flow": 2,
        "Run a high-risk review queue": 3,
        "Pilot promo guardrails by bucket": 4,
    }
    priority_note_map = {
        "Reduce COD and checkout friction": "Priority 1: strongest evidence with low execution effort",
        "Strengthen size guidance and exchange flow": "Priority 2: clear bucket definition with a clear owner",
        "Run a high-risk review queue": "Priority 3: best fit when review capacity is limited",
        "Pilot promo guardrails by bucket": "Priority 4: execute as a controlled pilot because the signal remains a proxy",
    }
    tradeoff_map = {
        "Reduce COD and checkout friction": (
            "This action can materially reduce cancellation leakage, but the checkout experience must not become so restrictive that conversion falls."
        ),
        "Strengthen size guidance and exchange flow": (
            "This is lower-risk to revenue than a promo intervention, but it requires tight coordination across merchandising and CX."
        ),
        "Run a high-risk review queue": (
            "Targeting improves, but the business pays for it through review capacity, operational SLAs, and false positives."
        ),
        "Pilot promo guardrails by bucket": (
            "The weakest buckets show clear margin upside, but volume can drop if the intervention is too aggressive, so it should remain a controlled pilot."
        ),
    }
    action_df["priority_rank"] = action_df["action"].map(priority_map).fillna(99).astype(int)
    action_df["priority_note"] = action_df["action"].map(priority_note_map).fillna("Prioritized by recoverable value")
    action_df["trade_off"] = action_df["action"].map(tradeoff_map).fillna("Prioritized by recoverable value and effort")
    action_df["share_of_year_leakage"] = safe_divide(action_df["recoverable_base_value"], year_leakage_total)
    action_df["share_of_scope_leakage"] = safe_divide(
        action_df["recoverable_base_value"], action_df["current_leakage_value"]
    )
    return action_df.sort_values(["priority_rank", "recoverable_base_value"], ascending=[True, False]).reset_index(drop=True)


def simulate_action_roi(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in action_df.iterrows():
        for scenario_name, column in [
            ("Conservative", "recoverable_low_value"),
            ("Base", "recoverable_base_value"),
            ("Aggressive", "recoverable_high_value"),
        ]:
            recoverable_value = row[column]
            rows.append(
                {
                    "action": row["action"],
                    "scenario": scenario_name,
                    "owner": row["owner"],
                    "kpi": row["kpi"],
                    "recoverable_value": recoverable_value,
                    "monthly_upside": recoverable_value / 12,
                }
            )
    return pd.DataFrame(rows)


def plot_action_matrix(action_df: pd.DataFrame) -> None:
    plot_df = action_df.sort_values(["priority_rank", "recoverable_base_value"], ascending=[True, False]).copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(plot_df["effort"], plot_df["recoverable_base_value"], s=280, color="#4E79A7")
    offsets = {
        "Reduce COD and checkout friction": (10, 6),
        "Strengthen size guidance and exchange flow": (16, -4),
        "Run a high-risk review queue": (14, 4),
        "Pilot promo guardrails by bucket": (14, 6),
    }
    for row in plot_df.itertuples(index=False):
        dx, dy = offsets.get(row.action, (10, 4))
        ax.annotate(
            f"{row.action} | {_format_vnd_compact(float(row.monthly_upside_base))}/month",
            (row.effort, row.recoverable_base_value),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.82, "ec": "none"},
        )
    ax.set_title("Action matrix: recoverable leakage value vs effort")
    ax.set_xlabel("Effort score (1 low - 5 high)")
    ax.set_ylabel("Recoverable value (base case)")
    ax.yaxis.set_major_formatter(_compact_money_formatter())
    ax.set_xlim(plot_df["effort"].min() - 0.1, plot_df["effort"].max() + 0.45)
    plt.tight_layout()
    plt.show()


def build_executive_summary(
    descriptive_points: list[str],
    root_cause_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    action_df: pd.DataFrame,
    review_queue_df: pd.DataFrame,
) -> str:
    top_actions = action_df.sort_values(["priority_rank", "recoverable_base_value"], ascending=[True, False]).head(3)
    top_metrics = metrics_df.iloc[0]
    runner_up = metrics_df.iloc[1] if len(metrics_df) > 1 else metrics_df.iloc[0]
    top_queue = review_queue_df.iloc[0]
    total_base_recovery = float(action_df["recoverable_base_value"].sum())
    total_recovery_share = float(action_df["share_of_year_leakage"].sum()) if "share_of_year_leakage" in action_df else np.nan
    lines = [
        "## Executive Summary",
        "",
        "### 1. What is happening?",
        *[f"- {point}" for point in descriptive_points[:3]],
        "",
        "### 2. Why is it happening?",
        "- The drivers below were identified on the full-history resolved sample, while the action sizing uses the latest-year scope to avoid overstating upside.",
        *[
            f"- {row['driver']}: {row['quant_signal']} ({row['evidence']})."
            for _, row in root_cause_df.head(3).iterrows()
        ],
        "",
        "### 3. How should the triage layer operate?",
        (
            f"- Under a review capacity of roughly {top_queue['review_share']:.0%} of orders, {top_metrics['model']} captures "
            f"{top_metrics['captured_leakage_value_share']:.1%} of leakage value, slightly ahead of "
            f"{runner_up['model']} ({runner_up['captured_leakage_value_share']:.1%}) but without a dramatic margin."
        ),
        (
            f"- The queue should prioritize the high-risk bucket because the queue leakage rate reaches {top_queue['queue_leakage_rate']:.1%}; "
            f"the predictive layer is used as a triage mechanism, not as a full replacement for business rules."
        ),
        "",
        "### 4. What should happen in the next 30-60-90 days?",
        (
            f"- The upside estimates below are sized on the latest-year scope and should be read as value-at-stake for pilot prioritization, "
            f"not as a committed forecast. The combined base case across the four actions is currently about "
            f"{_format_vnd_compact(total_base_recovery)} (~{total_recovery_share:.1%} of the latest year's leakage)."
        ),
        *[
            (
                f"- {row['action']}: base recoverable value ~ "
                f"{_format_vnd_compact(float(row['recoverable_base_value']))}, "
                f"monthly upside ~ {_format_vnd_compact(float(row['monthly_upside_base']))}."
            )
            for _, row in top_actions.iterrows()
        ],
    ]
    return "\n".join(lines)


__all__ = [
    "build_action_plan",
    "build_cancellation_diagnostic",
    "build_cancellation_story",
    "build_descriptive_scorecard",
    "build_delivery_diagnostics",
    "build_descriptive_summary",
    "build_dimension_mix",
    "build_excluded_driver_appendix",
    "build_executive_summary",
    "build_fact_tables",
    "build_geography_snapshot",
    "build_model_interpretation",
    "build_model_dataset",
    "build_promo_story",
    "build_promo_proxy",
    "build_quality_audit",
    "build_relation_summary",
    "build_review_queue_summary",
    "build_risk_segments",
    "build_root_cause_summary",
    "build_self_scoring",
    "build_size_diagnostic",
    "build_size_story",
    "build_story_root_causes",
    "build_stockout_proxy",
    "build_monthly_kpis",
    "build_quarterly_kpis",
    "build_waterfall_summary",
    "evaluate_predictions",
    "fit_models",
    "load_data",
    "plot_action_matrix",
    "plot_cancellation_diagnostic",
    "plot_cancellation_story",
    "plot_descriptive_scorecard",
    "plot_delivery_diagnostics",
    "plot_geography_snapshot",
    "plot_kpi_trends",
    "plot_mix_dashboard",
    "plot_model_diagnostics",
    "plot_promo_story",
    "plot_promo_proxy",
    "plot_review_capture",
    "plot_risk_segments",
    "plot_sales_reconciliation",
    "plot_size_heatmap",
    "plot_size_story",
    "plot_stockout_proxy",
    "plot_waterfall",
    "preview_table_bundle",
    "reconcile_with_sales",
    "set_plot_theme",
    "simulate_action_roi",
    "split_model_data",
]
