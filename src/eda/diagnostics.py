from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.eda.data import _format_vnd_compact, safe_divide


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


def build_cancellation_diagnostic(
    fact_order: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
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
            {
                "metric": "Overall cancellation rate",
                "value": overall_cancel_rate,
                "note": "Cancellation rate across all resolved orders",
            },
            {
                "metric": f"{top_method['payment_method']} cancellation rate",
                "value": float(top_method["cancel_rate"]),
                "note": f"Lift of {top_method['rate_lift_vs_avg']:.2f}x versus the baseline",
            },
            {
                "metric": f"{top_method['payment_method']} vs non-{top_method['payment_method']} gap",
                "value": float(top_method["cancel_rate"] - resolved.loc[~top_method_mask, "is_cancelled_order"].mean()),
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
        _chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
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
                stats.binomtest(recent_year_negative_margin_months, recent_year_months, 0.5, alternative="greater").pvalue
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
        "headline": f"corr(stockout_rate, leakage_rate) = {stockout_stats['corr_stockout_vs_leakage']:.3f}",
        "evidence": (
            f"corr(fill_rate, realized_net_revenue) = {stockout_stats['corr_fill_vs_net']:.3f} | "
            f"hotspot {top_stockout['dominant_category']} stockout {top_stockout['stockout_rate']:.1%}"
        ),
        "decision": "Kept in the appendix because the correlation is weak; the signal is better suited for monitoring than for the main root-cause story.",
    }

    return pd.DataFrame([delivery_summary, stockout_summary])


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
    promo_recent_orders = (
        int(promo_row["recent_year_promo_orders"])
        if promo_row.get("recent_year_promo_orders", 0)
        else int(promo_row["promo_orders"])
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
                    "Matched monthly comparisons within the same category and source show that "
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


__all__ = [
    "build_cancellation_diagnostic",
    "build_cancellation_story",
    "build_delivery_diagnostics",
    "build_excluded_driver_appendix",
    "build_promo_proxy",
    "build_promo_story",
    "build_root_cause_summary",
    "build_size_diagnostic",
    "build_size_story",
    "build_stockout_proxy",
    "build_story_root_causes",
]
