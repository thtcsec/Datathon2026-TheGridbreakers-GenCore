from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.transforms import offset_copy
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.eda.data import _format_vnd_compact, build_dimension_mix, safe_divide


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


def plot_size_heatmap(pivot: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlOrRd")
    plt.title("Wrong-size return rate theo category x size")
    plt.xlabel("Size")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()


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


def plot_cancellation_story(method_summary: pd.DataFrame, combo_summary: pd.DataFrame) -> None:
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
    sample["plot_orders"] = sample["recent_year_promo_orders"].where(sample["recent_year_promo_orders"].gt(0), sample["promo_orders"])
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


__all__ = [
    "plot_action_matrix",
    "plot_cancellation_diagnostic",
    "plot_cancellation_story",
    "plot_delivery_diagnostics",
    "plot_descriptive_scorecard",
    "plot_geography_snapshot",
    "plot_kpi_trends",
    "plot_mix_dashboard",
    "plot_model_diagnostics",
    "plot_promo_proxy",
    "plot_promo_story",
    "plot_review_capture",
    "plot_risk_segments",
    "plot_sales_reconciliation",
    "plot_size_heatmap",
    "plot_size_story",
    "plot_stockout_proxy",
    "plot_waterfall",
]
