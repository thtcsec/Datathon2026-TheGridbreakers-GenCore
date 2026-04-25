import importlib
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

msno = None
try:
    msno = importlib.import_module("missingno")
    HAS_MISSINGNO = True
except Exception:
    HAS_MISSINGNO = False


def _mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error (%).

    Uses a small *eps* floor on the denominator to avoid division-by-zero
    when actual values are exactly 0.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def _smape(y_true, y_pred, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error (%).

    Bounded in [0, 200], treats over- and under-predictions symmetrically —
    easier to interpret for business stakeholders.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MAE, RMSE, R2, MAPE (%), sMAPE (%).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    smape = _smape(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "sMAPE": smape,
    }

def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and print evaluation metrics.
    """
    metrics = calculate_metrics(y_true, y_pred)
    print(f"--- {model_name} Performance ---")
    print(f"MAE:   {metrics['MAE']:,.2f}")
    print(f"RMSE:  {metrics['RMSE']:,.2f}")
    print(f"R2:    {metrics['R2']:.4f}")
    print(f"MAPE:  {metrics['MAPE']:.2f}%")
    print(f"sMAPE: {metrics['sMAPE']:.2f}%")
    return metrics



DEFAULT_DATA_PATH = "../data/raw/"
GLOBAL_DATE_MIN = pd.Timestamp("2012-01-01")
GLOBAL_DATE_MAX = pd.Timestamp("2024-12-31")
PLOT_SAMPLE_SIZE = 120000
MISSING_MATRIX_SAMPLE_SIZE = 2000

TABLE_ORDER = [
    "products.csv",
    "customers.csv",
    "promotions.csv",
    "geography.csv",
    "orders.csv",
    "order_items.csv",
    "payments.csv",
    "shipments.csv",
    "returns.csv",
    "reviews.csv",
    "sales.csv",
    "sample_submission.csv",
    "inventory.csv",
    "web_traffic.csv",
]

TABLE_CONFIG = {
    "products.csv": {
        "layer": "Master",
        "pk": ["product_id"],
        "distribution_cols": ["price", "cogs"],
        "box": {"value": "price", "by": "category"},
        "corr_cols": ["price", "cogs"],
    },
    "customers.csv": {
        "layer": "Master",
        "pk": ["customer_id"],
        "date_cols": ["signup_date"],
    },
    "promotions.csv": {
        "layer": "Master",
        "pk": ["promo_id"],
        "date_cols": ["start_date", "end_date"],
        "distribution_cols": ["discount_value", "min_order_value"],
        "box": {"value": "discount_value", "by": "promo_type"},
    },
    "geography.csv": {
        "layer": "Master",
        "pk": ["zip"],
    },
    "orders.csv": {
        "layer": "Transaction",
        "pk": ["order_id"],
        "date_cols": ["order_date"],
        "time_col": "order_date",
    },
    "order_items.csv": {
        "layer": "Transaction",
        "pk": ["order_id", "product_id"],
        "distribution_cols": ["quantity", "unit_price", "discount_amount"],
        "box": {"value": "unit_price", "by": "promo_id"},
    },
    "payments.csv": {
        "layer": "Transaction",
        "pk": ["order_id"],
        "distribution_cols": ["payment_value", "installments"],
        "box": {"value": "payment_value", "by": "payment_method"},
    },
    "shipments.csv": {
        "layer": "Transaction",
        "pk": ["order_id"],
        "date_cols": ["ship_date", "delivery_date"],
        "distribution_cols": ["shipping_fee"],
        "time_col": "ship_date",
    },
    "returns.csv": {
        "layer": "Transaction",
        "pk": ["return_id"],
        "date_cols": ["return_date"],
        "distribution_cols": ["return_quantity", "refund_amount"],
        "box": {"value": "refund_amount", "by": "return_reason"},
    },
    "reviews.csv": {
        "layer": "Transaction",
        "pk": ["review_id"],
        "date_cols": ["review_date"],
        "distribution_cols": ["rating"],
        "box": {"value": "rating", "by": "review_title"},
    },
    "sales.csv": {
        "layer": "Analytical",
        "pk": ["Date"],
        "date_cols": ["Date"],
        "distribution_cols": ["Revenue", "COGS"],
        "time_col": "Date",
        "seasonality_value_col": "Revenue",
        "seasonality_agg": "sum",
        "corr_cols": ["Revenue", "COGS"],
    },
    "sample_submission.csv": {
        "layer": "Analytical",
        "pk": ["Date"],
        "date_cols": ["Date"],
        "distribution_cols": ["Revenue", "COGS"],
        "time_col": "Date",
        "seasonality_value_col": "Revenue",
        "seasonality_agg": "sum",
        "corr_cols": ["Revenue", "COGS"],
    },
    "inventory.csv": {
        "layer": "Operational",
        "pk": ["snapshot_date", "product_id"],
        "date_cols": ["snapshot_date"],
        "distribution_cols": ["stock_on_hand", "days_of_supply", "fill_rate"],
        "box": {"value": "stock_on_hand", "by": "category"},
        "time_col": "snapshot_date",
        "seasonality_value_col": "units_sold",
        "seasonality_agg": "sum",
    },
    "web_traffic.csv": {
        "layer": "Operational",
        "pk": ["date", "traffic_source"],
        "date_cols": ["date"],
        "distribution_cols": ["sessions", "page_views", "conversion_rate"],
        "box": {"value": "sessions", "by": "traffic_source"},
        "time_col": "date",
        "seasonality_value_col": "sessions",
        "seasonality_agg": "sum",
    },
}


def load_table(file_name, data_path=DEFAULT_DATA_PATH):
    """Load one CSV table from the configured raw data directory."""
    file_path = os.path.join(data_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")
    return pd.read_csv(file_path, low_memory=False)


def infer_id_columns(df):
    """Infer likely identifier columns from a dataframe schema."""
    return [c for c in df.columns if c.lower().endswith("_id") or c.lower() in {"id", "zip"}]


def infer_date_columns(df):
    """Infer likely date columns from a dataframe schema."""
    return [c for c in df.columns if "date" in c.lower() or c == "Date"]


def count_iqr_outliers(series):
    """Count outliers using the IQR rule on a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return 0

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def profile_table(
    file_name,
    data_path=DEFAULT_DATA_PATH,
    table_config=None,
    global_date_min=GLOBAL_DATE_MIN,
    global_date_max=GLOBAL_DATE_MAX,
):
    """Build a profiling dictionary and return (dataframe, profile)."""
    if table_config is None:
        table_config = TABLE_CONFIG

    cfg = table_config.get(file_name, {})
    layer = cfg.get("layer", "Unknown")
    df = load_table(file_name, data_path=data_path)

    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    configured_pk = [c for c in cfg.get("pk", []) if c in df.columns]
    pk_cols = configured_pk if configured_pk else infer_id_columns(df)

    configured_dates = [c for c in cfg.get("date_cols", []) if c in df.columns]
    date_cols = configured_dates if configured_dates else infer_date_columns(df)

    dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})

    missing_count = df.isna().sum().sort_values(ascending=False)
    missing_pct = ((missing_count / max(len(df), 1)) * 100).round(2)
    missing_df = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
        }
    )
    missing_df = missing_df[missing_df["missing_count"] > 0]

    duplicate_rows = int(df.duplicated().sum())

    pk_unique = None
    if pk_cols:
        pk_unique = bool(not df.duplicated(subset=pk_cols).any())

    id_float_cols = [c for c in pk_cols if c in df.columns and pd.api.types.is_float_dtype(df[c])]
    date_object_cols = [
        c
        for c in date_cols
        if c in df.columns and (df[c].dtype == "object" or str(df[c].dtype).startswith("string"))
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    zero_counts = {}
    negative_counts = {}
    for c in numeric_cols:
        z = int((df[c] == 0).sum())
        if z > 0:
            zero_counts[c] = z

        n = int((df[c] < 0).sum())
        if n > 0:
            negative_counts[c] = n

    date_ranges = {}
    date_outside_window = {}
    for c in date_cols:
        parsed = pd.to_datetime(df[c], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            continue

        below_min = int((valid < global_date_min).sum())
        above_max = int((valid > global_date_max).sum())
        date_ranges[c] = {
            "min": str(valid.min().date()),
            "max": str(valid.max().date()),
            "invalid": int(parsed.isna().sum()),
        }
        if below_min > 0 or above_max > 0:
            date_outside_window[c] = {"below_min": below_min, "above_max": above_max}

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cardinality_df = pd.DataFrame(
        {
            "column": cat_cols,
            "unique_values": [int(df[c].nunique(dropna=True)) for c in cat_cols],
        }
    ).sort_values("unique_values", ascending=False)

    outlier_candidates = [c for c in cfg.get("distribution_cols", []) if c in numeric_cols]
    if not outlier_candidates:
        outlier_candidates = [c for c in numeric_cols if df[c].nunique(dropna=True) > 20][:2]

    outlier_counts = {c: count_iqr_outliers(df[c]) for c in outlier_candidates}
    outlier_counts = {k: v for k, v in outlier_counts.items() if v > 0}

    logic_issues = []
    if not missing_df.empty:
        max_missing = float(missing_df["missing_pct"].max())
        max_missing_col = str(missing_df.iloc[0]["column"])
        if max_missing > 30:
            logic_issues.append(f"high_missing:{max_missing_col}={max_missing:.2f}%")
        elif max_missing > 5:
            logic_issues.append(f"medium_missing:{max_missing_col}={max_missing:.2f}%")

    if duplicate_rows > 0:
        logic_issues.append(f"exact_duplicates={duplicate_rows}")

    if pk_unique is False:
        logic_issues.append("pk_not_unique")

    if id_float_cols:
        logic_issues.append("id_as_float:" + ",".join(id_float_cols))

    if date_object_cols:
        logic_issues.append("date_as_object:" + ",".join(date_object_cols))

    if negative_counts:
        logic_issues.append("negative_numeric:" + ",".join(negative_counts.keys()))

    if date_outside_window:
        logic_issues.append("date_outside_expected_window")

    if outlier_counts:
        top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        logic_issues.append("outlier_iqr:" + ",".join([f"{k}={v}" for k, v in top_outlier_cols]))

    if not logic_issues:
        logic_issues = ["no_major_issue_detected"]

    profile = {
        "file_name": file_name,
        "layer": layer,
        "shape": (rows, cols),
        "memory_mb": round(float(memory_mb), 3),
        "pk_cols": pk_cols,
        "pk_unique": pk_unique,
        "date_cols": date_cols,
        "dtypes_df": dtypes_df,
        "missing_df": missing_df.sort_values("missing_count", ascending=False),
        "duplicate_rows": duplicate_rows,
        "id_float_cols": id_float_cols,
        "date_object_cols": date_object_cols,
        "zero_counts": zero_counts,
        "negative_counts": negative_counts,
        "date_ranges": date_ranges,
        "date_outside_window": date_outside_window,
        "cardinality_df": cardinality_df,
        "outlier_counts": outlier_counts,
        "logic_issues": logic_issues,
    }
    return df, profile


def build_raw_audit_row(profile):
    """Convert one profile dictionary into one audit report row."""
    type_issues = []
    if profile["id_float_cols"]:
        type_issues.append("id_as_float:" + ",".join(profile["id_float_cols"]))
    if profile["date_object_cols"]:
        type_issues.append("date_as_object:" + ",".join(profile["date_object_cols"]))
    type_issue_text = "; ".join(type_issues) if type_issues else "none"

    if profile["missing_df"].empty:
        max_missing_text = "0.00% (none)"
    else:
        first_row = profile["missing_df"].iloc[0]
        max_missing_text = f"{first_row['missing_pct']:.2f}% ({first_row['column']})"

    logic_text = "; ".join(profile["logic_issues"][:3])

    return {
        "Ten File": profile["file_name"],
        "Tong dong": profile["shape"][0],
        "Cot loi Type": type_issue_text,
        "% Missing cao nhat": max_missing_text,
        "So dong trung": profile["duplicate_rows"],
        "Van de Logic (Outlier/Min-Max)": logic_text,
    }


def _dict_to_df(data_dict, key_name="column", value_name="value"):
    """Convert a dictionary to a two-column dataframe for display."""
    if not data_dict:
        return pd.DataFrame(columns=[key_name, value_name])
    return pd.DataFrame({key_name: list(data_dict.keys()), value_name: list(data_dict.values())})


def show_profile(profile, show_full_dtype=False):
    """Print and display the key diagnostics for one table profile."""
    print(f"File: {profile['file_name']} | Layer: {profile['layer']}")
    print(f"Shape: {profile['shape']}")
    print(f"Memory: {profile['memory_mb']} MB")
    print(f"PK columns: {profile['pk_cols']} | PK unique: {profile['pk_unique']}")
    print(f"Date columns: {profile['date_cols']}")
    print("Logic issues:", "; ".join(profile["logic_issues"]))
    print()

    print("Top missing columns:")
    if profile["missing_df"].empty:
        print("  No missing values detected.")
    else:
        display(profile["missing_df"].head(10).reset_index(drop=True))

    print("Zero-value counts (top 10):")
    zero_df = _dict_to_df(profile["zero_counts"], key_name="column", value_name="zero_count")
    if zero_df.empty:
        print("  No zero values detected in numeric columns.")
    else:
        display(zero_df.sort_values("zero_count", ascending=False).head(10).reset_index(drop=True))

    print("Negative-value counts:")
    neg_df = _dict_to_df(profile["negative_counts"], key_name="column", value_name="negative_count")
    if neg_df.empty:
        print("  No negative values detected in numeric columns.")
    else:
        display(neg_df.sort_values("negative_count", ascending=False).reset_index(drop=True))

    print("Date ranges:")
    if not profile["date_ranges"]:
        print("  No date-like columns detected.")
    else:
        date_df = pd.DataFrame.from_dict(profile["date_ranges"], orient="index").reset_index()
        date_df = date_df.rename(columns={"index": "date_column"})
        display(date_df)

    print("Categorical cardinality (top 15):")
    if profile["cardinality_df"].empty:
        print("  No categorical columns detected.")
    else:
        display(profile["cardinality_df"].head(15).reset_index(drop=True))

    print("Data types:")
    if show_full_dtype:
        display(profile["dtypes_df"])
    else:
        display(profile["dtypes_df"].head(20))


def plot_distribution(df, column, bins=40, title_prefix=""):
    """Plot histogram/KDE and a boxplot for one numeric column."""
    if column not in df.columns:
        print(f"Skip distribution: {column} not in dataframe.")
        return

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        print(f"Skip distribution: {column} has no numeric values.")
        return

    axes = plt.subplots(1, 2, figsize=(14, 4))[1]
    sns.histplot(series, bins=bins, kde=True, ax=axes[0], color="#4E79A7")
    axes[0].set_title(f"{title_prefix} Histogram/KDE - {column}")

    sns.boxplot(x=series, ax=axes[1], color="#F28E2B")
    axes[1].set_title(f"{title_prefix} Boxplot - {column}")

    plt.tight_layout()
    plt.show()


def plot_box_by_category(df, value_col, category_col, top_n=12, title_prefix=""):
    """Plot value distribution by top categories using boxplots."""
    if value_col not in df.columns or category_col not in df.columns:
        print(f"Skip boxplot by category: missing {value_col} or {category_col}.")
        return

    temp = df[[value_col, category_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        print("Skip boxplot by category: no valid rows after cleaning.")
        return

    top_categories = temp[category_col].astype(str).value_counts().head(top_n).index
    temp = temp[temp[category_col].astype(str).isin(top_categories)]

    plt.figure(figsize=(14, 5))
    sns.boxplot(data=temp, x=category_col, y=value_col, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{title_prefix} Boxplot {value_col} by {category_col} (top {top_n})")
    plt.tight_layout()
    plt.show()


def plot_daily_count(df, date_col, title_prefix=""):
    """Plot daily record counts based on one date column."""
    if date_col not in df.columns:
        print(f"Skip daily trend: {date_col} not in dataframe.")
        return

    date_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if date_series.empty:
        print(f"Skip daily trend: {date_col} has no valid datetime values.")
        return

    daily_count = (
        date_series.to_frame(name=date_col)
        .groupby(date_col)
        .size()
        .rename("record_count")
        .reset_index()
    )

    plt.figure(figsize=(14, 4))
    plt.plot(daily_count[date_col], daily_count["record_count"], linewidth=0.8)
    plt.title(f"{title_prefix} Daily record count based on {date_col}")
    plt.xlabel("Date")
    plt.ylabel("Record count")
    plt.tight_layout()
    plt.show()


def plot_seasonality_heatmap(df, date_col, value_col=None, agg="count", title_prefix=""):
    """Plot weekday vs month seasonality as a heatmap."""
    if date_col not in df.columns:
        print(f"Skip seasonality heatmap: {date_col} not in dataframe.")
        return

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        print("Skip seasonality heatmap: no valid datetime rows.")
        return

    temp["month"] = temp[date_col].dt.month
    temp["weekday"] = temp[date_col].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if value_col is None or value_col not in temp.columns or agg == "count":
        pivot = temp.groupby(["weekday", "month"]).size().reset_index(name="metric")
    else:
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
        temp = temp.dropna(subset=[value_col])
        if temp.empty:
            print(f"Skip seasonality heatmap: {value_col} has no numeric values.")
            return
        pivot = temp.groupby(["weekday", "month"])[value_col].agg(agg).reset_index(name="metric")

    heatmap_df = pivot.pivot(index="weekday", columns="month", values="metric").reindex(weekday_order)

    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_df, cmap="YlGnBu", linewidths=0.3)
    metric_name = "count" if value_col is None or agg == "count" else f"{agg}({value_col})"
    plt.title(f"{title_prefix} Seasonality heatmap weekday vs month - {metric_name}")
    plt.xlabel("Month")
    plt.ylabel("Weekday")
    plt.tight_layout()
    plt.show()


def plot_missing_bar(df, top_n=15, title_prefix=""):
    """Plot top columns with missing values."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
    if missing.empty:
        print("Skip missing bar chart: no missing values.")
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing.values, y=missing.index, orient="h", color="#E15759")
    plt.title(f"{title_prefix} Null count by column (top {top_n})")
    plt.xlabel("Missing count")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.show()


def plot_missing_matrix(df, sample_rows=MISSING_MATRIX_SAMPLE_SIZE, title_prefix=""):
    """Visualize missingness matrix on a sampled subset of rows."""
    if not df.isna().any().any():
        print("Skip missing matrix: no missing values.")
        return

    sample_df = df.sample(min(sample_rows, len(df)), random_state=42)

    if HAS_MISSINGNO:
        msno.matrix(sample_df, figsize=(12, 4), sparkline=False)
        plt.title(f"{title_prefix} Missing matrix (sample={len(sample_df)})")
        plt.show()
    else:
        plt.figure(figsize=(12, 4))
        sns.heatmap(sample_df.isna().astype(int).T, cbar=False, cmap="viridis")
        plt.title(f"{title_prefix} Missing heatmap fallback (sample={len(sample_df)})")
        plt.xlabel("Sample row index")
        plt.ylabel("Column")
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df, cols=None, max_cols=12, title_prefix=""):
    """Plot a correlation heatmap for numeric features."""
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        num_df = df[cols].copy() if cols else pd.DataFrame()
    else:
        num_df = df.select_dtypes(include=[np.number]).copy()

    if num_df.empty:
        print("Skip correlation heatmap: no numeric columns available.")
        return

    if num_df.shape[1] > max_cols:
        variance_rank = num_df.var(numeric_only=True).sort_values(ascending=False)
        chosen_cols = variance_rank.head(max_cols).index.tolist()
        num_df = num_df[chosen_cols]

    corr = num_df.corr(numeric_only=True)
    if corr.empty:
        print("Skip correlation heatmap: correlation matrix is empty.")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"{title_prefix} Correlation heatmap")
    plt.tight_layout()
    plt.show()


def plot_id_overlap(left_df, left_col, right_df, right_col, left_name, right_name):
    """Plot overlap counts between two ID sets (left only/intersection/right only)."""
    if left_col not in left_df.columns or right_col not in right_df.columns:
        print(f"Skip overlap: missing {left_col} or {right_col}.")
        return None

    left_set = set(left_df[left_col].dropna().unique())
    right_set = set(right_df[right_col].dropna().unique())

    intersection = len(left_set & right_set)
    left_only = len(left_set - right_set)
    right_only = len(right_set - left_set)

    overlap_df = pd.DataFrame(
        {
            "segment": [f"{left_name} only", "intersection", f"{right_name} only"],
            "count": [left_only, intersection, right_only],
        }
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(data=overlap_df, x="segment", y="count", color="#76B7B2")
    plt.title(f"ID overlap concept: {left_name}.{left_col} vs {right_name}.{right_col}")
    plt.tight_layout()
    plt.show()

    return overlap_df


def run_table_audit(
    file_name,
    data_path=DEFAULT_DATA_PATH,
    table_config=None,
    show_plots=True,
    show_dtype=False,
    plot_sample_size=PLOT_SAMPLE_SIZE,
):
    """Run full audit (stats + optional plots) for a single table."""
    if table_config is None:
        table_config = TABLE_CONFIG

    cfg = table_config.get(file_name, {})
    df, profile = profile_table(file_name, data_path=data_path, table_config=table_config)

    print("====================================================")
    print(f"Running audit for {file_name}")
    print("====================================================")
    show_profile(profile, show_full_dtype=show_dtype)

    if show_plots:
        if len(df) > plot_sample_size:
            plot_df = df.sample(plot_sample_size, random_state=42)
            print(f"Plotting with sampled rows: {len(plot_df)}/{len(df)}")
        else:
            plot_df = df

        plot_missing_bar(df, title_prefix=file_name)
        plot_missing_matrix(df, title_prefix=file_name)

        for c in cfg.get("distribution_cols", [])[:2]:
            plot_distribution(plot_df, c, title_prefix=file_name)

        box_cfg = cfg.get("box")
        if box_cfg:
            plot_box_by_category(
                plot_df,
                value_col=box_cfg.get("value"),
                category_col=box_cfg.get("by"),
                title_prefix=file_name,
            )

        if profile["date_cols"]:
            date_col = cfg.get("time_col", profile["date_cols"][0])
            if date_col in df.columns:
                plot_daily_count(df, date_col, title_prefix=file_name)
                seasonality_value_col = cfg.get("seasonality_value_col")
                seasonality_agg = cfg.get("seasonality_agg", "count")
                plot_seasonality_heatmap(
                    df,
                    date_col=date_col,
                    value_col=seasonality_value_col,
                    agg=seasonality_agg,
                    title_prefix=file_name,
                )

        corr_cols = cfg.get("corr_cols")
        plot_correlation_heatmap(plot_df, cols=corr_cols, title_prefix=file_name)

    return df, profile


def run_relation_checks(data_path=DEFAULT_DATA_PATH):
    """Run cross-table relation checks and return a summary dataframe."""
    orders_df = load_table("orders.csv", data_path=data_path)
    payments_df = load_table("payments.csv", data_path=data_path)
    shipments_df = load_table("shipments.csv", data_path=data_path)
    order_items_df = load_table("order_items.csv", data_path=data_path)
    products_df = load_table("products.csv", data_path=data_path)
    returns_df = load_table("returns.csv", data_path=data_path)
    reviews_df = load_table("reviews.csv", data_path=data_path)

    relation_rows = []

    def relation_stat(left_df, left_key, right_df, right_key, relation_name):
        left_set = set(left_df[left_key].dropna().unique())
        right_set = set(right_df[right_key].dropna().unique())
        relation_rows.append(
            {
                "relation": relation_name,
                "left_unique": len(left_set),
                "right_unique": len(right_set),
                "intersection": len(left_set & right_set),
                "left_only": len(left_set - right_set),
                "right_only": len(right_set - left_set),
            }
        )

    relation_stat(orders_df, "order_id", payments_df, "order_id", "orders -> payments")
    relation_stat(orders_df, "order_id", shipments_df, "order_id", "orders -> shipments")
    relation_stat(order_items_df, "product_id", products_df, "product_id", "order_items -> products")
    relation_stat(returns_df, "order_id", orders_df, "order_id", "returns -> orders")
    relation_stat(reviews_df, "order_id", orders_df, "order_id", "reviews -> orders")

    relation_df = pd.DataFrame(relation_rows)
    display(relation_df)

    plot_id_overlap(orders_df, "order_id", payments_df, "order_id", "orders", "payments")
    plot_id_overlap(orders_df, "order_id", shipments_df, "order_id", "orders", "shipments")
    plot_id_overlap(order_items_df, "product_id", products_df, "product_id", "order_items", "products")

    return relation_df


__all__ = [
    "calculate_metrics",
    "print_metrics",
    "DEFAULT_DATA_PATH",
    "GLOBAL_DATE_MIN",
    "GLOBAL_DATE_MAX",
    "PLOT_SAMPLE_SIZE",
    "MISSING_MATRIX_SAMPLE_SIZE",
    "TABLE_ORDER",
    "TABLE_CONFIG",
    "HAS_MISSINGNO",
    "load_table",
    "infer_id_columns",
    "infer_date_columns",
    "count_iqr_outliers",
    "profile_table",
    "build_raw_audit_row",
    "show_profile",
    "plot_distribution",
    "plot_box_by_category",
    "plot_daily_count",
    "plot_seasonality_heatmap",
    "plot_missing_bar",
    "plot_missing_matrix",
    "plot_correlation_heatmap",
    "plot_id_overlap",
    "run_table_audit",
    "run_relation_checks",
]
