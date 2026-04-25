from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.eda.data import safe_divide

DEFAULT_REVIEW_SHARE = 0.15


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


def choose_operating_threshold(valid_scores: np.ndarray, review_share: float = DEFAULT_REVIEW_SHARE) -> float:
    return float(np.quantile(valid_scores, 1 - review_share))


def lift_at_fraction(y_true: pd.Series, scores: np.ndarray, fraction: float) -> float:
    frame = pd.DataFrame({"target": y_true.to_numpy(), "score": scores})
    top_n = max(int(math.ceil(len(frame) * fraction)), 1)
    top_rate = frame.sort_values("score", ascending=False).head(top_n)["target"].mean()
    base_rate = frame["target"].mean()
    return safe_divide(top_rate, base_rate)


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

    score_map: dict[str, dict[str, Any]] = {
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


__all__ = [
    "build_model_dataset",
    "split_model_data",
    "fit_models",
    "evaluate_predictions",
    "build_review_queue_summary",
    "build_model_interpretation",
    "build_risk_segments",
]
