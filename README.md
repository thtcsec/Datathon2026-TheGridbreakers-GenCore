# DATATHON 2026 — The Gridbreakers | Đội ngũ GenCore

## 🌟 Giới thiệu
Đây là repository của đội **GenCore** tham dự cuộc thi **DATATHON 2026 — The Gridbreakers** tại **VinUniversity**. Dự án tập trung vào việc xử lý và phân tích dữ liệu e-commerce thời trang tại Việt Nam (2012-2022) để giải quyết các bài toán về MCQ, EDA và dự báo doanh thu.

---

## 👥 Thành viên Đội ngũ (Team GenCore)

| Họ và Tên | Vai trò | Trường Đại học |
| :--- | :--- | :--- |
| **Trịnh Hoàng Tú** | **Trưởng nhóm** | Trường Đại học Ngoại ngữ - Tin học TP.HCM (HUFLIT) |
| Nguyễn Tấn Thắng | Thành viên | Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM (HCMUS) |
| Nguyễn Trọng Hưởng | Thành viên | Trường Đại học Kinh tế TP.HCM (UEH) |
| Nguyễn Minh Nhựt | Thành viên | Trường Đại học Kinh tế TP.HCM (UEH) |

**Mã đội thi:** `YdHJLESH3WgWYCgY8nnb`

---

## 📂 Cấu trúc Thư mục

```
├── data/
│   └── raw/              # 15 tệp CSV gốc (Master, Transaction, Analytical, Operational)
├── notebooks/
│   ├── 01_MCQ_Solver.ipynb
│   ├── 02_EDA_Prescriptive_Analysis.ipynb
│   ├── 03_Forecasting_Model.ipynb
│   ├── 09_Kaggle_Submission_v6_Auxiliary.ipynb # Pipeline Prophet (Aux) + LGBM
│   └── baseline.ipynb
├── src/
│   ├── preprocessing.py  # Xử lý & gộp bảng dữ liệu thô
│   ├── features.py       # Feature engineering (time, lag, external)
│   ├── models.py         # Train & evaluate XGBoost / LightGBM
│   ├── tuning.py         # Hyperparameter tuning (Optuna + TimeSeriesSplit)
│   ├── evaluation.py     # So sánh models (baseline vs XGB vs LGBM) → CSV
│   └── utils.py          # Metrics: MAE, RMSE, R², MAPE, sMAPE
├── output/
│   ├── submission.csv              # File dự báo nộp bài
│   ├── model_comparison.csv        # Bảng so sánh các mô hình
│   ├── feature_importance_*.png    # Biểu đồ feature importance
│   └── tuning/                     # Artifacts từ hyperparameter tuning
│       ├── best_model_*.joblib     # Model đã train (serialized)
│       ├── best_params.json        # Best hyperparameters + metrics
│       └── tuning_report.txt       # Báo cáo tuning
└── requirements.txt
```

---

## ⚡ Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

> **macOS**: XGBoost cần OpenMP runtime — chạy `brew install libomp` trước khi cài.

---

## 🚀 Hướng dẫn chạy Pipeline

### 1. Tiền xử lý & Feature Engineering

```python
from src.preprocessing import load_and_merge
from src.features import build_features

# Load & gộp dữ liệu thô
train_df, forecast_df = load_and_merge(data_path='data/raw')

# Tạo features (time, lag, external)
result = build_features(train_df, forecast_df, data_path='data/raw')
train_features  = result['train_features']
forecast_features = result['forecast_features']
feature_cols    = result['feature_cols']
```

### 2. Train & Đánh giá nhanh (Baseline)

```python
from src.models import train_and_evaluate

# Train với default params + TimeSeriesSplit CV
models, metrics = train_and_evaluate(
    train_features, feature_cols,
    model_name='xgboost',   # hoặc 'lightgbm'
    n_splits=5,
)
print(metrics)
```

### 3. So sánh Models → CSV

```python
from src.evaluation import build_comparison_table, save_comparison

# Chạy MeanBaseline, SeasonalNaive, XGBoost, LightGBM trên cùng CV folds
comparison = build_comparison_table(train_features, feature_cols)
save_comparison(comparison, out_path='output/model_comparison.csv')
```

Output mẫu (`model_comparison.csv`):

| Model | Target | MAE | RMSE | R² | MAPE | sMAPE |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | Revenue | ... | ... | ... | ...% | ...% |
| LightGBM | Revenue | ... | ... | ... | ...% | ...% |
| MeanBaseline | Revenue | ... | ... | ... | ...% | ...% |

### 4. Hyperparameter Tuning (Optuna)

```python
from src.tuning import run_full_tuning

# Tuning XGBoost & LightGBM với Optuna (Bayesian TPE) + TimeSeriesSplit
results = run_full_tuning(
    train_features, feature_cols,
    target_cols=('Revenue', 'COGS'),
    n_trials=50,       # số trials Optuna cho mỗi model
    n_splits=5,        # folds cho TimeSeriesSplit
    metric='MAE',      # metric tối ưu
)
# Artifacts tự động lưu tại output/tuning/
```

**Tham số được tuning:**

| Tham số | XGBoost | LightGBM |
| :--- | :--- | :--- |
| `n_estimators` | 300–1500 | 300–1500 |
| `max_depth` | 3–10 | 3–10 |
| `learning_rate` | 0.01–0.3 | 0.01–0.3 |
| `subsample` | 0.5–1.0 | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 | 0.5–1.0 |
| `reg_alpha` | 1e-8–10 | 1e-8–10 |
| `reg_lambda` | 1e-8–10 | 1e-8–10 |
| `min_child_weight` | 1–10 | — |
| `num_leaves` | — | 15–127 |

### 5. Inference & Tạo Submission

```python
from src.models import generate_submission

submission, path = generate_submission(
    forecast_features, models, feature_cols,
    out_path='output/submission.csv',
)
print(f'Submission saved → {path}')
```

---

## 🔍 Key Insights
- **Feature Importance Trap**: Mô hình ban đầu (XGBoost/LightGBM) sử dụng biến đồng thời (web_traffic, daily_orders) → CV MAE 417k nhưng LB MAE 2M. Nguyên nhân: các biến này không tồn tại trong giai đoạn dự báo 548 ngày.
- **Naive364 là backbone mạnh nhất**: Seasonal Naive với lookback 364 ngày (bảo toàn DOW) đạt MAE ~775k trên holdout 548 ngày — vượt trội so với Prophet đơn lẻ (~1.1M).
- **Prophet weight ≈ 0**: Trong ensemble tối ưu, Prophet không đóng góp trực tiếp. Hybrid (Prophet + LGB residual) chỉ chiếm 20-25% weight, phần lớn giá trị đến từ LGB correction.
- **Tết Nguyên Đán là yếu tố quyết định**: Doanh thu tăng mạnh 3-4 tuần trước Tết, giảm sâu trong tuần Tết. Calibration với strength=0.10 cho kết quả tốt nhất.
- **Mega-sales (11.11, 12.12, 30/4-1/5)**: Tạo spike doanh thu lớn, cần xử lý riêng biệt trong mô hình.

---

## 🚀 Model Performance
Kết quả dự báo trên tập holdout 548 ngày (2021-07-02 → 2022-12-31):

| Target | MAE | RMSE | R² | Best Blend |
| :--- | ---: | ---: | ---: | :--- |
| **Revenue** | 746,176 | 1,025,347 | 0.5517 | Naive 75% + Hybrid 25% |
| **COGS** | 659,624 | 907,481 | 0.5480 | Naive 80% + Hybrid 20% |

**Kiến trúc mô hình (Zero-Leakage):**
1. **Seasonal Naive 364** (backbone): Lookback 364 ngày + window median blend + trend adjustment
2. **Prophet Ensemble** (2 configs): Logistic growth, multiplicative seasonality, Tết holidays
3. **LightGBM Residual Corrector**: Sửa sai số của Prophet, features hoàn toàn deterministic
4. **Tết Calibration**: Empirical multipliers từ 13 năm lịch sử, strength=0.10

---

## 🛠️ Ràng buộc & Đặc điểm Kỹ thuật
Dự án được xây dựng dựa trên các quy định sau:
-   **Đơn vị tiền tệ**: Sử dụng VND (Tỷ giá tham chiếu: 1 USD ≈ 25,450 VND).
-   **Dữ liệu**: Không sử dụng nguồn dữ liệu bên ngoài.
-   **Rò rỉ dữ liệu (Leakage)**: Zero-leakage architecture — tất cả features đều deterministic tại thời điểm dự báo.
-   **Validation**: 548-day holdout (matching Kaggle test horizon) — không dùng random split.
-   **Mô hình**: Seasonal Naive + Prophet + LightGBM Residual Ensemble, tối ưu blend weights + Tết strength.
-   **Đánh giá**: MAE (primary), RMSE, R² — so sánh với baseline (Naive364, Prophet, Hybrid).
-   **Môi trường**: Python 3.13+, Pandas, NumPy, Scikit-learn, Prophet, LightGBM, Optuna.

---
*Mã nguồn dành cho cuộc thi VinUniversity Datathon 2026.*

