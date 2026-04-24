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
- [Phát hiện 1]: (Sẽ cập nhật sau quá trình EDA)
- [Phát hiện 2]: (Sẽ cập nhật sau quá trình EDA)
- [Phát hiện 3]: (Sẽ cập nhật sau quá trình EDA)

---

## 🚀 Model Performance
Kết quả dự báo trên tập validation (TimeSeriesSplit CV):

| Chỉ số | Kết quả |
| :--- | :--- |
| **MAE** | ... |
| **RMSE** | ... |
| **R²** | ... |
| **MAPE** | ...% |
| **sMAPE** | ...% |

---

## 🛠️ Ràng buộc & Đặc điểm Kỹ thuật
Dự án được xây dựng dựa trên các quy định sau:
-   **Đơn vị tiền tệ**: Sử dụng VND (Tỷ giá tham chiếu: 1 USD ≈ 25,450 VND).
-   **Dữ liệu**: Không sử dụng nguồn dữ liệu bên ngoài.
-   **Rò rỉ dữ liệu (Leakage)**: Không sử dụng giá trị Revenue hay COGS của tập test để làm đặc trưng huấn luyện.
-   **Validation**: TimeSeriesSplit (walk-forward CV) — không dùng random split.
-   **Mô hình**: XGBoost & LightGBM với hyperparameter tuning bằng Optuna (Bayesian TPE sampler).
-   **Đánh giá**: MAE, RMSE, R², MAPE, sMAPE — so sánh với baseline (Mean, Seasonal Naive).
-   **Môi trường**: Python 3.13+, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Optuna.

---
*Mã nguồn dành cho cuộc thi VinUniversity Datathon 2026.*

