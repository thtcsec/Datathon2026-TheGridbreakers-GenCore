# DATATHON 2026 — The Gridbreakers · Team GenCore

Repository của đội **GenCore** dự thi **DATATHON 2026 — The Gridbreakers** (VinTelligence — VinUniversity Data Science & AI Club).

Đề Phần 3 (Kaggle): dự báo `Revenue` và `COGS` **theo ngày** trong khoảng **2023-01-01 → 2024-07-01** (**548** dòng), từ dữ liệu lịch sử đến **2022-12-31**.

## Thành viên đội (Team GenCore)

| Họ và Tên | Vai trò | Trường Đại học |
| :--- | :--- | :--- |
| **Trịnh Hoàng Tú** | **Trưởng nhóm** | Trường Đại học Ngoại ngữ - Tin học TP.HCM (HUFLIT) |
| Nguyễn Tấn Thắng | Thành viên | Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM (HCMUS) |
| Nguyễn Trọng Hưởng | Thành viên | Trường Đại học Kinh tế TP.HCM (UEH) |
| Nguyễn Minh Nhựt | Thành viên | Trường Đại học Kinh tế TP.HCM (UEH) |

**Mã đội thi:** `YdHJLESH3WgWYCgY8nnb`

## Tuân thủ BTC — Phần 3 (Kaggle)

- **Đầu vào:** chỉ dữ liệu **BTC cung cấp** trong bundle thi (`sales.csv`, `returns.csv`, `promotions.csv`, `web_traffic.csv`, `inventory.csv`, `sample_submission.csv`, và các CSV gốc còn lại trong đề). **Không** dùng nguồn dữ liệu **bên ngoài**.
- **Submission:** đúng **548** hàng; cột và thứ tự **`Date`,`Revenue`,`COGS`**; **không** đổi **thứ tự** hàng so với `sample_submission.csv`; **không** âm / NaN / Inf trên các giá trị dự báo.
- Pipeline trong notebook **không** đọc file submission cũ từ thí nghiệm để làm “anchor” leaderboard; anchor neural **b39** được **train lại trong repo** (`src/neural_blend_refined_b39.py`), chỉ đọc raw BTC + template ngày từ `sample_submission.csv`.

## File nộp leaderboard

| File | Mô tả |
| :--- | :--- |
| `output/submission.csv` | File nộp chính thức (sinh sau `Restart & Run All` của notebook cuối). |
| `output/submission_final_best_673250.csv` | Bản sao của submission trùng với artefact đã đạt public MAE ~**673 250** (`submission_v41_edge_only_w35.csv`). |

Format trùng `data/raw/sample_submission.csv`.

## Notebook sinh submission cuối

`notebooks/14_Final_LB_Optimization_Journey.ipynb` là notebook **BTC-compliant** chính thức (**Restart & Run All**).  
`notebooks/13_LB_Optimization_Log.ipynb` chỉ là **timeline / điểm LB**, không phải pipeline tái tạo submission.

**Đường leaderboard đã chốt:** file tốt nhất đã verify là `submission_v41_edge_only_w35.csv` (public MAE **673 250.34766**). Notebook 14 xuất `submission = v41.copy()` → `output/submission.csv` khớp artefact đó; các biến thể **v43** trong notebook chỉ mang tính thử — điểm kém hơn (~675 108), **không** dùng làm file nộp chính.

**Kaggle sau deadline:** nếu `/kaggle/input` trống hoặc không còn bundle BTC, mọi notebook đều báo `sales.csv not found` — cần **Add Data** hoặc tự upload dataset chứa các CSV đề bài (logic tìm file trong notebook 12 / 14 / 06: `os.walk` trước, `glob` sau).

**Pipeline notebook 14 (Restart & Run All):**

1. **Train neural anchor b39** từ raw (`src/neural_blend_refined_b39.py`; logic port từ nhánh teammate `Thang-B`: `src/r.py` / `Kaggle_Sub_Refined_Monthly_COGS.ipynb`), ghi kèm `output/submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv`.
2. **GBDT tabular** (`src/ml_tabular_blend.py`): XGBoost + LightGBM trên lag / calendar / `web_traffic` / `inventory` — **TimeSeriesSplit** (MAE từng fold trong log) + blend vào anchor; **walk-forward** optional trên cuối 2022 (§1c).  
   *Khớp byte với artefact cũ **`submission_v41_edge_only_w35.csv`** (chỉ neural b39 → v41): trong notebook 14 §1b đặt **`ML_TABULAR_WEIGHT = 0`**.*
3. **v20 … v41 (frozen calibration):** các hệ số (`alpha` **4.30**, scale, edge, …) là **legacy** từ thí nghiệm leaderboard — không retune trong notebook; **không** nhầm với metric hold-out ở bước 2.
4. Validate schema BTC, ghi **`output/submission.csv`** và **`submission_final_best_673250.csv`**.

**Kaggle:** trong notebook bấm **Add Data** và gắn bundle thi (giống notebook 12 — phải thấy `/kaggle/input/.../sales.csv`). Module pipeline được **giải nén từ zlib+base64 nhúng trong notebook** ra `/kaggle/working` — đó là **source code**, không phải submission hay csv BTC lạ.

## Cấu trúc thư mục

```
├── data/raw/                    # CSV gốc BTC (không commit trong git — xem .gitignore)
├── notebooks/
│   ├── 01_MCQ_Solver.ipynb      # Phần 1
│   ├── 02_EDA_...               # Phần 2
│   ├── 03_Forecasting_Model.ipynb
│   ├── 13_LB_Optimization_Log.ipynb
│   └── 14_Final_LB_Optimization_Journey.ipynb   # Submission Phần 3 (chính)
├── output/
│   ├── submission.csv
│   ├── submission_final_best_673250.csv
│   └── … (một số artefact verify / PNG phục vụ báo cáo; sweep thử LB cục bộ bị .gitignore)
├── src/
│   ├── neural_blend_refined_b39.py
│   ├── ml_tabular_blend.py   # XGB+LGB, TimeSeries CV + walk-forward + blend vào anchor
│   └── v20…v43 pipeline scripts phụ các bước LB-guided
├── report/                      # Báo cáo NeurIPS-style
├── requirements.txt
├── GenCore.pdf
└── README.md
```

## Cài đặt và chạy lại

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
jupyter notebook notebooks/14_Final_LB_Optimization_Journey.ipynb
```

Sau đó **Restart & Run All**. Bước neural thường mất **vài phút** CPU.

## Pipeline tóm tắt (public LB đã submit)

| Bước | Mô tả | Artefact đối chứng (tracked) | Public MAE ~ |
| :--- | :--- | :--- | ---: |
| 0 | Neural **b39** (train trong notebook / `src/neural_blend_refined_b39.py`) | `submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv` | 725 504 |
| 1 | Shape + alpha 4.3 | `submission_v23_b39_all_430.csv` | 704 169 |
| 2 | Scale +3% | `submission_v30_v23_both_up_300pct.csv` | 697 984 |
| 3 | Monthly rebalance + scale 1.025 | `submission_v37_rebal_s10250.csv` | 675 314 |
| 4 | Edge **w** = 0.35 | **`submission_v41_edge_only_w35.csv`** | **673 250** |

*Nếu neural retrain lệch byte-for-byte so với artifact cũ, các file verify có thể lệch nhẹ — công thức v20–v41 giữ cố định.*

## Phần 1 và Phần 2

- Phần 1: `notebooks/01_MCQ_Solver.ipynb`
- Phần 2: `notebooks/02_EDA_Prescriptive_Analysis.ipynb`
- Báo cáo: `report/main.tex` → `GenCore.pdf`

---

*Repository phục vụ VinUniversity DATATHON 2026.*
