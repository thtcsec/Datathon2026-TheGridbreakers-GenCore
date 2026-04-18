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
Dự án được phân chia để phục vụ các giai đoạn khác nhau của cuộc thi:

-   📂 **/data/**:
    -   `raw/`: Chứa 15 tệp CSV gốc (Master, Transaction, Analytical, Operational).
    -   `processed/`: Chứa dữ liệu sau khi được làm sạch và kết hợp.
-   📂 **/notebooks/**:
    -   `01_MCQ_Solver.ipynb`: Lập trình giải quyết 10 câu hỏi trắc nghiệm Part 1.
    -   `02_EDA_Prescriptive_Analysis.ipynb`: Phân tích dữ liệu và đề xuất giải pháp cho Part 2.
    -   `03_Forecasting_Model.ipynb`: Xây dựng pipeline dự báo cho Part 3.
-   📂 **/src/**:
    -   `preprocessing.py`: Các hàm xử lý và gộp bảng dữ liệu.
    -   `utils.py`: Các hàm tính toán chỉ số MAE, RMSE, R².
-   📂 **/output/**: Lưu trữ tệp dự báo `submission.csv` và các biểu đồ.

---

## 🔍 Key Insights
- [Phát hiện 1]: (Sẽ cập nhật sau quá trình EDA)
- [Phát hiện 2]: (Sẽ cập nhật sau quá trình EDA)
- [Phát hiện 3]: (Sẽ cập nhật sau quá trình EDA)

---

## 🚀 Model Performance
Kết quả dự báo trên tập validation (2021-2022):

| Chỉ số | Kết quả |
| :--- | :--- |
| **MAE** | ... |
| **RMSE** | ... |
| **R²** | ... |

---

## 🛠️ Ràng buộc & Đặc điểm Kỹ thuật
Dự án được xây dựng dựa trên các quy định sau:
-   **Đơn vị tiền tệ**: Sử dụng VND (Tỷ giá tham chiếu: 1 USD ≈ 25,450 VND).
-   **Dữ liệu**: Không sử dụng nguồn dữ liệu bên ngoài.
-   **Rò rỉ dữ liệu (Leakage)**: Không sử dụng giá trị Revenue hay COGS của tập test để làm đặc trưng huấn luyện.
-   **Phương pháp dự báo**: Kết hợp Seasonal profile (theo ngày trong năm) và xu hướng tăng trưởng YoY.
-   **Môi trường**: Python 3.13+, sử dụng Pandas, NumPy, Scikit-learn, XGBoost/LightGBM.

---
*Mã nguồn dành cho cuộc thi VinUniversity Datathon 2026.*
