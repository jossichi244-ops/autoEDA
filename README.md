# 📊 AutoEDA Service

---

## 🚀 Giới thiệu

**AutoEDA Service** là một dịch vụ API xây dựng trên **FastAPI** giúp:

- Tự động phân tích dữ liệu (EDA – Exploratory Data Analysis).
- Đưa ra **thống kê mô tả** , **biểu đồ trực quan** , **mối quan hệ giữa biến** , **phát hiện bất thường** , và **gợi ý xử lý dữ liệu** .
- Hỗ trợ nhiều định dạng đầu vào: `.csv`, `.xlsx`, `.json`.

Mục tiêu: giúp **Data Scientist / Business Analyst** nhanh chóng **nắm bắt chất lượng dữ liệu** , **giảm rủi ro sai sót** và **tăng tốc quyết định** .

---

## 🛠️ Công nghệ sử dụng

- **FastAPI** – Xây dựng REST API nhanh, chuẩn OpenAPI.
- **Pandas / NumPy** – Xử lý dữ liệu dạng bảng.
- **Scikit-learn** – Clustering, outlier detection, feature importance.
- **Statsmodels / SciPy** – Phân tích thống kê (ANOVA, Chi-square, VIF…).
- **Plotly** – Hỗ trợ sinh dữ liệu trực quan hóa (histogram, boxplot, pie chart…).
- **dotenv** – Cấu hình environment.

---

## ⚙️ Cài đặt & Chạy thử

### 1. Clone repo & setup môi trường

<pre class="overflow-visible!" data-start="1490" data-end="1701"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>git </span><span>clone</span><span> https://github.com/your-repo/autoeda-service.git
</span><span>cd</span><span> autoeda-service
python -m venv venv
</span><span>source</span><span> venv/bin/activate  </span><span># hoặc .\venv\Scripts\activate trên Windows</span><span>
pip install -r requirements.txt
</span></span></code></div></div></pre>

### 2. Tạo file `.env`

<pre class="overflow-visible!" data-start="1726" data-end="1782"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-env"><span>APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=*
</span></code></div></div></pre>

### 3. Chạy server

<pre class="overflow-visible!" data-start="1803" data-end="1879"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
</span></span></code></div></div></pre>

API docs sẽ có tại: 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 API chính

### `POST /api/parse-file`

Upload file dữ liệu và nhận lại:

- **Schema** : loại biến (numeric, categorical, datetime, text…).
- **Preview** : vài dòng đầu sau làm sạch.
- **Inspection** : missing values, duplicates, gợi ý xử lý.
- **Descriptive** : thống kê số học & tần suất.
- **Visualizations** : dữ liệu cho biểu đồ (histogram, boxplot, pie, sunburst…).
- **Relationships** : tương quan numeric-numeric, cat-cat, cat-num.
- **Advanced** : clustering, anomaly detection, significance tests, time series decomposition.

**Ví dụ cURL:**

<pre class="overflow-visible!" data-start="2515" data-end="2606"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>curl -X POST </span><span>"http://localhost:8000/api/parse-file"</span><span> \
  -F </span><span>"file=@yourdata.csv"</span><span>
</span></span></code></div></div></pre>

---

## 🔍 Một số tính năng nổi bật

- **Làm sạch dữ liệu** : chuẩn hóa tên cột, xử lý missing values, expand multi-select.
- **Thống kê mô tả** : mean, median, skewness, kurtosis, outlier counts.
- **Mối quan hệ biến** :
- Correlation heatmap.
- ANOVA / Chi-square.
- Boxplot & barplot categorical-numeric.
- **Clustering & Anomaly Detection** :
- KMeans + Silhouette Score.
- Isolation Forest cho outlier flags.
- **Phân tích chuỗi thời gian** :
- Trend, seasonality, residuals.
- Outlier detection theo rolling z-score.
- **Gợi ý xử lý** : cột hằng số → drop, nhiều missing → impute/drop, high-cardinality → embedding…

---

## 🧭 Hướng dẫn đọc kết quả

- **Inspection → suggested_actions** : nơi đầu tiên bạn nên xem để hiểu "nguy cơ" của dataset.
- **Descriptive → remarks** : tôi để lại "nhận xét tự động" (giống kinh nghiệm kiểm toán dữ liệu).
- **Advanced → feature_importance** : hữu ích khi bạn có biến mục tiêu (`target`) để định hướng mô hình.
- **Visualizations** : chỉ trả về **data structure** (bins, counts, categories) → bạn render frontend tùy ý.

---

## 📌 Ghi chú từ kinh nghiệm của tôi

- **Không có dữ liệu nào hoàn hảo.** Quan trọng là biết được dữ liệu _thiếu cái gì_ và _có thể tin tưởng điều gì_ .
- **EDA không thay thế domain knowledge.** Công cụ này chỉ gợi ý, quyết định cuối cùng phải dựa trên bối cảnh kinh doanh.
- **Đừng bỏ qua remarks.** Chúng là những “red flags” giúp bạn tránh bẫy phân tích.

---

## 📜 Giấy phép

MIT License – dùng tự do, nhưng vui lòng ghi nguồn nếu bạn phát triển thêm.
