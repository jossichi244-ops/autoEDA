# test_dataset.py
import pandas as pd
import numpy as np

def create_golden_test_dataset():
    """
    Tạo DataFrame chứa đầy đủ các trường hợp để test toàn bộ tính năng trong main.py
    """
    np.random.seed(42)  

    data = {
        # 1. Cột số nguyên (Integer)
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

        # 2. Cột số thực (Float) có outliers và skew
        "salary": [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 200000],  # outlier ở cuối

        # 3. Cột phân loại cân bằng (Balanced Categorical)
        "department": ["IT", "HR", "Finance", "IT", "Marketing", "HR", "Finance", "IT", "Marketing", "Finance"],

        # 4. Cột phân loại lệch mạnh (Imbalanced Categorical)
        "is_premium": ["Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No"],  # 90% "No"

        # 5. Cột datetime
        "join_date": pd.date_range(start="2023-01-01", periods=10, freq="D").tolist(),

        # 6. Cột text thông thường
        "comments": [
            "Great work!",
            "Needs improvement",
            "Excellent performance",
            "Average",
            "Outstanding",
            "Below expectations",
            "Good job",
            "Poor",
            "Fantastic",
            "Mediocre"
        ],

        # 7. Cột multi-select (dùng dấu phẩy)
        "skills": [
            "Python,SQL",
            "Python,JavaScript,React",
            "SQL",
            "JavaScript,React,Node.js",
            "Python,SQL,Tableau",
            "React,Node.js",
            "Python",
            "SQL,Tableau",
            "JavaScript,React",
            "Python,JavaScript,SQL,React,Node.js,Tableau"
        ],

        # 8. Cột có giá trị bị thiếu (Missing Values)
        "bonus": [5000, np.nan, 3000, 4000, np.nan, 2000, np.nan, 6000, 1000, np.nan],

        # 9. Cột hằng số (Constant Column)
        "company": ["ABC Corp"] * 10,

        # 10. Cột định danh (Identifier Candidate - unique)
        "email": [f"user{i}@example.com" for i in range(1, 11)],

        # 11. Cột số với giá trị 0
        "login_count": [10, 0, 5, 8, 0, 12, 3, 0, 7, 9],

        # 12. Cột hỗn hợp kiểu dữ liệu (Mixed Types)
        "mixed_data": [1, "two", 3.0, "four", 5, None, "seven", 8.5, "nine", 10],

        # 13. Cột phân phối lệch phải (Right Skewed)
        "project_count": [1, 1, 1, 2, 2, 3, 3, 4, 5, 10],  # lệch phải

        # 14. Cột phân phối lệch trái (Left Skewed)
        "satisfaction_score": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4],  # lệch trái

        # 15. Cột phân tích chuỗi thời gian (cho hàm analyze_timeseries)
        "purchase_date": pd.date_range(start="2023-01-01", periods=10, freq="D").tolist(),
        "purchase_amount": [100, 150, 200, 120, 180, 220, 130, 190, 210, 250],

        # 16. Cột cho phân tích đa chiều (High Dimensional)
        "feature_a": np.random.randn(10),
        "feature_b": np.random.randn(10),
        "feature_c": np.random.randn(10),
        "feature_d": np.random.randn(10),

        # 17. Cột cho phân tích redundancy (tương quan cao)
        "height_cm": [170, 165, 180, 175, 160, 185, 172, 168, 178, 182],
        "height_m": [1.70, 1.65, 1.80, 1.75, 1.60, 1.85, 1.72, 1.68, 1.78, 1.82],  # highly correlated

        # 18. Cột boolean
        "is_active": [True, False, True, True, False, True, False, True, True, False],

        # 19. Cột có entropy thấp (dễ dự đoán)
        "status": ["Active", "Active", "Active", "Inactive", "Active", "Active", "Inactive", "Active", "Active", "Active"],

        # 20. Cột có entropy cao (khó dự đoán)
        "random_category": [f"Cat_{i}" for i in range(10)],  # mỗi giá trị là duy nhất
    }

    df = pd.DataFrame(data)

    # Thêm 1 dòng trùng lặp để test duplicate detection
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    # Thêm 1 cột chỉ chứa NaN để test edge case
    df["all_missing"] = [np.nan] * len(df)

    # Thêm 1 cột chứa inf để test edge case
    df["with_inf"] = [1.0, 2.0, np.inf, 4.0, 5.0, -np.inf, 7.0, 8.0, 9.0, 10.0, 1.0]

    return df

GOLDEN_DF = create_golden_test_dataset()

if __name__ == "__main__":
    print("Golden Test Dataset Shape:", GOLDEN_DF.shape)
    print("\nColumn Types:")
    print(GOLDEN_DF.dtypes)
    print("\nSample Data:")
    print(GOLDEN_DF.head())
    print("\nMissing Values:")
    print(GOLDEN_DF.isnull().sum())