# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from io import BytesIO
import numpy as np
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import math
import re
from scipy.stats import skew, kurtosis
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
import logging
from typing import Optional
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]

app = FastAPI(title="AutoEDA Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd
import numpy as np

def infer_schema_from_df(df: pd.DataFrame, max_unique_for_enum: int = 20) -> dict:
    props = {}

    for col in df.columns:
        series = df[col].dropna()
        dtype = "string"

        # --- Kiểm tra dtype gốc ---
        if pd.api.types.is_integer_dtype(series):
            dtype = "integer"
        elif pd.api.types.is_float_dtype(series):
            dtype = "number"
        elif pd.api.types.is_bool_dtype(series):
            dtype = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        else:
            # --- fallback: thử parse string ---
            sample_str = series.astype(str).head(100)

            # integer regex
            if sample_str.str.match(r"^-?\d+$").all():
                dtype = "integer"
            # float regex
            elif sample_str.str.match(r"^-?\d+(\.\d+)?$").all():
                dtype = "number"
            else:
                parsed = pd.to_datetime(sample_str, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    dtype = "datetime"
                else:
                    dtype = "string"

        unique_count = series.nunique()

        # --- Build prop ---
        if dtype == "datetime":
            prop = {"type": "string", "format": "date-time"}
        elif dtype == "string" and unique_count <= max_unique_for_enum:
            prop = {"type": "string", "enum": series.unique().tolist()}
        else:
            prop = {"type": dtype}

        props[col] = prop

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GeneratedSchema",
        "type": "object",
        "properties": props
    }

    return schema

def shannon_entropy(values):
    freq = values.value_counts(normalize=True)
    return -(freq * np.log2(freq + 1e-9)).sum()

def analyze_column(col: pd.Series) -> dict:
    series = col.dropna().astype(str)
    dtype = "text"
    unique_count = series.nunique()
    total = len(series)

    # numeric
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        dtype = "numeric"

    # datetime (cột vốn đã là datetime hoặc parse được từ chuỗi)
    elif pd.api.types.is_datetime64_any_dtype(col):
        dtype = "datetime"
    else:
        # thử parse datetime từ chuỗi
        dt_parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if dt_parsed.notnull().mean() > 0.8:  # parse được >=80%
            dtype = "datetime"
        elif unique_count <= 50 or unique_count < 0.05 * total:
            dtype = "categorical"
        elif (
            series.str.contains(r"[;,|]").any() or
            (unique_count > 50 and shannon_entropy(series) < math.log2(unique_count) * 0.3)
        ):
            dtype = "multi-select"
        else:
            dtype = "text"

    # Stats
    stats = {}
    if dtype == "numeric":
        numeric_col = pd.to_numeric(col, errors="coerce")
        stats = {
            "min": convert_numpy_types(numeric_col.min()),
            "max": convert_numpy_types(numeric_col.max()),
            "mean": convert_numpy_types(numeric_col.mean()),
            "std": convert_numpy_types(numeric_col.std()),
        }
    elif dtype in ("categorical", "multi-select"):
        stats = {
            "unique_values": convert_numpy_types(unique_count),
            "top_values": convert_numpy_types(series.value_counts().head(5).to_dict())
        }
    elif dtype == "datetime":
        dt_col = pd.to_datetime(col, errors="coerce")
        stats = {
            "min_date": str(dt_col.min()) if not pd.isna(dt_col.min()) else None,
            "max_date": str(dt_col.max()) if not pd.isna(dt_col.max()) else None
        }
    else:  # text
        str_lengths = series.map(len)
        stats = {
            "avg_length": convert_numpy_types(str_lengths.mean()),
            "sample_values": convert_numpy_types(series.head(5).tolist())
        }

    return {
        "name": col.name,
        "inferred_type": dtype,
        "non_null": convert_numpy_types(col.notnull().sum()),
        "nulls": convert_numpy_types(col.isnull().sum()),
        "example": convert_numpy_types(series.head(3).tolist()),
        "stats": stats
    }

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) and np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        # ✅ XỬ LÝ KHÓA: Chuyển pd.Timestamp thành str nếu là key
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, pd.Timestamp):
                key = str(key)
            elif isinstance(key, (np.integer, np.floating)):
                key = convert_numpy_types(key)  # cũng xử lý numpy key
            new_dict[key] = convert_numpy_types(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

def inspect_dataset(df: pd.DataFrame, max_sample: int = 10, target: Optional[str] = None) -> dict:
    """Comprehensive Data Inspection (Bước 2) — returns detailed diagnostics with target-aware correlation."""

    total_rows, total_cols = df.shape
    head = df.head(max_sample).to_dict(orient="records")
    tail = df.tail(max_sample).to_dict(orient="records")

    # dtypes
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # memory usage
    try:
        mem_series = df.memory_usage(deep=True)
        memory_per_column = {col: int(mem_series[col]) for col in mem_series.index}
        memory_total = int(mem_series.sum())
    except Exception:
        memory_per_column = {}
        memory_total = None

    # missing values
    null_counts = df.isnull().sum()
    null_percent = (null_counts / max(1, total_rows)).round(4)
    missing_summary = {
        "total_cells": int(total_rows * total_cols),
        "total_missing": int(null_counts.sum()),
        "percent_missing": float((null_counts.sum() / max(1, total_rows * total_cols)).round(4)),
        "columns_missing": null_counts.to_dict(),
        "columns_missing_percent": null_percent.to_dict(),
        "top_missing_columns": null_percent.sort_values(ascending=False).head(10).to_dict(),
        "columns_many_missing": (null_percent[null_percent > 0.5]).sort_values(ascending=False).to_dict()
    }

    # duplicates
    try:
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_rows_sample = df[df.duplicated(keep=False)].head(5).to_dict(orient="records")
    except Exception:
        duplicate_rows_count = 0
        duplicate_rows_sample = []

    duplicates_summary = {
        "duplicate_count": duplicate_rows_count,
        "duplicate_sample": duplicate_rows_sample,
    }

    # optional target
    target_series = None
    if target and target in df.columns:
        target_series = df[target]

    columns = {}
    for col in df.columns:
        col_series = df[col]
        non_null = int(col_series.notnull().sum())
        nulls = int(col_series.isnull().sum())
        unique_count = int(col_series.nunique(dropna=True))
        unique_pct = round(unique_count / max(1, total_rows), 4)
        top_values = col_series.value_counts(dropna=True).head(5).to_dict()

        # mixed type detection
        sample_vals = col_series.dropna().head(1000).tolist()
        types_seen = set(type(v).__name__ for v in sample_vals if v is not None)
        mixed_types = len(types_seen) > 1

        # parse datetime
        try:
            parsed_dt = pd.to_datetime(col_series, errors="coerce", infer_datetime_format=True, dayfirst=True)
        except Exception:
            parsed_dt = pd.to_datetime(col_series, errors="coerce")
        dt_parsed_pct = float(parsed_dt.notnull().sum() / max(1, total_rows))

        # parse numeric
        numeric_parsed = pd.to_numeric(col_series, errors="coerce")
        num_parsed_pct = float(numeric_parsed.notnull().sum() / max(1, total_rows))

        # detect multi-select
        sep_candidates = [",", ";", "|", " / "]
        contains_sep = col_series.astype(str).str.contains("|".join([s.replace(" ", r"\s*") for s in sep_candidates]),
                                                          regex=True, na=False)
        has_separator = bool(contains_sep.sum() / max(1, total_rows) > 0.2)

        # numeric stats & outliers
        numeric_stats, outlier_count, skew, kurt, zero_count = None, None, None, None, None
        if num_parsed_pct > 0.5:
            num_series = pd.to_numeric(col_series, errors="coerce").dropna()
            if len(num_series) > 0:
                q1, q3 = float(num_series.quantile(0.25)), float(num_series.quantile(0.75))
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (num_series < lower) | (num_series > upper)
                outlier_count = int(outlier_mask.sum())
                numeric_stats = {
                    "min": float(num_series.min()),
                    "q1": q1,
                    "median": float(num_series.median()),
                    "q3": q3,
                    "max": float(num_series.max()),
                    "mean": float(num_series.mean()),
                    "std": float(num_series.std()),
                    "iqr": iqr,
                    "outlier_count": outlier_count,
                }
                try:
                    skew, kurt = float(num_series.skew()), float(num_series.kurt())
                except Exception:
                    pass
                zero_count = int((num_series == 0).sum())

        # datetime stats
        dt_stats = None
        if dt_parsed_pct > 0.5:
            dt_series = pd.to_datetime(col_series, errors="coerce").dropna()
            if len(dt_series) > 0:
                dt_stats = {
                    "min_date": str(dt_series.min()),
                    "max_date": str(dt_series.max()),
                    "count_parsed": int(dt_series.shape[0]),
                }

        # entropy
        entropy = None
        if unique_count > 0 and unique_count < 5000 and top_values:
            counts = list(col_series.value_counts(dropna=True).values)
            probs = [c / sum(counts) for c in counts]
            entropy = float(round(-sum(p * math.log2(p) for p in probs if p > 0), 4))

        # correlation with target
        correlation_with_target = None
        if target_series is not None and col != target:
            try:
                if pd.api.types.is_numeric_dtype(target_series) and num_parsed_pct > 0.5:
                    correlation_with_target = float(pd.to_numeric(col_series, errors="coerce").corr(target_series))
                elif pd.api.types.is_numeric_dtype(target_series) and dt_parsed_pct > 0.5:
                    correlation_with_target = float(parsed_dt.astype(np.int64).corr(target_series))
            except Exception:
                correlation_with_target = None

        # cardinality classification
        if unique_count == total_rows and total_rows > 1:
            cardinality_type = "id"
        elif unique_pct > 0.7:
            cardinality_type = "high"
        elif unique_pct < 0.05:
            cardinality_type = "low"
        else:
            cardinality_type = "medium"

        # suggestions
        suggestions = []
        if nulls / max(1, total_rows) > 0.5:
            suggestions.append("High missing rate (>50%): consider drop column or strong imputation.")
        if cardinality_type == "id":
            suggestions.append("Looks like a unique identifier (candidate primary key).")
        if unique_count <= 1:
            suggestions.append("Constant column: likely safe to drop.")
        if num_parsed_pct > 0.5 and outlier_count and outlier_count / max(1, total_rows) > 0.01:
            suggestions.append("Numeric column with outliers: consider winsorize or inspect outlier rows.")
        if has_separator:
            suggestions.append("Multi-select / delimiter detected: consider explode into multiple boolean columns.")
        if mixed_types:
            suggestions.append("Mixed types detected: inspect parsing or coerce consistently.")
        if unique_pct > 0.95 and unique_count > 50 and num_parsed_pct < 0.2:
            suggestions.append("High-cardinality categorical: consider hashing/embedding or keep as text.")
        if skew and abs(skew) > 2:
            suggestions.append("Highly skewed numeric: consider log-transform.")
        if dt_stats and (dt_stats["min_date"] < "1900-01-01" or dt_stats["max_date"] > "2100-01-01"):
            suggestions.append("Datetime range unusual: check parsing errors.")

        # memory optimization suggestions
        if pd.api.types.is_integer_dtype(col_series):
            min_val, max_val = col_series.min(), col_series.max()
            for t in [np.int8, np.int16, np.int32]:
                if np.iinfo(t).min <= min_val <= max_val <= np.iinfo(t).max:
                    suggestions.append(f"Downcast int64 → {t.__name__}")
                    break
        elif pd.api.types.is_float_dtype(col_series):
            if col_series.astype(np.float32).equals(col_series.dropna()):
                suggestions.append("Downcast float64 → float32")

        columns[col] = {
            "name": col,
            "non_null": non_null,
            "nulls": nulls,
            "null_percent": round(nulls / max(1, total_rows), 4),
            "unique_count": unique_count,
            "unique_percent": unique_pct,
            "cardinality_type": cardinality_type,
            "is_constant": unique_count <= 1,
            "mixed_types_sample": list(types_seen)[:5],
            "inferred_numeric_fraction": round(num_parsed_pct, 4),
            "inferred_datetime_fraction": round(dt_parsed_pct, 4),
            "multi_select_detected": has_separator,
            "top_values": top_values,
            "numeric_stats": numeric_stats,
            "numeric_skew": skew,
            "numeric_kurtosis": kurt,
            "zero_count": zero_count,
            "datetime_stats": dt_stats,
            "entropy": entropy,
            "correlation_with_target": correlation_with_target,
            "sample_values": col_series.dropna().head(5).tolist(),
            "suggested_actions": suggestions
        }

    return {
        "head": head,
        "tail": tail,
        "shape": {"rows": int(total_rows), "columns": int(total_cols)},
        "dtypes": dtypes,
        "memory": {"per_column": memory_per_column, "total": memory_total},
        "missing_summary": missing_summary,
        "duplicates": duplicates_summary,
        "columns": columns
    }

def clean_column_name(name: str) -> str:
    """Chuẩn hóa tên cột: viết thường, bỏ ký tự đặc biệt"""
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name).strip("_")
    return name

def clean_dataset(df: pd.DataFrame, important_cols: list[str] = None, multi_select_cols: list[str] = None) -> dict:
    important_cols = important_cols or []
    multi_select_cols = multi_select_cols or []   

    before_shape = df.shape
    before_missing = df.isnull().sum().to_dict()

    # Chuẩn hóa tên cột
    rename_map = {c: clean_column_name(c) for c in df.columns}
    df = df.rename(columns=rename_map)

    # Chuẩn hóa categorical
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .replace({"nan": None, "none": None})
            )
            df[col] = df[col].fillna("không trả lời")

    # Missing values
    if important_cols:
        df = df.dropna(subset=important_cols)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("không trả lời")

    # ✅ Multi-select expansion CHỈ cho những cột được khai báo
    new_cols = {}
    sep_candidates = [",", ";", "|", " / "]
    for col in multi_select_cols:
        if col in df.columns:
            expanded = df[col].astype(str).str.get_dummies(sep=",")
            expanded = expanded.rename(columns=lambda x: f"{col}__{x.strip().lower()}")
            new_cols[col] = expanded.columns.tolist()
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    after_shape = df.shape
    after_missing = df.isnull().sum().to_dict()

    summary = {
        "before_shape": before_shape,
        "after_shape": after_shape,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "renamed_columns": rename_map,
        "expanded_columns": new_cols,
    }

    return {
        "cleaned_preview": df.head(20).to_dict(orient="records"),
        "summary": convert_numpy_types(summary),
    }

def descriptive_statistics(df: pd.DataFrame, max_categories: int = 10) -> dict:
    result = {
        "numeric": {},
        "categorical": {},
        "remarks": []
    }

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        desc = {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)),
            "skew": float(skew(series)),
            "kurtosis": float(kurtosis(series))
        }

        # ✅ Detect outliers using IQR
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        desc["outliers"] = int(outliers)

        result["numeric"][col] = desc

        # 📝 Rule-based remarks
        if desc["skew"] > 1:
            result["remarks"].append(f"Cột '{col}' phân phối lệch phải (nhiều giá trị nhỏ, ít giá trị rất lớn).")
        elif desc["skew"] < -1:
            result["remarks"].append(f"Cột '{col}' phân phối lệch trái (nhiều giá trị lớn, ít giá trị rất nhỏ).")

        if desc["outliers"] > 0:
            result["remarks"].append(f"Cột '{col}' có {desc['outliers']} giá trị bất thường (outliers).")

    # 🔠 Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        freq = series.value_counts()
        total = len(series)

        # nếu số lượng unique nhỏ hơn 20 thì lấy hết, ngược lại lấy top N
        if series.nunique() <= 20:
            freq = freq
        else:
            freq = freq.head(max_categories)

        values = [
            {"value": val, "count": int(cnt), "percent": round(cnt / total * 100, 2)}
            for val, cnt in freq.items()
        ]

        result["categorical"][col] = {
            "count": int(total),
            "unique": int(series.nunique()),
            "top_values": values
        }

        # 📝 Remarks
        if series.nunique() == 1:
            result["remarks"].append(f"Cột '{col}' chỉ có 1 giá trị duy nhất, ít thông tin.")
        elif series.nunique() < 5:
            result["remarks"].append(f"Cột '{col}' có ít hơn 5 loại dữ liệu → đơn giản.")
        else:
            top_val = values[0]
            if top_val["percent"] > 80:
                result["remarks"].append(f"Cột '{col}' bị lệch mạnh: {top_val['value']} chiếm {top_val['percent']}%.")
    
    return result

def generate_visualizations(df: pd.DataFrame, max_categories: int = 15) -> dict:
    result = {"numeric": {}, "categorical": {}, "multilevel": {}}

    # 📊 Numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if series.empty:
            continue

        # Histogram
        hist_data = np.histogram(series, bins=20)

        result["numeric"][col] = {
            "histogram": {
                "bins": hist_data[1].tolist(),
                "counts": hist_data[0].tolist(),
                "x_label": f"{col} (value)",
                "y_label": "Frequency (count)",
                "description": f"Histogram of {col}"
            },
            "kde": series.tolist()[:5000],  
            "boxplot": {
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "outliers": series[
                    (series < series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25))) |
                    (series > series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25)))
                ].tolist()[:200],
                "y_label": col,
                "description": f"Boxplot showing distribution of {col}"
            }
        }

    # 🔠 Categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        vc = series.value_counts().head(max_categories)
        result["categorical"][col] = {
            "bar": [{"value": k, "count": int(v)} for k, v in vc.items()],
            "pie": [{"value": k, "percent": round(v / len(series) * 100, 2)} for k, v in vc.items()],
            "x_label": col,
            "y_label": "Count",
            "description": f"Distribution of {col} categories"
        }

    # 🌳 Multi-level (Sunburst/Treemap) → ví dụ 2 cột "Country" + "City"
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) >= 2:
        cat_cols = obj_cols[:2]
        grouped = df.groupby(list(cat_cols)).size().reset_index(name="count")
        result["multilevel"]["sunburst"] = {
            "data": grouped.to_dict(orient="records"),
            "labels": list(cat_cols),
            "description": f"Sunburst chart of {cat_cols[0]} and {cat_cols[1]}"
        }

    return result

def generate_relationships(df: pd.DataFrame, max_categories: int = 15, sample_size: int = 500) -> dict:
    """
    Generate relationships between categorical & numeric variables in a format
    optimized for visualization (heatmaps, scatter, bar/box plots, interactive charts).
    """

    result = {
        "categorical_vs_categorical": {},   
        "numeric_vs_numeric": {},           
        "mixed": {},                        
        "interactive": {},
        # "schema": df.columns.tolist()     
    }

    # --- Identify variable types ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- 1. Categorical vs Categorical (normalized heatmap) ---
    for i, col1 in enumerate(cat_cols):
        if df[col1].nunique() > max_categories:
            continue
        for col2 in cat_cols[i+1:]:
            if df[col2].nunique() > max_categories:
                continue

            table = pd.crosstab(df[col1], df[col2], normalize="index")
            melted = table.reset_index().melt(id_vars=col1, var_name=col2, value_name="percentage")

            result["categorical_vs_categorical"][f"{col1}__vs__{col2}"] = melted.to_dict(orient="records")

    # --- 2. Numeric vs Numeric ---
    if len(num_cols) >= 2:
        # Correlation matrix
        corr = df[num_cols].corr()
        result["numeric_vs_numeric"]["correlation"] = corr.round(3).to_dict()

        # Scatter pairs (sampled)
        pairs = []
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                subset = df[[col1, col2]].dropna().sample(min(sample_size, len(df)))
                pairs.append({
                    "x": col1,
                    "y": col2,
                    "data": subset.to_dict(orient="records")
                })
        result["numeric_vs_numeric"]["pairs"] = pairs

    # --- 3. Categorical vs Numeric ---
    for cat in cat_cols:
        if df[cat].nunique() > max_categories:
            continue
        for num in num_cols:
            grouped = df.groupby(cat)[num].agg(["mean", "median", "count"]).reset_index()
            result["mixed"][f"{cat}__vs__{num}"] = grouped.to_dict(orient="records")

    # --- 4. Interactive Visualizations ---
    interactive = {}

    # Sunburst (2 categorical levels)
    if len(cat_cols) >= 2:
        grouped = df.groupby(list(cat_cols[:2])).size().reset_index(name="count")
        interactive["sunburst"] = grouped.to_dict(orient="records")

    # Parallel Coordinates (numeric subset)
    if len(num_cols) >= 3:
        subset = df[num_cols[:5]].dropna().sample(min(300, len(df)))
        interactive["parallel_coordinates"] = subset.to_dict(orient="records")

    # Sankey (cat → num bins)
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        cat = cat_cols[0]
        num = num_cols[0]
        sankey_df = df[[cat, num]].dropna().copy()
        sankey_df[num] = pd.qcut(sankey_df[num], q=5, duplicates="drop").astype(str)  # binning numeric
        sankey_data = sankey_df.groupby([cat, num]).size().reset_index(name="count")
        interactive["sankey"] = sankey_data.to_dict(orient="records")

    result["interactive"] = interactive

    return result

def analyze_multiselect(df, sep=",", top_k=20):
    results = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dropna().astype(str).str.contains(sep).any():
            exploded = df[col].dropna().astype(str).str.split(sep).explode().str.strip()
            freq = exploded.value_counts().reset_index()
            freq.columns = [col, "count"]
            # Giới hạn top-k
            results[col] = freq.head(top_k).to_dict(orient="records")
            # Đa dạng hóa (Shannon entropy)
            p = freq["count"] / freq["count"].sum()
            entropy = -(p * np.log2(p)).sum()
            results[f"{col}_diversity"] = {"entropy": entropy}
    return results

def analyze_clustering(df, num_clusters=4):
    num_cols = df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
    if num_cols.shape[1] < 2: 
        return {}
    
    X = num_cols.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)  # ✅ Clean inf
    if not np.isfinite(X.values).all():
        return {"error": "Dữ liệu có giá trị vô hạn"}
    
    scaler = StandardScaler()
    X = scaler.fit_transform(num_cols.fillna(0))
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    df_out = df.copy()
    df_out["cluster"] = labels
    df["cluster"] = labels
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=num_cols.columns)
    cluster_sizes = pd.Series(labels).value_counts().to_dict()
    
    return {
        "assignments": df_out[["cluster"]].to_dict(orient="records"),
        "centroids": centers.to_dict(orient="records"),
        "cluster_sizes": cluster_sizes,
        "silhouette_score": score
    }

def analyze_highdimensional(df, sample_size=500):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        return []
    sampled = df[num_cols].dropna().sample(min(sample_size, len(df)))
    # Top 3 numeric by variance
    top3 = sampled.var().sort_values(ascending=False).head(3).index.tolist()
    return {
        "sampled": sampled.to_dict(orient="records"),
        "top3": top3
    }

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    return np.sqrt(phi2/min(k-1, r-1))

def eta_squared_anova(f_stat, df_between, df_within):
    return (f_stat * df_between) / (f_stat * df_between + df_within)

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
            return None
        return float(x)
    except Exception:
        return None
    
def analyze_significance(df, max_categories=50):
    results = {"chi2": {}, "anova": {}}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    # --- Chi-squared + Cramér’s V ---
    for i, c1 in enumerate(cat_cols):
        if df[c1].nunique() > max_categories:
            continue
        for c2 in cat_cols[i+1:]:
            if df[c2].nunique() > max_categories:
                continue
            table = pd.crosstab(df[c1], df[c2])
            if table.size == 0:
                results["chi2"][f"{c1}__vs__{c2}"] = {
                    "chi2": None,
                    "p_value": None,
                    "cramers_v": None,
                    "note": "No data"
                }
                continue
            chi2, p, _, _ = chi2_contingency(table)
            cv = cramers_v(table)
            results["chi2"][f"{c1}__vs__{c2}"] = {
                "chi2": safe_float(chi2),
                "p_value": safe_float(p),
                "cramers_v": safe_float(cv),
            }

    # --- ANOVA + η² ---
    for cat in cat_cols:
        if df[cat].nunique() > max_categories:
            continue
        for num in num_cols:
            groups = [vals[num].dropna().values for _, vals in df.groupby(cat)]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                try:
                    f, p = f_oneway(*groups)
                    df_between = len(groups) - 1
                    df_within = len(df) - len(groups)
                    eta2 = eta_squared_anova(f, df_between, df_within)
                    results["anova"][f"{cat}__vs__{num}"] = {
                        "f_stat": safe_float(f),
                        "p_value": safe_float(p),
                        "eta_squared": safe_float(eta2),
                    }
                except Exception as e:
                    results["anova"][f"{cat}__vs__{num}"] = {
                        "f_stat": None,
                        "p_value": None,
                        "eta_squared": None,
                        "note": str(e)
                    }

    return results

def safe_isolation_forest(X, contamination=0.05, random_state=42):
    """
    An toàn chạy IsolationForest sau khi xử lý inf, NaN và kiểm tra tính hợp lệ.
    """
    if X.empty or X.shape[1] < 2:
        return None

    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if not np.isfinite(X_clean.values).all():
        return None

    try:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        return iso.fit_predict(X_clean)
    except Exception:
        return None

def analyze_patterns(df, target=None):
    """
    Phát hiện bất thường + độ quan trọng biến (nếu có target).
    """
    results = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    # --- 1. Phát hiện bất thường ---
    if len(num_cols) >= 2:
        X = df[num_cols]
        scores = safe_isolation_forest(X)
        if scores is not None:
            results["anomalies"] = {"outlier_flags": scores.tolist()}
        else:
            results["anomalies"] = {
                "error": "Không thể phát hiện bất thường: dữ liệu không hợp lệ hoặc thiếu biến."
            }

    # --- 2. Feature Importance (nếu có biến mục tiêu) ---
    if target and target in df.columns:
        X = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df[target].dropna()
        X = X.loc[y.index]  # align X với y

        if len(X) == 0 or len(set(y)) <= 1:
            results["feature_importance"] = {
                "warning": "Không đủ lớp hoặc dữ liệu để tính độ quan trọng biến."
            }
        else:
            try:
                if y.dtype == "object" or y.dtype == "category":
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)

                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                results["feature_importance"] = importances.to_dict()

            except Exception as e:
                results["feature_importance"] = {"error": f"Không thể tính feature importance: {str(e)}"}

    return results

def generate_advanced_eda(df: pd.DataFrame) -> dict:
    return {
        "multiselect": analyze_multiselect(df),
        "clustering": analyze_clustering(df),
        "highdimensional": analyze_highdimensional(df),
        "significance": analyze_significance(df),
        "patterns": analyze_patterns(df),
        "redundancy" : analyze_redundancy(df),
        "timeseries": analyze_timeseries(df, freq="D"),
        # "rfm": analyze_rfm(df)  
    }

def analyze_redundancy(df, vif_threshold=5, corr_threshold=0.9):
    results = {}
    num_cols = df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")
    if num_cols.shape[1] < 2:
        return results

    # --- 1. VIF ---
    X = num_cols.fillna(0)
    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": X.columns[i], "vif": float(vif)})
    results["vif"] = vif_data

    # --- 2. Correlation clustering ---
    corr = X.corr().abs()
    seen = set()
    clusters = []
    for col in corr.columns:
        if col in seen:
            continue
        group = set([col])
        for other in corr.columns:
            if col != other and corr.loc[col, other] >= corr_threshold:
                group.add(other)
                seen.add(other)
        clusters.append(list(group))
        seen.update(group)
    results["correlation_clusters"] = clusters

    return results

def analyze_timeseries(df, freq="D"):
    """
    Phân tích chuỗi thời gian: trend, seasonality, residuals, outliers
    Tự động tìm cột datetime + numeric
    """
    results = {}

    # --- tìm cột datetime ---
    candidate_dates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_col = None
    for c in candidate_dates:
        try:
            pd.to_datetime(df[c].dropna().iloc[0])  # test parse
            date_col = c
            break
        except Exception:
            continue
    if not date_col:
        return results

    # --- tìm cột numeric phù hợp ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    preferred = [c for c in num_cols if c.lower() in ["quantity", "unitprice", "retweet_count"]]
    value_col = preferred[0] if preferred else (num_cols[0] if num_cols else None)
    if not value_col:
        return results

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, value_col])

        # Tổng hợp theo ngày/tuần/tháng
        daily = df.groupby(df[date_col].dt.to_period(freq))[value_col].sum()
        daily.index = daily.index.to_timestamp()

        # Nếu dữ liệu quá ít hoặc phẳng
        if len(daily) < 10 or daily.std() == 0:
            return results

        # Phân rã chuỗi thời gian
        decomposition = seasonal_decompose(daily, model="additive", period=7)

        results["date_col"] = date_col
        results["value_col"] = value_col

        results["aggregated"] = daily.reset_index().rename(
            columns={date_col: "date", value_col: "value"}
        ).to_dict(orient="records")

        results["trend"] = decomposition.trend.dropna().reset_index().rename(
            columns={0: "value", "index": "date"}
        ).to_dict(orient="records")

        results["seasonal"] = decomposition.seasonal.dropna().reset_index().rename(
            columns={0: "value", "index": "date"}
        ).to_dict(orient="records")

        results["resid"] = decomposition.resid.dropna().reset_index().rename(
            columns={0: "value", "index": "date"}
        ).to_dict(orient="records")

        # Outlier detection (z-score theo rolling window)
        df_roll = daily.rolling(window=7, center=True).mean()
        std_roll = daily.rolling(window=7, center=True).std()
        zscores = (daily - df_roll) / std_roll
        outliers = daily[zscores.abs() > 3]

        results["outliers"] = outliers.reset_index().rename(
            columns={date_col: "date", value_col: "value"}
        ).to_dict(orient="records")

    except Exception as e:
        results["error"] = str(e)

    return results

def generate_business_report(eda_results: dict) -> str:
    """
    Generate an executive-level business report from EDA results.
    The style mimics a senior data analyst/consultant with 10+ years of experience.
    """

    sections = []

    # =====================================================
    # 1. Executive Summary
    # =====================================================
    summary = "📌 Executive Summary\n"

    # 1️⃣ Tổng quan dataset
    inspection = eda_results.get("inspection", {})
    shape = inspection.get("shape", {})
    rows = shape.get("rows", 0)
    cols = shape.get("columns", 0)
    summary += f"- Dataset gồm {rows:,} bản ghi và {cols} biến.\n"

    # 2️⃣ Missing data
    missing_summary = inspection.get("missing_summary", {})
    pct_missing = missing_summary.get("percent_missing", 0)
    if pct_missing > 0.1:
        summary += f"- ⚠️ Dữ liệu thiếu đáng kể (~{pct_missing*100:.1f}%) → cần xử lý trước khi phân tích sâu.\n"
    elif pct_missing > 0:
        summary += f"- ⚠️ Một số giá trị bị thiếu (~{pct_missing*100:.1f}%), nên kiểm tra nguyên nhân.\n"
    else:
        summary += "- ✅ Dữ liệu đầy đủ — không có giá trị bị thiếu.\n"

    # 3️⃣ Clustering / Segmentation
    advanced = eda_results.get("advanced", {})
    cluster_info = advanced.get("clustering", {})
    if cluster_info and "centroids" in cluster_info:
        centroids = cluster_info.get("centroids", [])
        n_clusters = len(centroids)
        silhouette = cluster_info.get("silhouette_score", 0)
        summary += f"- 👥 Phân tích phân cụm phát hiện {n_clusters} nhóm chính (điểm silhouette = {silhouette:.3f} → cấu trúc rõ ràng).\n"
        for i, c in enumerate(centroids):
            # Lấy đặc trưng nổi bật (giá trị trung tâm)
            top_features = ", ".join(f"{k}≈{round(v,2)}" for k, v in c.items() if isinstance(v, (int, float)))
            summary += f"    Nhóm {i+1}: đặc trưng bởi {top_features}\n"
    else:
        summary += "- ℹ️ Không thực hiện phân cụm hoặc dữ liệu không phù hợp để phân nhóm.\n"

    # 4️⃣ Time series trends
    ts_info = advanced.get("timeseries", {})
    if ts_info and "aggregated" in ts_info and len(ts_info["aggregated"]) > 0:
        # Lấy ngày đầu và cuối
        dates = [item["date"] for item in ts_info["aggregated"] if "date" in item]
        if dates:
            start_date = min(dates)
            end_date = max(dates)
            summary += f"- 📈 Phân tích chuỗi thời gian từ {start_date} đến {end_date}:\n"
            # Kiểm tra có trend/seasonal không (dựa trên sự tồn tại của decomposition)
            if "trend" in ts_info and len(ts_info["trend"]) > 0:
                summary += "    Xu hướng dài hạn và tính mùa vụ được phát hiện → có thể lập kế hoạch theo chu kỳ.\n"
            if ts_info.get("outliers") and len(ts_info["outliers"]) > 0:
                n_outliers_ts = len(ts_info["outliers"])
                summary += f"    🚨 Phát hiện {n_outliers_ts} điểm bất thường trong chuỗi thời gian → cần điều tra nguyên nhân (khuyến mãi, sự kiện, lỗi hệ thống...).\n"
    else:
        summary += "- ℹ️ Không phát hiện hoặc không đủ dữ liệu để phân tích chuỗi thời gian.\n"

    # 5️⃣ Anomalies (Isolation Forest)
    patterns = advanced.get("patterns", {})
    anomalies = patterns.get("anomalies", {})
    if anomalies and "outlier_flags" in anomalies:
        outlier_flags = anomalies["outlier_flags"]
        n_outliers = sum(1 for f in outlier_flags if f == -1)
        if n_outliers > 0:
            pct_outliers = n_outliers / len(outlier_flags) * 100
            summary += f"- 🚨 Đã phát hiện {n_outliers} điểm dữ liệu bất thường ({pct_outliers:.1f}%) → cần điều tra để loại trừ gian lận hoặc lỗi nhập liệu.\n"
        else:
            summary += "- ✅ Không phát hiện điểm dữ liệu bất thường nào.\n"
    else:
        summary += "- ℹ️ Không thực hiện phân tích bất thường hoặc không đủ biến số để chạy thuật toán.\n"

    sections.append(summary)

    # =====================================================
    # 2. Data Quality & Reliability
    # =====================================================
    dq = "📊 Data Quality & Reliability\n"
    inspection = eda_results.get("inspection", {})
    shape = inspection.get("shape", {})
    rows = shape.get("rows", 0)
    cols = shape.get("columns", 0)
    total_cells = rows * cols

    # 1. Missing data
    missing_summary = inspection.get("missing_summary", {})
    missing_cells = missing_summary.get("total_missing", 0)
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    dq += f"- Có {missing_cells:,} ô trống (~{missing_pct:.1f}% tổng dữ liệu). "
    if missing_pct > 10:
        dq += "⚠️ Mức độ thiếu cao — có thể làm sai lệch kết quả phân tích và dự báo nếu không xử lý.\n"
    elif missing_pct > 0:
        dq += "ℹ️ Nên kiểm tra nguyên nhân thiếu (lỗi nhập liệu, không bắt buộc...) và quyết định impute hay loại bỏ.\n"
    else:
        dq += "✅ Dữ liệu đầy đủ — không có ô trống.\n"

    # 2. Duplicate rows
    duplicates = inspection.get("duplicates", {})
    dup_count = duplicates.get("duplicate_count", 0)
    dup_pct = (dup_count / rows * 100) if rows > 0 else 0
    if dup_count > 0:
        dq += f"- Phát hiện {dup_count:,} bản ghi trùng lặp (~{dup_pct:.1f}% tổng số dòng). "
        dq += "⚠️ Cảnh báo: Có thể dẫn đến phóng đại doanh thu, số lượng giao dịch hoặc khách hàng trong báo cáo.\n"
    else:
        dq += "- ✅ Dữ liệu không có bản ghi trùng lặp — đảm bảo tính duy nhất của từng quan sát.\n"

    # 3. Constant columns
    columns = inspection.get("columns", {})
    const_cols = [col for col, info in columns.items() if info.get("is_constant", False)]
    if const_cols:
        dq += f"- Có {len(const_cols)} biến hằng số (ít giá trị phân tích): {', '.join(const_cols)} → nên xem xét loại bỏ để giảm nhiễu.\n"
    else:
        dq += "- ✅ Không có biến hằng số — tất cả cột đều mang thông tin phân biệt.\n"

    # 4. Overall Data Quality Assessment (Tổng kết đánh giá)
    quality_score = 10
    if missing_pct > 20:
        quality_score -= 4
    elif missing_pct > 5:
        quality_score -= 2

    if dup_count > 0 and dup_pct > 5:
        quality_score -= 3
    elif dup_count > 0:
        quality_score -= 1

    if len(const_cols) > 2:
        quality_score -= 2
    elif len(const_cols) > 0:
        quality_score -= 1

    dq += "\n📈 Đánh giá tổng quan chất lượng dữ liệu: "
    if quality_score >= 9:
        dq += "✅ Rất tốt — Dữ liệu sạch, đáng tin cậy để đưa ra quyết định chiến lược."
    elif quality_score >= 7:
        dq += "🟡 Khá tốt — Có một số vấn đề nhỏ, cần xử lý trước khi huấn luyện mô hình hoặc phân tích sâu."
    elif quality_score >= 5:
        dq += "⚠️ Trung bình — Dữ liệu có vấn đề đáng kể (thiếu, trùng, hằng số) → cần làm sạch kỹ trước khi sử dụng."
    else:
        dq += "🔴 Kém — Dữ liệu không đáng tin cậy do tỷ lệ thiếu/trùng quá cao → nên thu thập hoặc kiểm tra lại dữ liệu gốc."

    sections.append(dq)

    # =====================================================
    # 3. Business Insights
    # =====================================================
    insights = "🔎 Business Insights\n"

    # Segmentation
    cluster = eda_results["advanced"].get("clustering", {})
    if cluster:
        centroids = cluster.get("centroids", [])
        sizes = cluster.get("sizes", {})  # đảm bảo không lỗi nếu thiếu
        insights += (
            f"- Dữ liệu được phân chia thành {len(centroids)} nhóm chính "
            f"(các nhóm có hành vi/đặc điểm khác biệt rõ rệt).\n"
        )
        for i, c in enumerate(centroids):
            size = sizes.get(i, 0)
            top_features = ", ".join(f"{k}≈{round(v,2)}" for k, v in c.items())
            insights += f"    Nhóm {i} ({size} đối tượng): đặc trưng bởi {top_features}. → Chúng ta có thể thiết kế chiến lược riêng cho nhóm này.\n"

    # Time Series
    ts = eda_results["advanced"].get("timeseries", {})
    if ts:
        date_col = ts.get("date_col", "date")
        value_col = ts.get("value_col", "value")
        aggregated = ts.get("aggregated", [])
        if aggregated:
            dates = [item.get("date") for item in aggregated if "date" in item]
            if dates:
                start_date = min(dates)
                end_date = max(dates)
                insights += f"- Phân tích chuỗi thời gian {value_col} từ {start_date} đến {end_date}:\n"
        if ts.get("seasonal") and len(ts["seasonal"]) > 0:
            insights += "    Có tính mùa vụ rõ rệt (lặp lại theo chu kỳ) → phù hợp để lập kế hoạch tồn kho, chiến dịch marketing định kỳ.\n"
        if ts.get("trend") and len(ts["trend"]) > 0:
            first_val = ts["trend"][0].get("value", 0)
            last_val = ts["trend"][-1].get("value", 0)
            if last_val > first_val:
                insights += "    Xu hướng tăng dần theo thời gian → nhu cầu/sản lượng đang phát triển.\n"
            elif last_val < first_val:
                insights += "    Xu hướng giảm dần theo thời gian → cần điều tra nguyên nhân và có biện pháp can thiệp.\n"
        if ts.get("outliers") and len(ts["outliers"]) > 0:
            insights += f"    Phát hiện {len(ts['outliers'])} điểm bất thường → có thể do sự kiện đặc biệt (khuyến mãi, lỗi hệ thống) → cần ghi chú để không ảnh hưởng dự báo.\n"

    # Key Drivers
    sig = eda_results["advanced"].get("significance", {})
    if sig.get("anova"):
        for key, stats in sig["anova"].items():
            if stats.get("p_value", 1) < 0.05:
                cat, num = key.split("__vs__")
                p_val = stats.get("p_value", 0)
                eta2 = stats.get("eta_squared", 0)

                # Xác định mức độ ảnh hưởng
                effect_level = "nhỏ"
                if eta2 >= 0.14:
                    effect_level = "lớn"
                elif eta2 >= 0.06:
                    effect_level = "trung bình"

                insights += (
                    f"- ✅ Kiểm định ANOVA: Biến phân loại {cat} có ảnh hưởng có ý nghĩa thống kê tới {num} "
                    f"(p = {p_val:.3f} < 0.05, η² = {eta2:.3f} → mức độ ảnh hưởng {effect_level}).\n"
                    f"   → Tại sao kết luận như vậy? Vì p-value < 0.05 cho thấy sự khác biệt giữa các nhóm trong '{cat}' là không do ngẫu nhiên. "
                    f"η² cho biết '{cat}' giải thích được {eta2*100:.1f}% sự biến động của '{num}'.\n"
                    f"   → Ý nghĩa: Các nhóm trong '{cat}' có giá trị trung bình '{num}' khác biệt rõ rệt. "
                    f"Có thể tối ưu '{num}' bằng cách điều chỉnh '{cat}'.\n"
                )

    # Key Drivers — Chi-square
    if sig.get("chi2"):
        for key, stats in sig["chi2"].items():
            if stats.get("p_value", 1) < 0.05:
                c1, c2 = key.split("__vs__")
                p_val = stats.get("p_value", 0)
                cramers_v = stats.get("cramers_v", 0)

                # Xác định mức độ liên hệ
                strength = "yếu"
                if cramers_v >= 0.5:
                    strength = "rất mạnh"
                elif cramers_v >= 0.3:
                    strength = "trung bình đến mạnh"
                elif cramers_v >= 0.1:
                    strength = "yếu đến trung bình"

                insights += (
                    f"- ✅ Kiểm định Chi-square: Có mối quan hệ có ý nghĩa thống kê giữa {c1} và {c2} "
                    f"(p = {p_val:.3f} < 0.05, Cramér’s V = {cramers_v:.3f} → mức độ liên hệ {strength}).\n"
                    f"   → Tại sao kết luận như vậy? Vì p-value < 0.05 chứng tỏ mối liên hệ không phải ngẫu nhiên. "
                    f"Cramér’s V đo lường mức độ liên hệ — giá trị {cramers_v:.3f} cho thấy '{c1}' và '{c2}' có xu hướng thay đổi cùng nhau.\n"
                    f"   → Ý nghĩa: Biết giá trị của '{c2}' giúp dự đoán '{c1}' tốt hơn (và ngược lại). "
                
                )

    sections.append(insights)

    # =====================================================
    # 4. Risks & Opportunities
    # =====================================================
    risks = "⚠️ Risks & Opportunities\n"

    # Multicollinearity
    vif_list = eda_results["advanced"].get("redundancy", {}).get("vif", [])
    high_vif = [item for item in vif_list if item["vif"] > 5]
    if high_vif:
        risky_vars = ", ".join([item["feature"] for item in high_vif])
        risks += f"- Một số biến có thông tin trùng lặp cao ({risky_vars}). → Điều này có thể gây nhiễu khi dự báo, cần chọn lọc biến quan trọng nhất để đảm bảo mô hình ổn định.\n"

    # Anomalies
    anomalies = eda_results["advanced"].get("patterns", {}).get("anomalies", {})
    if anomalies.get("outlier_flags"):
        n_outliers = sum(1 for f in anomalies["outlier_flags"] if f == -1)
        risks += (
            f"- Phát hiện {n_outliers} điểm dữ liệu bất thường. "
            f"→ Đây có thể là gian lận, sự kiện đặc biệt, hoặc lỗi hệ thống. "
            f"Nếu không xử lý, có thể dẫn tới sai lệch trong báo cáo hoặc quyết định kinh doanh.\n"
        )

    sections.append(risks)

    # =====================================================
    # 5. Strategic Recommendations
    # =====================================================
    rec = "💡 Strategic Recommendations\n\n"

    # --- 1. Ngắn hạn: Dựa trên Data Quality Issues ---
    inspection = eda_results.get("inspection", {})
    missing_summary = inspection.get("missing_summary", {})
    duplicates = inspection.get("duplicates", {})
    columns = inspection.get("columns", {})

    short_term_actions = []

    # Missing data
    pct_missing = missing_summary.get("percent_missing", 0)
    if pct_missing > 0.05:  # >5%
        top_missing_col = next(iter(missing_summary.get("top_missing_columns", {})), None)
        if top_missing_col:
            short_term_actions.append(f"Làm sạch dữ liệu thiếu (đặc biệt cột '{top_missing_col}') bằng imputation hoặc loại bỏ.")

    # Duplicates
    dup_count = duplicates.get("duplicate_count", 0)
    if dup_count > 0:
        short_term_actions.append(f"Loại bỏ {dup_count:,} bản ghi trùng lặp để đảm bảo tính duy nhất và độ chính xác thống kê.")

    # Constant columns
    const_cols = [col for col, info in columns.items() if info.get("is_constant", False)]
    if const_cols:
        short_term_actions.append(f"Loại bỏ {len(const_cols)} biến hằng số: {', '.join(const_cols)}.")

    if short_term_actions:
        rec += "1. Ngắn hạn (0–3 tháng) — Ưu tiên làm sạch và chuẩn hóa dữ liệu:\n"
        for i, action in enumerate(short_term_actions, 1):
            rec += f"   {i}. {action}\n"
    else:
        rec += "1. Ngắn hạn (0–3 tháng) — Dữ liệu đã sạch, không cần xử lý khẩn cấp.\n"

    rec += "\n"

    # --- 2. Trung hạn: Dựa trên Segmentation / Key Drivers / Relationships ---
    advanced = eda_results.get("advanced", {})
    clustering = advanced.get("clustering", {})
    significance = advanced.get("significance", {})
    patterns = advanced.get("patterns", {})

    mid_term_actions = []

    # Nếu có phân cụm tốt
    sil_score = clustering.get("silhouette_score", 0)
    if sil_score > 0.5 and "centroids" in clustering:
        n_clusters = len(clustering["centroids"])
        mid_term_actions.append(f"Phát triển chiến lược Marketing/CRM theo {n_clusters} phân khúc khách hàng đã xác định (silhouette={sil_score:.2f}).")

    # Nếu có mối quan hệ ANOVA mạnh
    anova_results = significance.get("anova", {})
    strong_anova = [
        key for key, stats in anova_results.items()
        if stats.get("p_value", 1) < 0.05 and stats.get("eta_squared", 0) >= 0.06
    ]
    if strong_anova:
        cat, num = strong_anova[0].split("__vs__")
        mid_term_actions.append(f"Tối ưu '{num}' bằng cách điều chỉnh '{cat}' — đã được chứng minh có ảnh hưởng mạnh (η² > 0.06).")

    # Nếu có mối quan hệ Chi-square mạnh
    chi2_results = significance.get("chi2", {})
    strong_chi2 = [
        key for key, stats in chi2_results.items()
        if stats.get("p_value", 1) < 0.05 and stats.get("cramers_v", 0) >= 0.3
    ]
    if strong_chi2:
        c1, c2 = strong_chi2[0].split("__vs__")
        mid_term_actions.append(f"Cá nhân hóa trải nghiệm/dịch vụ dựa trên mối quan hệ giữa '{c1}' và '{c2}' (Cramér’s V > 0.3).")

    if mid_term_actions:
        rec += "2. Trung hạn (3–12 tháng) — Khai thác insight để tối ưu vận hành:\n"
        for i, action in enumerate(mid_term_actions, 1):
            rec += f"   {i}. {action}\n"
    else:
        rec += "2. Trung hạn (3–12 tháng) — Chưa phát hiện insight mạnh để hành động — cần thu thập thêm dữ liệu hoặc thử nghiệm A/B.\n"

    rec += "\n"

    # --- 3. Dài hạn: Dựa trên Time Series / Predictive Potential / Anomalies ---
    timeseries = advanced.get("timeseries", {})
    anomalies = patterns.get("anomalies", {})

    long_term_actions = []

    # Nếu có chuỗi thời gian
    if timeseries and timeseries.get("aggregated"):
        value_col = timeseries.get("value_col", "giá trị")
        long_term_actions.append(f"Xây dựng mô hình dự báo {value_col} theo thời gian để lập kế hoạch phù hợp.")

    # Nếu có bất thường
    if anomalies and "outlier_flags" in anomalies:
        n_outliers = sum(1 for f in anomalies["outlier_flags"] if f == -1)
        if n_outliers > 0:
            long_term_actions.append("Triển khai hệ thống phát hiện bất thường tự động (real-time anomaly detection) để cảnh báo gian lận hoặc lỗi hệ thống.")

    # Nếu có feature importance (giả sử bạn có target)
    feature_importance = patterns.get("feature_importance", {})
    if isinstance(feature_importance, dict) and len(feature_importance) > 0:
        top_feature = next(iter(feature_importance))
        long_term_actions.append(f"Xây dựng mô hình dự báo dựa trên các biến quan trọng nhất (đứng đầu: '{top_feature}').")

    if long_term_actions:
        rec += "3. Dài hạn (1–3 năm) — Đầu tư vào hệ thống phân tích tiên tiến:\n"
        for i, action in enumerate(long_term_actions, 1):
            rec += f"   {i}. {action}\n"
    else:
        rec += "3. Dài hạn (1–3 năm) — Hiện tại chưa đủ điều kiện để triển khai hệ thống dự báo — cần nâng cấp hạ tầng dữ liệu trước.\n"

    rec += "\n"

    # --- 4. Governance: Dựa trên Data Quality & Monitoring Needs ---
    governance_actions = []

    if pct_missing > 0.1 or dup_count > 0 or const_cols:
        governance_actions.append("Thiết lập Dashboard Giám sát Chất lượng Dữ liệu (Data Quality Dashboard) theo thời gian thực.")

    if "advanced" in eda_results and "redundancy" in eda_results["advanced"]:
        vif_list = eda_results["advanced"]["redundancy"].get("vif", [])
        high_vif_cols = [item["feature"] for item in vif_list if item.get("vif", 0) > 5]
        if high_vif_cols:
            governance_actions.append(f"Thiết lập quy tắc quản trị dữ liệu: cấm đưa đồng thời các biến có VIF > 5 vào cùng một mô hình ({', '.join(high_vif_cols[:3])}...).")

    if governance_actions:
        rec += "4. Governance & DataOps — Xây dựng nền tảng bền vững:\n"
        for i, action in enumerate(governance_actions, 1):
            rec += f"   {i}. {action}\n"
    else:
        rec += "4. Governance & DataOps — Chất lượng dữ liệu hiện tại tốt — có thể hoãn đầu tư hệ thống giám sát chuyên sâu.\n"

    sections.append(rec)

    # =====================================================
    # 6. Appendix (Technical Notes)
    # =====================================================
    appendix = "📑 Appendix / Technical Notes\n"
    appendix += "- Phân tích được thực hiện tự động (Auto EDA + Significance Tests + Clustering).\n"
    appendix += "- Các thuật toán: ANOVA, Chi2-test, Isolation Forest, K-Means, VIF.\n"
    appendix += "- Báo cáo này được tạo tự động nhưng có thể được mở rộng thủ công bởi Data Analyst.\n"
    sections.append(appendix)

    return "\n\n".join(sections)

def extract_eda_insights(result: dict) -> list:
    """
    Automatically extract actionable, business-friendly insights from EDA results.
    Every insight is data-driven — NO hallucination, NO NLP generation.
    Each insight answers: What? → So what? → Now what?
    """
    insights = []

    # --- 1. Missing data ---
    missing_summary = result.get("inspection", {}).get("missing_summary", {})
    top_missing = missing_summary.get("top_missing_columns", {})
    for col, pct in top_missing.items():
        if pct > 0.3:  # >30%
            insights.append(
                f"❗ Column '{col}' has {pct*100:.1f}% missing values → "
                f"this affects the reliability of the analysis. "
                f"→ Recommended: (1) Drop the column if not important, or (2) Impute using median/mode if it's a predictive variable."
            )

    # --- 2. Duplicate rows ---
    duplicates = result.get("inspection", {}).get("duplicates", {})
    dup_count = duplicates.get("duplicate_count", 0)
    if dup_count > 0:
        insights.append(
            f"🔁 Detected {dup_count} duplicate rows → "
            f"could be due to data entry errors or multiple system records. "
            f"→ Action: check and remove to avoid biasing statistics (e.g., duplicated revenue)."
        )

    # --- 3. Constant columns ---
    columns = result.get("inspection", {}).get("columns", {})
    for col, col_info in columns.items():
        if col_info.get("is_constant", False):
            insights.append(
                f"📌 Column '{col}' contains only one unique value → "
                f"provides no information for analysis or modeling. "
                f"→ Recommended to drop to reduce noise and save processing resources."
            )

    # --- 4. High cardinality categorical ---
    for col, col_info in columns.items():
        if col_info.get("inferred_type") in ["categorical", "text"] and col_info.get("unique_percent", 0) > 0.9 and col_info.get("unique_count", 0) > 50:
            insights.append(
                f"🔢 Column '{col}' has {col_info.get('unique_count')} unique values (>90% of total) → "
                f"likely an ID, customer name, or transaction code. "
                f"→ Should not be used directly in classification models — consider hashing, embeddings, or dropping if not needed."
            )

    # --- 5. Numeric insights: skew, outliers ---
    descriptive = result.get("descriptive", {})
    numeric_stats = descriptive.get("numeric", {})
    for col, stats in numeric_stats.items():
        skew_val = stats.get("skew", 0)
        if abs(skew_val) > 1:
            direction = "right" if skew_val > 0 else "left"
            insights.append(
                f"📈 Variable '{col}' is {direction}-skewed (skew = {skew_val:.2f}) → "
                f"distribution is imbalanced, most values cluster on one side. "
                f"→ Consider log/square-root transformation to improve model accuracy."
            )

        outliers = stats.get("outliers", 0)
        total = stats.get("count", 1)
        outlier_pct = outliers / total * 100
        if outlier_pct > 5:
            insights.append(
                f"⚠️ Variable '{col}' has {outliers} outliers ({outlier_pct:.1f}%) → "
                f"could be data entry errors, special events (promotions, system glitches), or fraud. "
                f"→ Action: investigate cause — if error → fix/remove; if real → keep and handle separately (e.g., anomaly analysis)."
            )

    # --- 6. Imbalanced categorical ---
    categorical_stats = descriptive.get("categorical", {})
    for col, stats in categorical_stats.items():
        top_values = stats.get("top_values", [])
        if top_values:
            top_val = top_values[0]
            if top_val.get("percent", 0) > 80:
                insights.append(
                    f"⚖️ Column '{col}' is highly imbalanced: value '{top_val['value']}' accounts for {top_val['percent']}% → "
                    f"machine learning models may be biased toward this group, reducing accuracy for minority classes. "
                    f"→ Recommended: rebalance using oversampling, undersampling, or class weights."
                )

    # --- 7. Suggested actions from inspection ---
    for col, col_info in columns.items():
        suggestions = col_info.get("suggested_actions", [])
        for suggestion in suggestions:
            full_suggestion = f"💡 Suggestion for column '{col}': {suggestion}"
            if full_suggestion not in insights:
                insights.append(full_suggestion)

    # --- 8. ANOVA & Chi-square significance ---
    advanced = result.get("advanced", {})
    significance = advanced.get("significance", {})

    # ANOVA
    anova_results = significance.get("anova", {})
    for key, stats in anova_results.items():
        p_val = stats.get("p_value", 1)
        eta2 = stats.get("eta_squared", 0)
        if p_val is not None and p_val < 0.05:
            cat, num = key.split("__vs__")
            effect = "very small"
            if eta2 >= 0.14:
                effect = "large"
            elif eta2 >= 0.06:
                effect = "medium"
            elif eta2 >= 0.01:
                effect = "small"

            # Insight chuyên môn
            insights.append(
                f"✅ ANOVA: Categorical variable '{cat}' significantly impacts '{num}' "
                f"(p={p_val:.3f}, η²={eta2:.3f} → effect size {effect}). "
                f"→ Interpretation: the mean of '{num}' differs significantly across groups in '{cat}'. "
                f"→ Consider '{cat}' as an important predictor in modeling."
            )

            # Insight dễ hiểu cho non-IT / business
            insights.append(
                f"📊 Business meaning: '{cat}' clearly affects '{num}'. "
                f"Nói cách khác, giá trị trung bình của '{num}' thay đổi đáng kể theo từng nhóm trong '{cat}'. "
                f"→ Với doanh nghiệp: khi thay đổi '{cat}', bạn có thể kỳ vọng '{num}' cũng thay đổi theo."
            )

    # Chi-square
    chi2_results = significance.get("chi2", {})
    for key, stats in chi2_results.items():
        p_val = stats.get("p_value", 1)
        cramers_v = stats.get("cramers_v", 0)
        if p_val is not None and p_val < 0.05:
            col1, col2 = key.split("__vs__")
            strength = "very weak"
            if cramers_v >= 0.5:
                strength = "very strong"
            elif cramers_v >= 0.3:
                strength = "moderate to strong"
            elif cramers_v >= 0.1:
                strength = "weak to moderate"

            # Insight chuyên môn
            insights.append(
                f"✅ Chi-square test: Significant relationship between '{col1}' and '{col2}' "
                f"(p={p_val:.3f}, V={cramers_v:.3f} → strength {strength}). "
                f"→ Interpretation: distribution of '{col1}' changes depending on '{col2}' — one may predict the other. "
                f"→ Modeling note: consider keeping only one if they are redundant."
            )

            # Insight dễ hiểu cho non-IT / business
            insights.append(
                f"📊 Business meaning: '{col1}' có mối quan hệ chặt chẽ với '{col2}' "
                f"(mức độ {strength}). "
                f"→ Với doanh nghiệp: khi biết '{col2}', bạn có thể dự đoán hoặc giải thích xu hướng của '{col1}', "
                
                f"và ngược lại. Ví dụ: nếu '{col2}' là vùng miền và '{col1}' là sản phẩm, thì nhu cầu sản phẩm khác nhau theo vùng."
            )

    # --- 9. Clustering quality ---
    clustering = advanced.get("clustering", {})
    sil_score = clustering.get("silhouette_score")
    labels = clustering.get("labels", [])
    centroids = clustering.get("centroids", {})
    sizes = clustering.get("sizes", {})

    if sil_score is not None:
        if sil_score > 0.5:
            insights.append(
                f"🎯 The data has a clear clustering structure (silhouette score = {sil_score:.3f}) → "
                f"the groups are well separated. "
                f"→ The clustering result shows {len(centroids)} main clusters."
            )

            # Detailed description of each cluster
            for cluster_id, features in enumerate(centroids):
                size = sizes.get(cluster_id, 0)
                feature_desc = ", ".join(
                    f"{k} ≈ {v:.2f}" if isinstance(v, (int, float)) else f"{k} = {v}"
                    for k, v in features.items()
                )
                insights.append(
                    f"🔹 Cluster {cluster_id}: contains {size} data points. Average characteristics: {feature_desc}."
                )

            insights.append(
                "→ The above clusters reflect clear differences in data characteristics. "
                "They can be used for descriptive analysis, segmentation, or as features for modeling."
            )

        elif sil_score < 0.2:
            insights.append(
                f"📉 The data is difficult to cluster (silhouette score = {sil_score:.3f}) → "
                f"the data points do not form clear groups. "
                f"→ This may be due to a lack of suitable features, or because the data is too homogeneous. "
                f"→ Consider engineering additional features or not using clustering in this case."
            )


    # --- 10. Anomalies (Isolation Forest) ---
    patterns = advanced.get("patterns", {})
    anomalies = patterns.get("anomalies", {})
    outlier_flags = anomalies.get("outlier_flags", [])
    if len(outlier_flags) > 0:
        n_outliers = sum(1 for x in outlier_flags if x == -1)
        pct_outliers = n_outliers / len(outlier_flags) * 100
        if n_outliers > 0:
            insights.append(
                f"🚨 Detected {n_outliers} anomalies ({pct_outliers:.1f}%) using Isolation Forest → "
                f"these points differ strongly from the rest of the data. "
                f"→ Action: investigate — could be fraud, system errors, or rare events. If error → remove; if real → analyze separately."
            )

    # --- 11. Multicollinearity (VIF) ---
    redundancy = advanced.get("redundancy", {})
    vif_list = redundancy.get("vif", [])
    high_vif = [item for item in vif_list if item.get("vif", 0) > 5]
    if high_vif:
        high_vif_cols = ", ".join([f"'{item['feature']}' (VIF={item['vif']:.1f})" for item in high_vif])
        insights.append(f"🔗 Multicollinearity warning: The following variables have VIF > 5 → {high_vif_cols}")

    # --- 12. Sample size warning ---
    shape = result.get("inspection", {}).get("shape", {})
    total_rows = shape.get("rows", 0)
    if total_rows < 50:
        insights.append(
            f"📉 Dataset is too small ({total_rows} rows) → "
            f"statistical tests (significance, clustering, prediction) may be unreliable due to high sampling error. "
            f"→ Recommended: collect more data before making strategic decisions or training models."
        )
    
    # Remove duplicates and return
    return list(dict.fromkeys(insights))

@app.post("/api/parse-file")
async def parse_file(file: UploadFile = File(...)):
    name = file.filename.lower()
    content = await file.read()
    
    file_size_mb = len(content) / (1024 * 1024)
    logger.info(f"Received file: {file.filename} ({file_size_mb:.2f} MB)")

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(BytesIO(content), nrows=10000)
        elif name.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, list):
                df = pd.json_normalize(data[:10000])
            else:
                df = pd.json_normalize([data])
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content), nrows=10000)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")

    # ✅ Clean NaN/Inf → None
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.map(lambda x: None if isinstance(x, float) and np.isnan(x) else x)  # ← .map, not .applymap

    schema = infer_schema_from_df(df)
    preview = df.head(10).to_dict(orient="records")
    understanding = []
    for col in df.columns:
        understanding.append(analyze_column(df[col]))
    
    # Step 2: Detailed Inspection
    inspection = inspect_dataset(df, max_sample=10)
    
    cleaned_result = clean_dataset(df, important_cols=[])
    preview = cleaned_result["cleaned_preview"]  

    descriptive = descriptive_statistics(df)

    visualizations = generate_visualizations(df)

    relationships = generate_relationships(df)

    advanced = generate_advanced_eda(df)

    print("parse_file: Sending data to run_prediction for forecasting")
    prediction_result = await predict_from_df(df)
    print("parse_file: Received prediction result")
    result = {
        "schema": schema,
        "preview": preview,
        "understanding": understanding,
        "inspection": inspection,
        "cleaning": cleaned_result, 
        "descriptive": descriptive,
        "visualizations": visualizations,
        "relationships": relationships,
        "advanced": advanced,
        
        "metadata": {
            "original_file_size_mb": round(file_size_mb, 2),
            "final_shape": df.shape,
            "sampled": file_size_mb > 10 
        }
    }

    insights = extract_eda_insights(result)

    # ✅ THÊM insights VÀO result
    result["insights"] = insights
    
    business_report = generate_business_report(result)

    result["business_report"] = business_report

    # ✅ In sau khi đã có result
    print("All columns:", df.columns.tolist())
    print("Numeric cols:", list(descriptive["numeric"].keys()))
    print("Categorical cols:", list(descriptive["categorical"].keys()))
    print("Unique counts per categorical column:")
    for col in list(descriptive["categorical"].keys()):
        print(f"  {col}: {df[col].nunique()} unique values")
    cleaned = convert_numpy_types(result)
    cleaned["prediction_result"] = prediction_result
    return JSONResponse(
        content=cleaned,
        media_type="application/json"
    )

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": "unknown", 
        "memory_usage_mb": "not tracked"
    }

#============================================================
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from io import BytesIO
import json
from typing import Tuple, Dict, Any, List, Union
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
# === Scikit-learn ===
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch, OPTICS
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet 
# from tbats import TBATS, BATS, Theta  

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input, Conv1D, GlobalMaxPooling1D, Embedding

from tcn import TCN 


# Load pipeline config 1 lần khi app start
with open("pipelineAutoML.json", "r") as f:
    pipeline_config = json.load(f)

async def read_file_to_df(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    name = file.filename.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(BytesIO(content))
        elif name.endswith(".json"):
            return pd.read_json(BytesIO(content))
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

def detect_data_types(df: pd.DataFrame, detection_config: Dict[str, Any]) -> Dict[str, str]:
    detected = {}

    for col in df.columns:
        col_dtype = df[col].dtype

        # --- Time series detection ---
        if detection_config["time_series"]["enabled"]:
            # Nếu dtype đã là datetime
            if pd.api.types.is_datetime64_any_dtype(col_dtype):
                detected[col] = "time_series"
                continue

            # Nếu là object/string nhưng parse được sang datetime
            if col_dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                    # Nếu > 80% giá trị parse thành datetime hợp lệ → coi là time_series
                    if parsed.notna().mean() > 0.8:
                        detected[col] = "time_series"
                        continue
                except Exception:
                    pass

        # --- Numeric detection ---
        if detection_config["numeric"]["enabled"]:
            if col_dtype.name in detection_config["numeric"]["criteria"]:
                detected[col] = "numeric"
                continue

        # --- Categorical detection ---
        if detection_config["categorical"]["enabled"]:
            if col_dtype.name in detection_config["categorical"]["criteria"]:
                detected[col] = "categorical"
                continue

        # --- Text detection ---
        if detection_config["text"]["enabled"]:
            try:
                sample_vals = df[col].dropna().astype(str).sample(min(100, len(df[col])), random_state=42)
                avg_len = sample_vals.str.len().mean()
                avg_tokens = sample_vals.str.split().map(len).mean()
                if avg_len > 50 and avg_tokens > 5:
                    detected[col] = "text"
                    continue
            except Exception:
                pass

        # --- Default: categorical ---
        detected[col] = "categorical"

    return detected

def preprocess_data(
    df: pd.DataFrame, 
    detected_types: Dict[str, str], 
    preprocessing_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder

    df_processed = df.copy()
    log = {}
    updated_types = detected_types.copy()  # copy từ EDA (raw)

    for col, dtype in list(detected_types.items()):
        log[col] = []

        # --- Numeric ---
        if dtype == "numeric":
            imp_strategy = preprocessing_config["numeric"]["imputer"][0] 
            imputer = SimpleImputer(strategy=imp_strategy)
            df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
            log[col].append(f"imputer: {imp_strategy}")

            scaler_name = preprocessing_config["numeric"]["scaling"][0]
            if scaler_name == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_name == "RobustScaler":
                scaler = RobustScaler()
            elif scaler_name == "QuantileTransformer":
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                scaler = None

            if scaler:
                df_processed[[col]] = scaler.fit_transform(df_processed[[col]])
                log[col].append(f"scaler: {scaler_name}")

            outlier_method = preprocessing_config["numeric"]["outlier_handling"][0]
            log[col].append(f"outlier_handling: {outlier_method}")

        # --- Categorical ---
        elif dtype == "categorical":
            imp_strategy = preprocessing_config["categorical"]["imputer"][0]
            imputer = SimpleImputer(
                strategy=imp_strategy, 
                fill_value="missing" if imp_strategy == "missing_category" else None
            )
            df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
            log[col].append(f"imputer: {imp_strategy}")

            encoding_method = preprocessing_config["categorical"]["encoding"][0]
            if encoding_method == "OneHotEncoder":
                df_processed[col] = df_processed[col].astype(str)
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded = encoder.fit_transform(df_processed[[col]])
                cols_encoded = [f"{col}__{cat}" for cat in encoder.categories_[0]]
                df_encoded = pd.DataFrame(encoded, columns=cols_encoded, index=df_processed.index)

                # Replace cột gốc bằng cột mới
                df_processed = pd.concat([df_processed.drop(columns=[col]), df_encoded], axis=1)
                log[col].append("encoding: OneHotEncoder")

                # ✅ vẫn giữ thông tin column gốc, không thêm từng dummy col
                updated_types[col] = "categorical_encoded"

            else:
                log[col].append(f"encoding: {encoding_method} (not implemented)")

        # --- Text ---
        elif dtype == "text":
            if "lowercase" in preprocessing_config["text"]["cleaning"]:
                df_processed[col] = df_processed[col].astype(str).str.lower()
                log[col].append("cleaning: lowercase")
            
        # --- Time series ---
        elif dtype == "time_series":
            # Convert datetime if not already
            df_processed[col] = pd.to_datetime(df_processed[col], errors="coerce", utc=True)
            log[col].append("converted to datetime")

            # Extract useful features
            df_processed[f"{col}_year"] = df_processed[col].dt.year
            df_processed[f"{col}_month"] = df_processed[col].dt.month
            df_processed[f"{col}_day"] = df_processed[col].dt.day
            df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek

            # Drop raw datetime col (vì sklearn không hiểu được)
            df_processed = df_processed.drop(columns=[col])
            log[col].append("extracted year, month, day, dayofweek")

    return df_processed, log, updated_types

async def predict_from_df(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    print("Received data for prediction")

    # --- Step 1: Detect types ---
    detected_types = detect_data_types(df, pipeline_config["data_type_detection"])

    # --- Step 2: Preprocessing ---
    df_processed, preprocessing_log, updated_types = preprocess_data(
        df, detected_types, pipeline_config["preprocessing"]
    )

    # --- Step 3: Feature engineering ---
    df_fe, fe_log = feature_engineering(
        df_processed, updated_types, pipeline_config["feature_engineering"]
    )

    # --- Step 4: Feature selection ---
    df_fs, fs_log = feature_selection(
        df_fe, updated_types, pipeline_config["feature_selection"], target_col=target_col
    )

    # --- Step 5: Model selection ---
    model_selection_log = select_models_for_training(
        df_fs, updated_types, pipeline_config["model_selection"], target_col=target_col
    )

    # --- Step 6: Model training ---
    training_results = []
    res = train_models(
        df_fs,
        model_selection_log=model_selection_log,
        training_config=pipeline_config["model_training"]
    )

    # --- Step 7: Model comparison ---
    comparison = model_comparison(
        res,
        model_selection_log=model_selection_log,
        training_config=pipeline_config["model_training"]
    )

    training_results = [{
        "target_col": model_selection_log["target_col"],
        "problem_type": model_selection_log["problem_type"],
        "results": res,
        "comparison": comparison
    }]

    # --- Final output ---
    return {
        "detected_types": updated_types,
        "preprocessing_log": preprocessing_log,
        "feature_engineering_log": fe_log,
        "feature_selection_log": fs_log,
        "model_selection_log": model_selection_log,
        "training_results": training_results,
        "processed_data_preview": df_fs.head(5).to_dict(orient="records"),
    }

def feature_engineering(
    df: pd.DataFrame,
    detected_types: Dict[str, str],
    fe_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    import pandas as pd

    df_fe = df.copy()
    log = {}

    num_cols = [c for c, t in detected_types.items() if t == "numeric"]
    cat_cols = [c for c, t in detected_types.items() if t == "categorical"]

    # --- Numeric feature engineering ---
    if len(num_cols) >= 2 and "numeric" in fe_config:
        log["numeric"] = []
        strategies = fe_config["numeric"]["strategies"]

        # Interaction terms
        if "interaction_terms" in strategies:
            new_cols = {
                f"{c1}_x_{c2}": df[c1].values * df[c2].values
                for i, c1 in enumerate(num_cols)
                for c2 in num_cols[i+1:]
            }
            if new_cols:
                df_fe = pd.concat([df_fe, pd.DataFrame(new_cols, index=df.index)], axis=1)
                log["numeric"].append(f"Created {len(new_cols)} interaction features")

        # Polynomial features
        poly_degrees = [int(s.split("=")[1]) for s in strategies if s.startswith("polynomial_degree")]
        if poly_degrees:
            degree = max(poly_degrees)
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(df[num_cols].values)
            poly_names = poly.get_feature_names_out(num_cols)

            new_poly = {name: poly_features[:, i] for i, name in enumerate(poly_names) if name not in df_fe.columns}
            if new_poly:
                df_fe = pd.concat([df_fe, pd.DataFrame(new_poly, index=df.index)], axis=1)
                log["numeric"].append(f"Applied PolynomialFeatures degree={degree}, added {len(new_poly)} features")

    # --- Categorical cross features ---
    if len(cat_cols) >= 2 and "categorical" in fe_config:
        if "cross_features" in fe_config["categorical"]["strategies"]:
            log["categorical"] = []
            new_cols = {}
            for i, c1 in enumerate(cat_cols):
                c1_vals = df[c1].astype(str).values
                for c2 in cat_cols[i+1:]:
                    c2_vals = df[c2].astype(str).values
                    new_col = f"{c1}_x_{c2}"
                    new_cols[new_col] = np.char.add(np.char.add(c1_vals, "_"), c2_vals)
                    log["categorical"].append(f"Created cross feature {new_col}")
            if new_cols:
                df_fe = pd.concat([df_fe, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df_fe, log

def feature_selection(
    df: pd.DataFrame,
    detected_types: Dict[str, str],
    fs_config: Dict[str, Any],
    target_col: str = None   
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LassoCV

    df_fs = df
    log = {}

    # Numeric feature selection
    num_cols = [c for c, t in detected_types.items() if t == "numeric" and c in df_fs.columns]
    
    if num_cols and "numeric" in fs_config:
        log["numeric"] = []

        # Variance Threshold
        if "variance_threshold" in fs_config["numeric"]:
            sel = VarianceThreshold(threshold=0.0)
            sel.fit(df_fs[num_cols])

            support_mask = sel.get_support()
            kept = [col for col, keep in zip(num_cols, support_mask) if keep]
            dropped = [col for col, keep in zip(num_cols, support_mask) if not keep]

            df_fs = pd.concat([df_fs[kept], df_fs.drop(columns=num_cols)], axis=1)
            log["numeric"].append({
                "method": "variance_threshold",
                "kept": kept,
                "dropped": dropped,
                "explanation": "Dropped features with zero variance (no information)."
            })

        # PCA
        if "pca" in fs_config["numeric"] and len(num_cols) > 1:
            pca = PCA(n_components=min(5, len(num_cols)))
            components = pca.fit_transform(df_fs[num_cols])
            for i in range(components.shape[1]):
                df_fs[f"pca_comp_{i+1}"] = components[:, i]

            log["numeric"].append({
                "method": "pca",
                "original_features": num_cols,
                "components_added": [f"pca_comp_{i+1}" for i in range(components.shape[1])],
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explanation": "PCA reduces dimensionality by projecting onto top components."
            })

        # Lasso (chỉ chạy nếu có target)
        if "lasso" in fs_config["numeric"]:
            if target_col and target_col in df_fs.columns:
                try:
                    X = df_fs[num_cols]
                    y = df_fs[target_col]
                    lasso = LassoCV(cv=5)
                    lasso.fit(X, y)
                    selected_features = [num_cols[i] for i, coef in enumerate(lasso.coef_) if coef != 0]

                    log["numeric"].append({
                        "method": "lasso",
                        "selected_features": selected_features,
                        "explanation": "Lasso selected features with strongest predictive signal."
                    })
                except Exception as e:
                    log["numeric"].append({
                        "method": "lasso",
                        "error": str(e),
                        "explanation": "Lasso skipped due to error."
                    })
            else:
                log["numeric"].append({
                    "method": "lasso",
                    "selected_features": [],
                    "explanation": "Skipped because no target variable was provided."
                })

    # Categorical
    if "categorical" in fs_config:
        log["categorical"] = ["chi2 (placeholder)", "mutual_info (placeholder)"]

    # Text
    if "text" in fs_config:
        log["text"] = ["tfidf_feature_importance (placeholder)", "embedding_dim_reduction (placeholder)"]

    # Time series
    if "time_series" in fs_config:
        log["time_series"] = ["autocorrelation_selection (placeholder)", "feature_importance (placeholder)"]

    return df_fs, log

def auto_detect_target(df: pd.DataFrame, detected_types: Dict[str, str]) -> str:
    common_target_names = ["target", "label", "y", "class", "price", "medv"]

    candidate_cols = [c for c in df.columns if "cluster" not in c.lower()]

    # Nếu chỉ có 1 numeric duy nhất
    numeric_cols = [c for c in candidate_cols if detected_types.get(c) == "numeric"]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    # Nếu có tên phổ biến
    for c in candidate_cols:
        if c.lower() in common_target_names:
            return c

    # Nếu có numeric continuous nhiều giá trị
    for c in numeric_cols:
        if df[c].nunique() > 20 and pd.api.types.is_numeric_dtype(df[c]):
            return c

    return None

def select_models_for_training(
    df: pd.DataFrame,
    detected_types: Dict[str, str],
    model_config: Dict[str, Any],
    target_col: str = None
) -> Dict[str, Any]:
    
    if not target_col:
        target_col = auto_detect_target(df, detected_types)

    # --- 1. Xác định loại bài toán ---
    if target_col and target_col in df.columns:
        y = df[target_col]

        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() <= 20 and str(y.dtype).startswith("int"):
                problem_type = "classification"
            else:
                problem_type = "regression"
        else:
            problem_type = "classification"
    else:
        problem_type = "clustering"

    # --- 2. Tạo context dataset ---
    n_features = df.shape[1] - (1 if target_col else 0)
    dataset_size = df.shape[0]

    context = {
        "problem_type": problem_type,
        "n_features": n_features,
        "dataset_size": dataset_size,
        "has_imbalanced_target": False,
        "interpretability_required": False,
        "has_seasonality": "time_series" in detected_types.values(),
        "has_lag_features": any("lag" in c for c in df.columns),
        "contains_text_features": any(v == "text" for v in detected_types.values()),
        "contains_categorical_features": any(v == "categorical" for v in detected_types.values()),
        "n_unique_categories": max(
            [df[c].nunique() for c, t in detected_types.items() if t == "categorical"],
            default=0
        ),
        "has_arbitrary_shapes": False
    }

    # --- 3. Check imbalance ---
    if target_col and problem_type == "classification":
        y_counts = df[target_col].value_counts(normalize=True)
        if y_counts.min() < 0.2:
            context["has_imbalanced_target"] = True

    # --- 4. Rule-based auto-select ---
    selected_models, justification = None, None
    auto_select_cfg = model_config.get("model_selection", {}).get("auto_select", {})
    justification_cfg = auto_select_cfg.get("justification", {})

    if auto_select_cfg.get("enabled", False):
        rules = justification_cfg.get("rules", [])
        for rule in rules:
            try:
                safe_globals = {
                    "__builtins__": {},
                    "true": True, "false": False, "True": True, "False": False
                }
                if eval(rule["condition"], safe_globals, context):
                    selected_models = rule.get("selected_models", ["RandomForest"])
                    justification = rule.get("justification", "Rule-based selection.")
                    break
            except Exception as e:
                print(f"⚠️ Rule eval error: {rule.get('condition')} → {e}")
                continue

    # --- 5. Fallback ---
    if not selected_models:
        fallback = justification_cfg.get(
            "fallback",
            {"selected_models": ["RandomForest"], "justification": "Fallback default model."}
        )
        selected_models = fallback.get("selected_models", ["RandomForest"])
        justification = fallback.get("justification", "Fallback default model.")

    return {
        "target_col": target_col,
        "problem_type": problem_type,
        "selected_models": selected_models,
        "justification": justification,
        "context": context
    }

def production_phase_tuning(model_cls, X_train, y_train, problem_type, training_config):
    def objective(trial):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42,
            stratify=y_train if problem_type == "classification" else None
        )

        params = {}
        model_name = model_cls.__name__

        # === Tree-based Models ===
        if model_name in ["RandomForestClassifier", "RandomForestRegressor",
                          "ExtraTreesClassifier", "ExtraTreesRegressor",
                          "GradientBoostingClassifier", "GradientBoostingRegressor"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42
            }

        elif model_name in ["XGBClassifier", "XGBRegressor"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42
            }

        elif model_name in ["LGBMClassifier", "LGBMRegressor"]:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42
            }

        elif model_name in ["CatBoostClassifier", "CatBoostRegressor"]:
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": 42,
                "verbose": False
            }

        # === Linear Models ===
        elif model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 500
            }

        elif model_name in ["Ridge", "Lasso", "ElasticNet"]:
            params = {
                "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True)
            }
            if model_name == "ElasticNet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        # === SVM ===
        elif model_name in ["SVC", "SVR", "LinearSVC"]:
            params = {
                "C": trial.suggest_float("C", 1e-3, 10, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }

        # === KNN ===
        elif model_name in ["KNeighborsClassifier", "KNeighborsRegressor"]:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
            }

        # === MLP ===
        elif model_name in ["MLPClassifier", "MLPRegressor"]:
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", [(50,), (100,), (100, 50), (200, 100)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "max_iter": 500,
                "early_stopping": True
            }

        # === Clustering ===
        elif model_name == "KMeans":
            params = {
                "n_clusters": trial.suggest_int("n_clusters", 2, 10),
                "init": trial.suggest_categorical("init", ["k-means++", "random"]),
                "n_init": 10,
                "random_state": 42
            }

        elif model_name == "DBSCAN":
            params = {
                "eps": trial.suggest_float("eps", 0.1, 5.0),
                "min_samples": trial.suggest_int("min_samples", 3, 20)
            }

        elif model_name == "GaussianMixture":
            params = {
                "n_components": trial.suggest_int("n_components", 2, 10),
                "covariance_type": trial.suggest_categorical(
                    "covariance_type", ["full", "tied", "diag", "spherical"]
                ),
                "random_state": 42
            }

        # === Time Series ===
        elif model_name == "Prophet":
            params = {
                "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.01, 0.5),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 1.0, 10.0)
            }

        elif model_name in ["ARIMA", "SARIMAX"]:
            params = {
                "order": (
                    trial.suggest_int("p", 0, 5),
                    trial.suggest_int("d", 0, 2),
                    trial.suggest_int("q", 0, 5)
                )
            }

        # Build model
        model = model_cls(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        if problem_type == "classification":
            return accuracy_score(y_val, preds)
        elif problem_type == "regression":
            return mean_squared_error(y_val, preds) ** 0.5
        else:
            return silhouette_score(X_val, preds) if hasattr(model, "labels_") else 0

    study = optuna.create_study(
        direction="maximize" if problem_type == "classification" else "minimize"
    )
    study.optimize(objective, n_trials=30,
                   timeout=training_config["training_config"]["max_runtime_minutes"] * 60)

    return study.best_params

def get_model_class(model_name: str, problem_type: str):
    """
    Trả về class model (chưa khởi tạo) dựa vào tên và loại bài toán.
    Bao phủ hầu hết các mô hình ML phổ biến (trừ LLM).
    """
    # === Classification ===
    if problem_type == "classification":
        classifiers = {
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "LinearSVC": LinearSVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
            "CatBoostClassifier": CatBoostClassifier,
            "RandomForest": RandomForestClassifier,
            # "MLPClassifier": MLPClassifier,
            "NaiveBayes": GaussianNB,
            "QDA": QuadraticDiscriminantAnalysis,
            "AdaBoostClassifier": AdaBoostClassifier,
            "BaggingClassifier": BaggingClassifier
        }
        return classifiers.get(model_name)

    # === Regression ===
    elif problem_type == "regression":
        regressors = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "RandomForestRegressor": RandomForestRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor,
            "CatBoostRegressor": CatBoostRegressor,
            "RandomForest": RandomForestRegressor,
            # "MLPRegressor": MLPRegressor,
            "AdaBoostRegressor": AdaBoostRegressor,
            "BaggingRegressor": BaggingRegressor
        }
        return regressors.get(model_name)

    # === Clustering ===
    elif problem_type == "clustering":
        clusterers = {
            "KMeans": KMeans,
            "MiniBatchKMeans": MiniBatchKMeans,
            "DBSCAN": DBSCAN,
            # "HDBSCAN": HDBSCAN,
            "AgglomerativeClustering": AgglomerativeClustering,
            # "GaussianMixture": GaussianMixture,
            "SpectralClustering": SpectralClustering,
            "Birch": Birch,
            "OPTICS": OPTICS
        }
        return clusterers.get(model_name)

    # === Time Series Forecasting ===
    elif problem_type == "time_series":
        ts_models = {
            "ARIMA": ARIMA,
            "SARIMA": SARIMAX,
            "ExponentialSmoothing": ExponentialSmoothing,
            "Prophet": Prophet,
            # "Theta": Theta,
            "XGBRegressor": XGBRegressor,  # with lag features
            "LGBMRegressor": LGBMRegressor,
            "CatBoostRegressor": CatBoostRegressor,
            # "RNN": SimpleRNN,
            # "LSTM": LSTM,
            # "GRU": GRU,
            # "TCN": TemporalConvNet
        }
        return ts_models.get(model_name)

    # === Text-specific Models (shallow ML, không phải LLM) ===
    elif problem_type == "text":
        text_models = {
            "LogisticRegression": LogisticRegression,  # baseline for text classification
            "LinearSVC": LinearSVC,
            "NaiveBayes": MultinomialNB,
            "SGDClassifier": SGDClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
            # "CNN_Text": TextCNN,
            # "RNN_Text": LSTM
        }
        return text_models.get(model_name)

    return None
def train_models(df, model_selection_log, training_config):
    target_col = model_selection_log.get("target_col")
    problem_type = model_selection_log.get("problem_type")
    model_candidates = model_selection_log.get("selected_models", [])

    if target_col and target_col in df.columns:
        X = df.drop(columns=target_col)
        y = df[target_col]
    else:
        # Clustering không có target
        X, y = df, None

    split_cfg = training_config.get("train_test_split", {})
    if problem_type != "clustering" and target_col:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42),
            stratify=y if problem_type == "classification" else None
        )
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    results = {}
    for model_name in model_candidates:
        print(f"Training with Optuna: {model_name}")
        model_cls = get_model_class(model_name, problem_type)
        if not model_cls:
            print(f"⚠️ Model {model_name} chưa được hỗ trợ")
            continue

        # Optuna tuning
        best_params = {}
        if problem_type in ["classification", "regression"]:
            try:
                best_params = production_phase_tuning(
                    model_cls, X_train, y_train, problem_type, training_config
                )
            except Exception as e:
                print(f"⚠️ Optuna tuning failed for {model_name}: {e}")

        # Train final model
        try:
            model = model_cls(**best_params)
        except TypeError:
            model = model_cls()

        try:
            if problem_type != "clustering":
                model.fit(X_train, y_train)
            else:
                model.fit(X)
        except Exception as e:
            print(f"⚠️ Training failed for {model_name}: {e}")
            continue

        # Evaluate test_score
        test_score, preds = None, None
        try:
            if problem_type == "classification":
                preds = model.predict(X_test)
                test_score = accuracy_score(y_test, preds)

            elif problem_type == "regression":
                preds = model.predict(X_test)
                test_score = root_mean_squared_error(y_test, preds)

            elif problem_type == "clustering":
                if hasattr(model, "labels_"):
                    preds = model.labels_
                    if X.shape[0] > 1 and X.shape[0] > X.shape[1]:
                        test_score = silhouette_score(X, preds)

            elif problem_type == "time_series":
                if X_test is not None:
                    preds = model.predict(len(X_test))
                    test_score = mean_squared_error(y_test, preds) ** 0.5

            elif problem_type == "text":
                preds = model.predict(X_test)
                test_score = f1_score(y_test, preds, average="weighted")
        except Exception as e:
            print(f"⚠️ Evaluation failed for {model_name}: {e}")
            test_score = None

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.joblib"
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"⚠️ Saving model {model_name} failed: {e}")

        results[model_name] = {
            "best_params": best_params,
            "model_path": model_path,
            "test_score": test_score,
            "preds": preds.tolist() if preds is not None else None
        }

    return results


def model_comparison(results: Dict[str, Any], model_selection_log: Dict[str, Any], training_config: Dict[str, Any]) -> Dict[str, Any]:
    problem_type = model_selection_log.get("problem_type")
    selection_metric = training_config.get("model_comparison", {}).get("selection_metric", "auto")

    metric = "accuracy" if (selection_metric == "auto" and problem_type == "classification") else "rmse"

    # Lọc ra những model có test_score hợp lệ
    valid_results = {m: r for m, r in results.items() if r.get("test_score") is not None}
    if not valid_results:
        raise ValueError("No valid models with test_score found. Check training pipeline.")

    if problem_type == "classification":
        best_model = max(valid_results, key=lambda m: valid_results[m]["test_score"])
    else:
        best_model = min(valid_results, key=lambda m: valid_results[m]["test_score"])

    best_score = valid_results[best_model]["test_score"]

    # Explanation
    explanation_cfg = training_config.get("model_comparison", {}).get("explanation_template", {})
    explanation = explanation_cfg.get("winning_model", "{model_name} wins with {metric_name}={score}").format(
        model_name=best_model,
        metric_name=metric,
        score=best_score,
        alpha=training_config.get("model_comparison", {}).get("alpha", 0.05)
    )
    justification = model_selection_log.get("justification", "")

    return {
        "best_model": best_model,
        "best_score": best_score,
        "p_value": None,  # Wilcoxon bỏ qua tạm
        "explanation": f"{explanation}\n→ {justification}"
    }

@app.post("/api/prediction")
async def run_prediction(file: UploadFile = File(...), target: str = Form(None)):
    df = await read_file_to_df(file)

    # Detect types
    detected_types = detect_data_types(df, pipeline_config["data_type_detection"])

    # Determine target
    if target and target in df.columns:
        target_col = target
    else:
        target_col = auto_detect_target(df, detected_types)

    # Đánh dấu target
    if target_col:
        detected_types[target_col] = "target"

    # Run pipeline
    result = await predict_from_df(df, target_col=target_col)

    # Attach info
    result["target_col"] = target_col
    result["candidate_targets"] = [
        c for c in df.columns if "cluster" not in c.lower()
    ]
    return result

