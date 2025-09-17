# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
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
from sklearn.metrics import silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
import logging
from typing import Optional

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

def infer_schema_from_df(df: pd.DataFrame) -> dict:
    props = {}
    for col in df.columns:
        series = df[col].dropna()
        dtype = "string"
        if pd.api.types.is_integer_dtype(series):
            dtype = "integer"
        elif pd.api.types.is_float_dtype(series):
            dtype = "number"
        elif pd.api.types.is_bool_dtype(series):
            dtype = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = "string"
            fmt = "date-time"
        else:
            sample = series.astype(str).head(100).tolist()
            if all(s.isdigit() for s in sample if s not in ("", "nan")):
                dtype = "integer"
            else:
                dtype = "string"
        prop = {"type": dtype}
        props[col] = prop
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GeneratedSchema",
        "type": "object",
        "properties": props
    }

def analyze_column(col: pd.Series) -> dict:
    series = col.dropna().astype(str)
    dtype = "text"
    unique_count = series.nunique()
    total = len(series)

    # Xác định loại dữ liệu
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        dtype = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(col):
        dtype = "datetime"
    else:
        # thử phân biệt categorical vs text
        if unique_count < 0.05 * total:  # ít hơn 5% giá trị khác biệt
            dtype = "categorical"
        elif series.str.contains(",").any():
            dtype = "multi-select"
        else:
            dtype = "text"

    # Thống kê cơ bản
    stats = {}
    if dtype == "numeric":
        numeric_col = pd.to_numeric(col, errors="coerce")
        stats = {
            "min": convert_numpy_types(numeric_col.min()),
            "max": convert_numpy_types(numeric_col.max()),
            "mean": convert_numpy_types(numeric_col.mean()),
            "std": convert_numpy_types(numeric_col.std()),
        }
    elif dtype == "categorical" or dtype == "multi-select":
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

def inspect_dataset(df: pd.DataFrame, max_sample: int = 10) -> dict:
    """Comprehensive Data Inspection (Bước 2) — returns detailed diagnostics."""
    total_rows, total_cols = df.shape
    # head & tail
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
        "columns_missing": convert_numpy_types(null_counts.to_dict()),
        "columns_missing_percent": convert_numpy_types(null_percent.to_dict()),
        "top_missing_columns": convert_numpy_types(
            null_percent.sort_values(ascending=False).head(10).to_dict()
        ),
        "columns_many_missing": convert_numpy_types(
            (null_percent[null_percent > 0.5]).sort_values(ascending=False).to_dict()
        )
    }

    # duplicates
    try:
        dup_mask = df.duplicated(keep=False)
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_rows_sample = df[df.duplicated(keep=False)].head(5).to_dict(orient="records")
    except Exception:
        duplicate_rows_count = 0
        duplicate_rows_sample = []

    duplicates_summary = {
        "duplicate_count": duplicate_rows_count,
        "duplicate_sample": convert_numpy_types(duplicate_rows_sample),
    }

    # quick column summaries
    columns = {}
    for col in df.columns:
        col_series = df[col]
        non_null = int(col_series.notnull().sum())
        nulls = int(col_series.isnull().sum())
        unique_count = int(col_series.nunique(dropna=True))
        unique_pct = round(unique_count / max(1, total_rows), 4)
        top_values = col_series.value_counts(dropna=True).head(5).to_dict()

        # mixed type detection (sample up to 1000 non-null)
        sample_vals = col_series.dropna().head(1000).tolist()
        types_seen = set(type(v).__name__ for v in sample_vals if v is not None)
        mixed_types = len(types_seen) > 1

        # detect datetime-parseable fraction
        # parsed_dt = pd.to_datetime(col_series, errors="coerce")
        try:
            parsed_dt = pd.to_datetime(
                col_series,
                errors="coerce",
                infer_datetime_format=True,
                dayfirst=True  # nếu file của bạn dùng định dạng ngày-tháng
            )
        except Exception:
            parsed_dt = pd.to_datetime(col_series, errors="coerce")

        dt_parsed_pct = float(parsed_dt.notnull().sum() / max(1, total_rows))

        # detect numeric-parseable fraction
        numeric_parsed = pd.to_numeric(col_series, errors="coerce")
        num_parsed_pct = float(numeric_parsed.notnull().sum() / max(1, total_rows))

        # detect multi-select heuristics (delimiter presence)
        has_separator = False
        sep_candidates = [",", ";", "|", " / "]
        if total_rows > 0:
            contains_sep = col_series.astype(str).str.contains("|".join([s.replace(" ", r"\s*") for s in sep_candidates]), regex=True, na=False)
            has_separator = bool(contains_sep.sum() / max(1, total_rows) > 0.2)

        # numeric stats & outlier detection (IQR)
        numeric_stats = None
        outlier_count = None
        skew = None
        kurt = None
        zero_count = None
        if num_parsed_pct > 0.5:
            num_series = pd.to_numeric(col_series, errors="coerce").dropna()
            if len(num_series) > 0:
                q1 = float(num_series.quantile(0.25))
                q3 = float(num_series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (num_series < lower) | (num_series > upper)
                outlier_count = int(outlier_mask.sum())
                numeric_stats = {
                    "min": convert_numpy_types(num_series.min()),
                    "q1": convert_numpy_types(q1),
                    "median": convert_numpy_types(num_series.median()),
                    "q3": convert_numpy_types(q3),
                    "max": convert_numpy_types(num_series.max()),
                    "mean": convert_numpy_types(num_series.mean()),
                    "std": convert_numpy_types(num_series.std()),
                    "iqr": convert_numpy_types(iqr),
                    "outlier_count": outlier_count,
                }
                try:
                    skew = convert_numpy_types(num_series.skew())
                    kurt = convert_numpy_types(num_series.kurt())
                except Exception:
                    skew = None
                    kurt = None
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

        # entropy (categorical) — lower = more predictable
        entropy = None
        if unique_count > 0 and unique_count < 5000 and top_values:
            counts = list(col_series.value_counts(dropna=True).values)
            probs = [c / sum(counts) for c in counts]
            entropy = float(round(-sum(p * math.log2(p) for p in probs if p > 0), 4))

        # suggested actions (heuristics)
        suggestions = []
        if nulls / max(1, total_rows) > 0.5:
            suggestions.append("High missing rate (>50%): consider drop column or strong imputation.")
        if unique_count == total_rows and total_rows > 1:
            suggestions.append("Looks like a unique identifier (candidate primary key).")
        if unique_count <= 1:
            suggestions.append("Constant column: likely safe to drop.")
        if num_parsed_pct > 0.5 and outlier_count and outlier_count / max(1, total_rows) > 0.01:
            suggestions.append("Numeric column with outliers: consider winsorize or inspect outlier rows.")
        if has_separator:
            suggestions.append("Multi-select / delimiter detected: consider explode into multiple boolean columns.")
        if mixed_types:
            suggestions.append("Mixed types detected (strings & numbers / etc): inspect parsing or coerce consistently.")
        if unique_pct > 0.95 and unique_count > 50 and num_parsed_pct < 0.2:
            suggestions.append("High-cardinality categorical: consider hashing/embedding or keep as text.")

        columns[col] = convert_numpy_types({
            "name": col,
            "non_null": non_null,
            "nulls": nulls,
            "null_percent": round(nulls / max(1, total_rows), 4),
            "unique_count": unique_count,
            "unique_percent": unique_pct,
            "is_constant": unique_count <= 1,
            "is_identifier_candidate": unique_count == total_rows and total_rows > 1,
            "mixed_types_sample": list(types_seen)[:5],
            "inferred_numeric_fraction": round(num_parsed_pct, 4),
            "inferred_datetime_fraction": round(dt_parsed_pct, 4),
            "multi_select_detected": has_separator,
            "top_values": convert_numpy_types(top_values),
            "numeric_stats": numeric_stats,
            "numeric_skew": skew,
            "numeric_kurtosis": kurt,
            "zero_count": zero_count,
            "datetime_stats": dt_stats,
            "entropy": entropy,
            "sample_values": convert_numpy_types(col_series.dropna().head(5).tolist()),
            "suggested_actions": suggestions
        })

    inspection = {
        "head": convert_numpy_types(head),
        "tail": convert_numpy_types(tail),
        "shape": {"rows": int(total_rows), "columns": int(total_cols)},
        "dtypes": dtypes,
        "memory": {"per_column": memory_per_column, "total": memory_total},
        "missing_summary": missing_summary,
        "duplicates": duplicates_summary,
        "columns": columns
    }
    return inspection

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
    df["cluster"] = labels
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=num_cols.columns)
    return {
        "assignments": df[["cluster"]].to_dict(orient="records"),
        "centroids": centers.to_dict(orient="records"),
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

def extract_eda_insights(result: dict) -> list:
    """
    Automatically extract insights from EDA results.
    Input: result (dict) — output from the full AutoEDA pipeline.
    Output: list[str] — list of insights in text format.
    """
    insights = []

    # --- 1. Missing data ---
    missing_summary = result.get("inspection", {}).get("missing_summary", {})
    top_missing = missing_summary.get("top_missing_columns", {})
    for col, pct in top_missing.items():
        if pct > 0.3:  # >30%
            insights.append(f"❗ Column '{col}' has a high missing rate ({pct*100:.1f}%) → consider dropping or imputing.")

    # --- 2. Duplicate rows ---
    duplicates = result.get("inspection", {}).get("duplicates", {})
    dup_count = duplicates.get("duplicate_count", 0)
    if dup_count > 0:
        insights.append(f"🔁 Detected {dup_count} duplicate rows → should be checked and handled.")

    # --- 3. Constant columns ---
    columns = result.get("inspection", {}).get("columns", {})
    for col, col_info in columns.items():
        if col_info.get("is_constant", False):
            insights.append(f"📌 Column '{col}' is constant (only one unique value) → can be removed.")

    # --- 4. High cardinality categorical ---
    for col, col_info in columns.items():
        if col_info.get("inferred_type") in ["categorical", "text"] and col_info.get("unique_percent", 0) > 0.9 and col_info.get("unique_count", 0) > 50:
            insights.append(f"🔢 Column '{col}' has very high cardinality ({col_info.get('unique_count')} unique values) → may represent an ID, should be handled specially.")

    # --- 5. Numeric insights: skew, outliers ---
    descriptive = result.get("descriptive", {})
    numeric_stats = descriptive.get("numeric", {})
    for col, stats in numeric_stats.items():
        skew_val = stats.get("skew", 0)
        if abs(skew_val) > 1:
            direction = "right" if skew_val > 0 else "left"
            insights.append(f"📈 Variable '{col}' is skewed to the {direction} (skew = {skew_val:.2f}) → consider transformation (log, box-cox).")

        outliers = stats.get("outliers", 0)
        total = stats.get("count", 1)
        outlier_pct = outliers / total * 100
        if outlier_pct > 5:
            insights.append(f"⚠️ Variable '{col}' has {outliers} outliers ({outlier_pct:.1f}%) → investigate cause or handle accordingly.")

    # --- 6. Imbalanced categorical ---
    categorical_stats = descriptive.get("categorical", {})
    for col, stats in categorical_stats.items():
        top_values = stats.get("top_values", [])
        if top_values:
            top_val = top_values[0]
            if top_val.get("percent", 0) > 80:
                insights.append(f"⚖️ Column '{col}' is highly imbalanced: '{top_val['value']}' accounts for {top_val['percent']}% → may bias the model.")

    # --- 7. Suggested actions from inspection ---
    for col, col_info in columns.items():
        suggestions = col_info.get("suggested_actions", [])
        for suggestion in suggestions:
            insights.append(f"💡 Suggestion for column '{col}': {suggestion}")

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
            insights.append(f"✅ ANOVA: Categorical variable '{cat}' significantly affects '{num}' (p={p_val:.3f}, η²={eta2:.3f} → {effect} effect size).")

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
            insights.append(f"✅ Chi-square: Significant relationship between '{col1}' and '{col2}' (p={p_val:.3f}, V={cramers_v:.3f} → {strength} association).")

    # --- 9. Clustering quality ---
    clustering = advanced.get("clustering", {})
    sil_score = clustering.get("silhouette_score")
    if sil_score is not None:
        if sil_score > 0.5:
            insights.append(f"🎯 Data can be well clustered (silhouette score = {sil_score:.3f}) → consider KMeans or cluster analysis.")
        elif sil_score < 0.2:
            insights.append(f"📉 Data is difficult to cluster (silhouette score = {sil_score:.3f}) → may lack clear cluster structure.")

    # --- 10. Anomalies (Isolation Forest) ---
    patterns = advanced.get("patterns", {})
    anomalies = patterns.get("anomalies", {})
    outlier_flags = anomalies.get("outlier_flags", [])
    if len(outlier_flags) > 0:
        n_outliers = sum(1 for x in outlier_flags if x == -1)
        pct_outliers = n_outliers / len(outlier_flags) * 100
        if n_outliers > 0:
            insights.append(f"🚨 Detected {n_outliers} anomalies ({pct_outliers:.1f}%) using Isolation Forest → investigate potential fraud or data entry errors.")

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
        insights.append(f"📉 Dataset is too small (only {total_rows} rows) → statistical results may not be reliable.")

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
    # ✅ In sau khi đã có result
    print("All columns:", df.columns.tolist())
    print("Numeric cols:", list(descriptive["numeric"].keys()))
    print("Categorical cols:", list(descriptive["categorical"].keys()))
    print("Unique counts per categorical column:")
    for col in list(descriptive["categorical"].keys()):
        print(f"  {col}: {df[col].nunique()} unique values")
    cleaned = convert_numpy_types(result)

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