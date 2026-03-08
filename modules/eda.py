
import numpy as np
import math
import re
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from dateutil import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def downcast_dtypes(df):
    for col in df.select_dtypes(include = ['int']).columns:

        df[col] = pd.to_numeric(df[col], downcast = 'integer')
    for col in df.select_dtypes(include =['float']).column:
        df[col] = pd.to_numeric(df[col], downcast='float')

def normalize_nulls(series: pd.Series):
    return series.replace(
        ["", " ", "  ", "NA", "N/A", "null", "NULL", "-", "--"],
        np.nan
    )

def infer_date_format(series: pd.Series, top_n: int = 100):
    """Thử suy luận định dạng datetime phổ biến nhất."""
    common_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
        "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"
    ]
    sample = series.dropna().astype(str).head(top_n)
    success_rate = {}
    for fmt in common_formats:
        ok = 0
        for val in sample:
            try:
                pd.to_datetime(val, format=fmt)
                ok += 1
            except Exception:
                pass
        success_rate[fmt] = ok / len(sample) if len(sample) > 0 else 0

    best_fmt = max(success_rate, key=success_rate.get)
    if success_rate[best_fmt] >= 0.5:
        return best_fmt
    return None

# def standardize_dataframe_types(df: pd.DataFrame,
#                                 numeric_threshold: float = 0.8,
#                                 datetime_threshold: float = 0.8):
#     """
#     Enterprise-grade automatic dtype inference & coercion.
#     Returns:
#         cleaned_df
#         detected_schema
#     """

#     df_clean = df.copy()
#     detected_schema = {}

#     for col in df_clean.columns:
#         original = df_clean[col]
#         series = normalize_nulls(original)

#         non_null = series.dropna()
#         total = len(series)

#         # ---------- NUMERIC DETECTION ----------
#         numeric_try = pd.to_numeric(non_null, errors="coerce")
#         numeric_ratio = numeric_try.notna().mean() if len(non_null) > 0 else 0

#         # ---------- DATETIME DETECTION ----------
#         dt_try1 = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
#         dt_try2 = pd.to_datetime(non_null, errors="coerce", dayfirst=False)

#         dt_ratio = max(dt_try1.notna().mean(), dt_try2.notna().mean()) if len(non_null) > 0 else 0

#         # ---------- BOOLEAN DETECTION ----------
#         bool_values = non_null.astype(str).str.lower().isin(["true","false","0","1"])
#         bool_ratio = bool_values.mean() if len(non_null) > 0 else 0

#         # ---------- DECISION TREE ----------
#         if numeric_ratio >= numeric_threshold:
#             df_clean[col] = pd.to_numeric(series, errors="coerce")
#             detected_schema[col] = "numeric"

#         elif dt_ratio >= datetime_threshold:
#             parsed = dt_try1 if dt_try1.notna().mean() >= dt_try2.notna().mean() else dt_try2
#             df_clean[col] = parsed
#             detected_schema[col] = "datetime"

#         elif bool_ratio >= 0.95:
#             df_clean[col] = series.astype(str).str.lower().map(
#                 {"true": True, "1": True, "false": False, "0": False}
#             )
#             detected_schema[col] = "boolean"

#         else:
#             # categorical vs text
#             unique_count = series.nunique()
#             if unique_count < 0.1 * total:
#                 df_clean[col] = series.astype("category")
#                 detected_schema[col] = "categorical"
#             else:
#                 df_clean[col] = series.astype(str)
#                 detected_schema[col] = "string"

#     return df_clean, detected_schema

def robust_datetime_parse(series: pd.Series):
    parsed_dates = []
    ambiguous_count = 0
    success_count = 0

    for val in series:

        if pd.isna(val):
            parsed_dates.append(pd.NaT)
            continue

        val = str(val).strip()
        if val == "":
            parsed_dates.append(pd.NaT)
            continue

        # 🚫 Guard: nếu numeric thuần → không parse datetime
        if re.fullmatch(r"^-?\d+(\.\d+)?$", val):
            parsed_dates.append(pd.NaT)
            continue

        try:
            dt_dayfirst = parser.parse(val, dayfirst=True, fuzzy=False)
            dt_monthfirst = parser.parse(val, dayfirst=False, fuzzy=False)

            if dt_dayfirst != dt_monthfirst:
                ambiguous_count += 1

            parsed_dates.append(dt_dayfirst)
            success_count += 1

        except Exception:
            parsed_dates.append(pd.NaT)

    parsed_series = pd.Series(parsed_dates)

    return parsed_series, {
        "success_ratio": success_count / max(1, len(series)),
        "ambiguous_ratio": ambiguous_count / max(1, len(series))
    }

def standardize_dataframe_types(
    df: pd.DataFrame,
    numeric_threshold: float = 0.9,
    datetime_threshold: float = 0.9,
):

    df = df.copy()

    # ==============================
    # 0️⃣ GLOBAL NULL NORMALIZATION
    # ==============================
    df = df.applymap(
        lambda x: np.nan
        if isinstance(x, str) and x.strip() in ["", "null", "NULL", "n/a", "N/A"]
        else x
    )

    schema = {}
    datetime_reports = {}

    for col in df.columns:

        series = df[col]

        # luôn làm việc trên bản đã dropna
        non_null = series.dropna()

        # ================= NUMERIC =================
        numeric_try = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = (
            numeric_try.notna().mean() if len(non_null) > 0 else 0
        )

        if numeric_ratio >= numeric_threshold:
            df[col] = pd.to_numeric(series, errors="coerce")
            schema[col] = "numeric"
            continue

        # ================= DATETIME =================
        parsed_dt, dt_report = robust_datetime_parse(non_null)
        dt_ratio = (
            parsed_dt.notna().mean() if len(non_null) > 0 else 0
        )

        if dt_ratio >= datetime_threshold:
            df[col] = pd.to_datetime(series, errors="coerce")
            schema[col] = "datetime"
            datetime_reports[col] = dt_report
            continue

        # ================= BOOLEAN =================
        bool_ratio = (
            non_null.astype(str)
            .str.lower()
            .isin(["true", "false", "0", "1"])
            .mean()
            if len(non_null) > 0
            else 0
        )

        if bool_ratio > 0.95:
            df[col] = (
                series.astype(str)
                .str.lower()
                .map({"true": True, "1": True, "false": False, "0": False})
            )
            schema[col] = "boolean"
            continue

        # ================= CATEGORICAL / STRING =================
        unique_ratio = (
            non_null.nunique() / max(1, len(non_null))
        )

        if unique_ratio < 0.1:
            df[col] = series.astype("category")
            schema[col] = "categorical"
        else:
            df[col] = series.astype("string")
            schema[col] = "string"

    return df, {
        "schema": schema,
        "datetime_quality": datetime_reports,
    }

def infer_schema_from_df(
    df: pd.DataFrame,
    type_info: dict | None = None,
    config: dict | None = None
) -> dict:

    type_info = type_info or {}
    schema_map = type_info.get("schema", {})
    datetime_quality = type_info.get("datetime_quality", {})

    props = {}

    for col in df.columns:
        series = df[col]
        dtype = schema_map.get(col, "string")

        unique_count = series.nunique(dropna=True)

        # === NUMERIC ===
        if dtype == "numeric":
            numeric_col = pd.to_numeric(series, errors="coerce")

            prop = {
                "type": "number",
                "minimum": float(numeric_col.min()) if not pd.isna(numeric_col.min()) else None,
                "maximum": float(numeric_col.max()) if not pd.isna(numeric_col.max()) else None,
                "examples": numeric_col.dropna().head(3).tolist()
            }

        # === DATETIME ===
        elif dtype == "datetime":
            dt_col = pd.to_datetime(series, errors="coerce")

            prop = {
                "type": "string",
                "format": "date-time",
                "examples": [str(x) for x in dt_col.dropna().head(3)],
                "min": str(dt_col.min()) if not pd.isna(dt_col.min()) else None,
                "max": str(dt_col.max()) if not pd.isna(dt_col.max()) else None,
                "datetime_quality": type_info["datetime_quality"].get(col)
            }

        # === CATEGORICAL ===
        elif dtype == "categorical":
            enum_values = (
                series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            prop = {
                "type": "string",
                "enum": sorted(enum_values),
                "examples": series.dropna().head(3).astype(str).tolist()
            }

        # === BOOLEAN ===
        elif dtype == "boolean":
            prop = {
                "type": "boolean",
                "examples": series.dropna().head(3).tolist()
            }

        else:
            prop = {
                "type": "string",
                "examples": series.dropna().head(3).tolist(),
                "avg_length": series.astype(str).map(len).mean()
            }

        prop["non_null"] = int(series.notnull().sum())
        prop["nulls"] = int(series.isnull().sum())
        prop["unique_count"] = int(unique_count)

        props[col] = prop

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "EnterpriseGeneratedSchema",
        "type": "object",
        "properties": props
    }

def shannon_entropy(values):
    freq = values.value_counts(normalize=True)
    return -(freq * np.log2(freq + 1e-9)).sum()

def analyze_column(col: pd.Series) -> dict:
    # 1. Chuẩn bị dữ liệu sạch để check
    col = normalize_nulls(col)
    series_dropped = col.dropna()
    series_str = series_dropped.astype(str)
    
    unique_count = col.nunique()
    total = len(col)
    dtype = "text"

    # === BƯỚC 1: Check Dtype gốc của Pandas ===
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        dtype = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(col):
        dtype = "datetime"
    
    # === BƯỚC 2: Nếu chưa ra, thử Parse String ===
    else:
        # 2.1 Check Boolean (True/False strings)
        if series_str.str.lower().isin(['true', 'false', '0', '1']).all() and unique_count <= 2:
             dtype = "boolean"
             
        # 2.2 Check Numeric dạng String (QUAN TRỌNG: Check cái này trước Categorical)
        # Dùng to_numeric với coerce để xem tỷ lệ chuyển đổi thành công
        elif len(series_dropped) > 0:
            numeric_converted = pd.to_numeric(series_dropped, errors='coerce')
            # Nếu > 90% dữ liệu chuyển được sang số -> là Numeric
            numeric_ratio = numeric_converted.notnull().mean()

            if numeric_ratio > 0.6:
                dtype = "numeric"
                col = numeric_converted 

        # 2.3 Check Datetime (như code cũ của bạn)
        if dtype == "text": # Chỉ check nếu chưa phải numeric
            parsed_success = False
            for dayfirst in [False, True]:
                try:
                    dt_parsed = pd.to_datetime(series_dropped, errors="coerce", dayfirst=dayfirst)
                    if dt_parsed.notnull().mean() > 0.8: # Hạ ngưỡng xuống hoặc check kỹ hơn
                        dtype = "datetime"
                        parsed_success = True
                        break
                except Exception:
                    continue

        # 2.4 Check Categorical (Chỉ check sau khi đã chắc chắn không phải số hay ngày)
        if dtype == "text":
            if unique_count <= 50 or (total > 100 and unique_count < 0.05 * total):
                dtype = "categorical"
            elif series_str.str.contains(r"[;,|]").any():
                dtype = "multi-select"

    # === Stats Calculation (Phần này giữ nguyên hoặc tuỳ chỉnh) ===
    stats = {}
    if dtype == "numeric":
        numeric_col = pd.to_numeric(col, errors="coerce")
        stats = {
            "min": convert_numpy_types(numeric_col.min()),
            "max": convert_numpy_types(numeric_col.max()),
            "mean": convert_numpy_types(numeric_col.mean()),
            "std": convert_numpy_types(numeric_col.std()),
        }
    elif dtype == "categorical":
        stats = {
            "unique_values": convert_numpy_types(unique_count),
            "top_values": convert_numpy_types(col.value_counts().head(5).to_dict())
        }
    elif dtype == "datetime":
        dt_col = pd.to_datetime(col, errors="coerce")
        stats = {
            "min_date": str(dt_col.min()) if not pd.isna(dt_col.min()) else None,
            "max_date": str(dt_col.max()) if not pd.isna(dt_col.max()) else None
        }
    else:  # text
        str_lengths = series_dropped.map(len) if len(series_dropped) > 0 else pd.Series([0])
        stats = {
            "avg_length": convert_numpy_types(str_lengths.mean()),
            "sample_values": convert_numpy_types(series_dropped.head(5).tolist())
        }

    return {
        "name": col.name,
        "inferred_type": dtype,
        "non_null": convert_numpy_types(col.notnull().sum()),
        "nulls": convert_numpy_types(col.isnull().sum()),
        "example": convert_numpy_types(series_dropped.head(3).tolist()),
        "stats": stats
    }

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or not np.isfinite(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict(orient="records"))
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    elif obj is pd.NA:
        return None
    elif pd.api.types.is_scalar(obj):
        if pd.isna(obj) or obj == np.inf or obj == -np.inf:
            return None
        return obj
    else:
        return obj

def numeric_advanced_stats(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return None

    mean = series.mean()
    std = series.std()

    z_scores = (series - mean) / (std if std != 0 else 1)

    p1 = float(series.quantile(0.01))
    p99 = float(series.quantile(0.99))

    cv = float(std / mean) if mean != 0 else None

    skew = float(series.skew())
    if abs(skew) < 0.5:
        shape = "approximately_normal"
    elif skew > 1:
        shape = "right_skewed"
    elif skew < -1:
        shape = "left_skewed"
    else:
        shape = "moderately_skewed"

    return {
        "p1": p1,
        "p99": p99,
        "coefficient_of_variation": cv,
        "z_score_outliers_2": int((abs(z_scores) > 2).sum()),
        "z_score_outliers_3": int((abs(z_scores) > 3).sum()),
        "max_abs_z_score": float(abs(z_scores).max()),
        "distribution_shape": shape
    }

def categorical_advanced_stats(series: pd.Series):
    series = series.dropna()
    if len(series) == 0:
        return None

    probs = series.value_counts(normalize=True)
    gini = float(1 - np.sum(probs ** 2))

    rare_ratio = float((probs < 0.01).sum() / len(probs))

    return {
        "gini_impurity": gini,
        "rare_category_ratio": rare_ratio
    }

def datetime_advanced_stats(series: pd.Series):
    dt = pd.to_datetime(series, errors="coerce").dropna()
    if len(dt) == 0:
        return None

    dt_sorted = dt.sort_values()
    gaps = dt_sorted.diff().dropna()

    weekday_dist = dt.dt.weekday.value_counts(normalize=True).to_dict()
    month_dist = dt.dt.month.value_counts(normalize=True).to_dict()

    return {
        "weekday_distribution": weekday_dist,
        "month_distribution": month_dist,
        "max_gap_days": float(gaps.max().days) if not gaps.empty else None,
        "median_gap_days": float(gaps.median().days) if not gaps.empty else None,
        "date_span_days": int((dt.max() - dt.min()).days)
    }

def benford_test(series: pd.Series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[series > 0]
    if len(series) == 0:
        return None

    first_digits = series.astype(str).str[0]
    actual = first_digits.value_counts(normalize=True).to_dict()

    benford_dist = {str(d): math.log10(1 + 1/d) for d in range(1,10)}

    deviation = sum(abs(actual.get(str(d), 0) - benford_dist[str(d)]) for d in range(1,10))

    return {
        "benford_deviation": float(deviation),
        "benford_flag": deviation > 0.15
    }

def detect_regex_pattern(series):
    patterns = {
        "email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
        "phone": r"^\+?\d{8,15}$",
        "uuid": r"^[a-f0-9\-]{36}$",
        "zipcode": r"^\d{4,6}$"
    }

    results = {}
    series = series.dropna().astype(str)

    for name, pattern in patterns.items():
        match_ratio = series.str.match(pattern).mean()
        if match_ratio > 0.8:
            results[name] = float(match_ratio)

    return results

def detect_multicollinearity(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None

    corr_matrix = numeric_df.corr().abs()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })

    return high_corr_pairs

def detect_multicollinearity(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None

    corr_matrix = numeric_df.corr().abs()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": float(corr_matrix.iloc[i, j])
                })

    return high_corr_pairs

def compute_mutual_information(df, target):
    if target not in df.columns:
        return None

    X = df.drop(columns=[target])
    y = df[target]

    numeric_X = X.select_dtypes(include=np.number).fillna(0)

    if len(numeric_X.columns) == 0:
        return None

    if pd.api.types.is_numeric_dtype(y):
        mi = mutual_info_regression(numeric_X, y.fillna(0))
    else:
        y_encoded = LabelEncoder().fit_transform(y.astype(str))
        mi = mutual_info_classif(numeric_X, y_encoded)

    return dict(zip(numeric_X.columns, mi.tolist()))

def inspect_dataset(df: pd.DataFrame, max_sample: int = 10, target: Optional[str] = None) -> dict:
    total_rows, total_cols = df.shape
    head = df.head(max_sample).to_dict(orient="records")
    tail = df.tail(max_sample).to_dict(orient="records")

    # dtypes
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ================= MEMORY =================
    try:
        mem_series = df.memory_usage(deep=True)
        memory_per_column = {col: int(mem_series[col]) for col in mem_series.index}
        memory_total = int(mem_series.sum())
    except Exception:
        memory_per_column = {}
        memory_total = None

    # ================= MISSING =================
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

    # ================= DUPLICATES =================
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
            if num_series.dtype == bool:
                num_series = num_series.astype(int)
                
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

        # -------- ADVANCED BLOCK --------
        advanced_numeric = numeric_advanced_stats(col_series) if num_parsed_pct > 0.5 else None
        benford = benford_test(col_series) if num_parsed_pct > 0.5 else None
        advanced_cat = categorical_advanced_stats(col_series) if 1 < unique_count < 100 else None
        advanced_dt = datetime_advanced_stats(col_series) if dt_parsed_pct > 0.5 else None
        regex_patterns = detect_regex_pattern(col_series)

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
            "suggested_actions": suggestions,
            "advanced_numeric": advanced_numeric,
            "advanced_categorical": advanced_cat,
            "advanced_datetime": advanced_dt,
            "regex_detected": regex_patterns,
            "benford_analysis": benford,
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
        "datetime": {},
        "remarks": []
    }

    # --- Numeric ---
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
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        desc["outliers"] = int(outliers)
        result["numeric"][col] = desc

    # --- Categorical & Datetime ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        # thử parse datetime (cả dayfirst True/False)
        parsed1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
        parsed2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        success_rate1 = parsed1.notna().mean()
        success_rate2 = parsed2.notna().mean()
        if max(success_rate1, success_rate2) > 0.5:
            parsed = parsed1 if success_rate1 >= success_rate2 else parsed2
            parsed = parsed.dropna()
            desc = {
                "count": int(parsed.count()),
                "min": str(parsed.min()),
                "max": str(parsed.max()),
                "range_days": int((parsed.max() - parsed.min()).days),
                # "frequency": freq_label,
                "dayfirst_used": bool(success_rate1 >= success_rate2)
            }
            diffs = parsed.sort_values().diff().dt.days.dropna()
            if not diffs.empty:
                median_gap = diffs.median()
                if median_gap <= 1:
                    desc["frequency"] = "daily"
                elif median_gap <= 7:
                    desc["frequency"] = "weekly"
                elif median_gap <= 31:
                    desc["frequency"] = "monthly"
                else:
                    desc["frequency"] = f"{median_gap:.0f}-day interval"
            result["datetime"][col] = desc
            continue  # dừng, không xử lý tiếp như categorical

        # categorical
        series = series.astype(str)
        freq = series.value_counts()
        total = len(series)
        if series.nunique() > max_categories:
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

#incase using vilm/vinallama-2.7b please change this function name to generate_business_report_template 
def generate_business_report(eda_results: dict) -> str: 
    def safe_float(x, default=0.0):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return default
            return float(x)
        except Exception:
            return default
    sections = []

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
        summary += f"- Dữ liệu thiếu đáng kể (~{pct_missing*100:.1f}%) → cần xử lý trước khi phân tích sâu.\n"
    elif pct_missing > 0:
        summary += f"- Một số giá trị bị thiếu (~{pct_missing*100:.1f}%), nên kiểm tra nguyên nhân.\n"
    else:
        summary += "- Dữ liệu đầy đủ — không có giá trị bị thiếu.\n"

    # 3️⃣ Clustering / Segmentation
    advanced = eda_results.get("advanced", {})
    cluster_info = advanced.get("clustering", {})
    if cluster_info and "centroids" in cluster_info:
        centroids = cluster_info.get("centroids", [])
        n_clusters = len(centroids)
        silhouette = cluster_info.get("silhouette_score", 0)
        summary += f"- Phân tích phân cụm phát hiện {n_clusters} nhóm chính (điểm silhouette = {silhouette:.3f} → cấu trúc rõ ràng).\n"
        for i, c in enumerate(centroids):
            # Lấy đặc trưng nổi bật (giá trị trung tâm)
            top_features = ", ".join(f"{k}≈{round(v,2)}" for k, v in c.items() if isinstance(v, (int, float)))
            summary += f"    Nhóm {i+1}: đặc trưng bởi {top_features}\n"
    else:
        summary += "- Không thực hiện phân cụm hoặc dữ liệu không phù hợp để phân nhóm.\n"

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
    relationships = eda_results.get("relationships", {}).get("mixed", {})

    if sig.get("anova"):
        for key, stats in sig["anova"].items():
            p_val = safe_float(stats.get("p_value"), 1.0)
            eta2 = safe_float(stats.get("eta_squared"), 0.0)
            if p_val < 0.05:
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
                    f"- ✅ Kiểm định ANOVA: Biến phân loại '{cat}' có ảnh hưởng có ý nghĩa thống kê tới '{num}' "
                    f"(p = {p_val:.3f} < 0.05, η² = {eta2:.3f} → mức độ ảnh hưởng {effect_level}).\n"
                    f"   → '{cat}' giải thích được {eta2*100:.1f}% sự biến động của '{num}'.\n"
                    f"- **What** Kết quả phân tích cho thấy yếu tố *{cat}* có tác động rõ rệt đến *{num}*. "
                    f"Trung bình của *{num}* thay đổi đáng kể giữa các nhóm {cat}.\n"
                    f"- **So what**: Điều này chứng minh rằng {cat} là một đòn bẩy quan trọng, "
                    f"có thể giải thích khoảng {eta2*100:.1f}% sự biến động của {num}. "
                    f"Nếu không quản lý tốt yếu tố này, kết quả {num} sẽ biến động khó kiểm soát.\n"

                )

                # 🔍 Liệt kê chi tiết theo nhóm (nếu có dữ liệu từ relationships)
                group_data = relationships.get(f"{cat}__vs__{num}", [])
                if group_data:
                    insights += f"   → Cụ thể theo từng nhóm trong '{cat}':\n"
                    for row in group_data:
                        group_val = row.get(cat, "N/A")
                        mean_val = row.get("mean", 0)
                        count = row.get("count", 0)
                        insights += f"      • Khi '{cat}' = '{group_val}': trung bình '{num}' = {mean_val:.2f} (dựa trên {count} mẫu).\n"
                    # Gợi ý hành động
                    top_group = max(group_data, key=lambda x: x.get("mean", 0))
                    worst_group = min(group_data, key=lambda x: x.get("mean", float('inf')))
                    insights += (
                        f"   → Hành động: Nhóm '{top_group[cat]}' đạt giá trị '{num}' cao nhất — nên nhân rộng yếu tố này. "
                        f"Ngược lại, nhóm '{worst_group[cat]}' cần được điều tra để cải thiện.\n"
                    )
                else:
                    insights += f"   → (Chi tiết theo nhóm chưa được cung cấp — vui lòng đảm bảo 'relationships' được tính trong EDA.)\n"

    # Key Drivers — Chi-square (Categorical vs Categorical)
    relationships_cat = eda_results.get("relationships", {}).get("categorical_vs_categorical", {})
    if sig.get("chi2"):
        for key, stats in sig["chi2"].items():
            p_val = safe_float(stats.get("p_value"), 1.0)
            cramers_v = safe_float(stats.get("cramers_v"), 0.0)

            if p_val < 0.05:
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
                    f"- ✅ Kiểm định Chi-square: Có mối quan hệ có ý nghĩa thống kê giữa '{c1}' và '{c2}' "
                    f"(p = {p_val:.3f} < 0.05, Cramér’s V = {cramers_v:.3f} → mức độ liên hệ {strength}).\n"
                )

                # 🔍 Liệt kê các kết hợp phổ biến
                pair_data = relationships_cat.get(f"{c1}__vs__{c2}", [])
                if pair_data:
                    # Nhóm theo c1
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for row in pair_data:
                        grouped[row[c1]].append((row[c2], row["percentage"]))
                    
                    insights += f"   → Mô hình hành vi cụ thể:\n"
                    for val1, combos in grouped.items():
                        top_combo = max(combos, key=lambda x: x[1])
                        insights += f"      • Khi '{c1}' = '{val1}', thường đi kèm '{c2}' = '{top_combo[0]}' ({top_combo[1]*100:.1f}% trường hợp).\n"
                    insights += f"   → Ý nghĩa: Có thể cá nhân hóa '{c2}' dựa trên '{c1}' — ví dụ: nếu khách hàng ở '{val1}', ưu tiên đề xuất '{top_combo[0]}'.\n"
                else:
                    insights += f"   → (Dữ liệu kết hợp chi tiết chưa được cung cấp — vui lòng đảm bảo 'relationships' được tính trong EDA.)\n"
    sections.append(insights)

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
        if p_val < 0.05 and stats.get("eta_squared", 0) >= 0.06
    ]
    if strong_anova:
        cat, num = strong_anova[0].split("__vs__")
        mid_term_actions.append(f"Tối ưu '{num}' bằng cách điều chỉnh '{cat}' — đã được chứng minh có ảnh hưởng mạnh (η² > 0.06).")

    # Nếu có mối quan hệ Chi-square mạnh
    chi2_results = significance.get("chi2", {})
    strong_chi2 = [
        key for key, stats in chi2_results.items()
        if p_val < 0.05 and stats.get("cramers_v", 0) >= 0.3
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

    appendix = "📑 Appendix / Technical Notes\n"
    appendix += "- Phân tích được thực hiện tự động (Auto EDA + Significance Tests + Clustering).\n"
    appendix += "- Các thuật toán: ANOVA, Chi2-test, Isolation Forest, K-Means, VIF.\n"
    appendix += "- Báo cáo này được tạo tự động nhưng có thể được mở rộng thủ công bởi Data Analyst.\n"
    sections.append(appendix)

    return "\n\n".join(sections)

# def generate_business_report(eda_results: dict) -> str:
#    if LLM_READY:
#     logger.info("Generate report using AI")
#     return reportAI(eda_results,"full")
#    else:
#         logger.warning("LLM not available, falling back to template report")
#         return generate_business_report_template(eda_results)

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

# LLM_Model_name = "vilm/vinallama-2.7b"
# Device = "cuda" if torch.cuda.is_available() else "cpu"
# print (f"using device: {Device}")
# try:
#     tokenizer = AutoTokenizer.from_pretrained(LLM_Model_name, trust_remote_code =True)
#     model = AutoModelForCausalLM.from_pretrained(
#         LLM_Model_name,
#         torch_dtype = torch.float16 if Device == "cuda" else torch.float32,
#         low_cpu_mem_usage=True,
#         trust_remote_code = True
#     ).to(Device)
#     model.eval()
#     LLM_READY = True
#     logger.info(f"Model loaded successfully")

# except Exception as e:
#     logger.error(f"Failed to load model:{e}")
#     LLM_READY =False

# def reportAI(eda_results:dict, section:str = "full"):
#     if not LLM_READY :
#         return "[Model did not load -  use default report]"
#     rows = eda_results.get("inspection", {}).get("shape",{}).get("rows",0)
#     cols = eda_results.get("inspection",{}).get("shape",{}).get("columns",0)

#     raw_insights = extract_eda_insights(eda_results)
#     insights_text = "\n".join(f"- {insight}" for insight in raw_insights)

#     # Tạo prompt tùy theo section
#     if section == "executive":
#         prompt = f"""Bạn là chuyên gia phân tích dữ liệu với 15 năm kinh nghiệm.
# Dựa trên kết quả phân tích dữ liệu sau, hãy viết một bản TÓM TẮT ĐIỀU HÀNH (Executive Summary) bằng tiếng Việt, ngắn gọn, súc tích, tập trung vào điểm then chốt cho ban lãnh đạo.

# Thông tin:
# - Dataset có {rows:,} dòng và {cols} cột.
# - Các insight chính:
# {insights_text}

# Yêu cầu:
# - Viết dưới 150 từ.
# - Dùng ngôn ngữ kinh doanh, không dùng thuật ngữ kỹ thuật.
# - Nêu bật rủi ro, cơ hội và khuyến nghị hành động cấp cao.

# Bản tóm tắt:"""

#     elif section == "insights":
#         prompt = f"""Bạn là chuyên gia phân tích dữ liệu.
# Dựa trên các insight sau, hãy viết lại chúng thành một phần "PHÂN TÍCH CHIẾN LƯỢC" mạch lạc, có cấu trúc, bằng tiếng Việt tự nhiên:

# {insights_text}

# Yêu cầu:
# - Nhóm các insight liên quan lại với nhau.
# - Giải thích ý nghĩa kinh doanh của từng insight.
# - Dùng biểu tượng cảm xúc (emoji) để làm nổi bật điểm quan trọng.
# - Viết như đang trình bày cho CEO nghe.

# Phân tích chiến lược:"""

#     elif section == "recommendations":
#         prompt = f"""Bạn là Cố vấn chiến lược.
# Dựa trên các insight sau, hãy đề xuất 3-5 KHUYẾN NGHỊ HÀNH ĐỘNG cụ thể, có thể thực thi, chia theo ngắn hạn (0-3 tháng), trung hạn (3-12 tháng), dài hạn (1-3 năm):

# {insights_text}

# Yêu cầu:
# - Mỗi khuyến nghị phải có: (1) Hành động cụ thể, (2) Bộ phận chịu trách nhiệm, (3) Kỳ vọng kết quả.
# - Ưu tiên tính khả thi và ROI.
# - Dùng ngôn ngữ mệnh lệnh, rõ ràng.

# Khuyến nghị hành động:"""

#     else:  
#         prompt = f"""Bạn là Trưởng phòng Phân tích Dữ liệu.
# Hãy viết một BÁO CÁO PHÂN TÍCH TOÀN DIỆN bằng tiếng Việt dựa trên dữ liệu sau:

# THÔNG TIN CHUNG:
# - Kích thước dataset: {rows:,} dòng × {cols} cột
# - Các insight chính:
# {insights_text}

# YÊU CẦU BÁO CÁO:
# 1. Executive Summary (Tóm tắt điều hành): 3-5 gạch đầu dòng chính.
# 2. Phân tích chiến lược: Nhóm insight, giải thích ý nghĩa kinh doanh.
# 3. Khuyến nghị hành động: Chia theo ngắn/trung/dài hạn, có tính khả thi.
# 4. Rủi ro & Cơ hội: Liệt kê và đề xuất cách xử lý.
# 5. Phụ lục: Ghi chú kỹ thuật ngắn gọn.

# VIẾT BẰNG NGÔN NGỮ KINH DOANH, KHÔNG DÙNG THUẬT NGỮ KỸ THUẬT. SỬ DỤNG EMOJI ĐỂ LÀM NỔI BẬT Ý CHÍNH.

# BÁO CÁO:"""

#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(Device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=1024,
#                 temperature=0.7,
#                 top_p=0.9,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id
#             )

#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Lấy phần sau prompt (vì model trả về cả prompt + response)
#         if response.startswith(prompt):
#             response = response[len(prompt):].strip()
        
#         # Làm sạch response
#         response = response.replace("BÁO CÁO:", "").replace("Executive Summary:", "").strip()
#         return response

#     except Exception as e:
#         logger.error(f"LLM generation failed: {e}")
#         return f"[Lỗi khi sinh báo cáo bằng LLM: {str(e)}]"