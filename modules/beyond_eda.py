# modules/beyond_eda.py
import json
import os
from typing import Any, Dict, List
from scipy.stats import entropy
import math
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from modules.eda import infer_schema_from_df , descriptive_statistics
from scipy.stats import ttest_ind
from patsy import dmatrices
from modules.prediction import auto_detect_target, detect_data_types, preprocess_data
from collections import defaultdict
from pprint import pprint
from sklearn.linear_model import LassoCV
from scipy.stats import spearmanr, kruskal, mannwhitneyu, pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "pipelineAutoML.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    pipeline_config = json.load(f)

REQUIREMENTS = {
    "psm": {
        "treatment_col": "binary (2 unique values, e.g. 0/1 or A/B)",
        "outcome_col": "numeric (continuous outcome variable)",
        "covariates": "list of numeric/categorical covariates"
    },
    "did": {
        "group_col": "binary or categorical (treatment vs control groups)",
        "time_col": "datetime or ordered numeric (time periods)",
        "outcome_col": "numeric (continuous outcome variable)"
    },
    "iv": {
        "instrument_col": "instrument variable (correlated with treatment, not directly with outcome)",
        "treatment_col": "binary/numeric treatment",
        "outcome_col": "numeric outcome"
    },
    "robustness": {
    "treatment_col": "binary (2 unique values, e.g. 0/1 or A/B)",
    "outcome_col": "numeric (continuous outcome variable)"
    },
    "cohort": {
        "user_col": "user identifier (string or int, unique per entity)",
        "date_col": "datetime (signup/purchase date)"
    },
    "survival": {
        "duration_col": "numeric (time-to-event in days/weeks/etc.)",
        "event_col": "binary (1 = event occurred, 0 = censored)"
    },
    "nonparam": {
        "group_col": "categorical (groups to compare, 2+ levels)",
        "outcome_col": "numeric (values to compare across groups)"
    },
    "effect": {
        "effect_col": "numeric effect size (e.g. ATT, ATE)",
        "base_col": "numeric baseline (e.g. revenue, population)"
    },
    "counterintuitive": {
        "metric_col": "numeric (KPI to check sign)",
        "value_col": "numeric (observed value)",
        "expected_sign_col": "categorical or numeric (+1/-1 expectation of sign)"
    }
}

# ----------------- Helpers -----------------
def with_hint(error_msg, candidates=None):
    return {"error": error_msg, "hint": candidates or []}

def safe_head(df, n=3):
    try:
        return df.head(n).to_dict(orient="records")
    except Exception:
        return []

def mean_ignorena(s):
    try:
        return float(np.nanmean(s))
    except Exception:
        return None

# ----------------- Auto-detect config -----------------
def _entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True, dropna=True)
    return -(probs * np.log2(probs + 1e-9)).sum()

def detect_outcome(df: pd.DataFrame, numeric_cols: list, top_k: int = 1) -> str:
    if not numeric_cols:
        return None

    n = len(df)

    # loại bỏ cột giống ID (unique gần bằng số dòng)
    candidates = [c for c in numeric_cols if df[c].nunique() < 0.9 * n]

    if not candidates:
        candidates = numeric_cols

    # Tính score theo variance + skewness
    scores = {}
    for c in candidates:
        var = df[c].var(skipna=True)
        skew = df[c].skew(skipna=True)
        # heuristic: variance quan trọng hơn, skewness dương được cộng thêm
        score = var + (abs(skew) * 0.1)
        scores[c] = score

    # Sắp xếp theo score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k == 1:
        return ranked[0][0]
    return [c for c, _ in ranked[:top_k]]

def score_outcome_candidate(df: pd.DataFrame, col: str) -> float:
    series = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(series) or len(series) < 10:
        return 0.0

    n = len(series)
    nunq = series.nunique()

    # 1. Variance score (chuẩn hóa về [0,1])
    var_raw = series.var()
    var_score = min(var_raw / (series.max() - series.min() + 1e-9), 1.0) if series.max() != series.min() else 0.0

    # 2. Skewness score
    try:
        skew_raw = abs(series.skew())
        skew_score = min(skew_raw / 3.0, 1.0)  # cap at 3
    except:
        skew_score = 0.0

    # 3. Cardinality ratio
    nunq_ratio = nunq / n

    # 4. Distribution entropy (10 bins)
    hist, _ = np.histogram(series, bins=10)
    probs = hist / (hist.sum() + 1e-9)
    hist_entropy = entropy(probs + 1e-9) / np.log(10)  # normalize to [0,1]

    # 5. Correlation penalty (nếu correlated với nhiều numeric → có thể là covariate)
    other_nums = [c for c in df.select_dtypes(include=[np.number]).columns if c != col]
    corr_penalty = 0.0
    if other_nums:
        corr_with_others = df[other_nums].corrwith(series).abs().mean()
        corr_penalty = corr_with_others

    # 6. Logic penalty: dựa trên entropy + cardinality
    # → Nếu entropy thấp và cardinality thấp → khả năng cao là numeric logic (age, BMI...)
    logic_penalty = (1 - hist_entropy) * 0.5 + (1 - nunq_ratio) * 0.5

    # 7. Tổng hợp score
    score = (
        0.5 * var_score +
        0.3 * skew_score +
        0.2 * nunq_ratio +
        0.1 * hist_entropy -
        0.1 * corr_penalty -
        0.1 * logic_penalty
    )

    return max(0.0, min(1.0, score))

def score_treatment_candidate(df: pd.DataFrame, col: str, outcome_col: str = None) -> float:
    series = df[col].dropna()
    
    # Phải là binary hoặc low-cardinality categorical
    if series.nunique() > 5:
        return 0.0
    
    # Tính balance (càng gần 0.5 càng tốt)
    vc = series.value_counts(normalize=True)
    if len(vc) < 2:
        return 0.0
    balance = 1 - abs(vc.iloc[0] - 0.5) / 0.5

    # Correlation với outcome (nếu có)
    corr_score = 0.0
    if outcome_col and outcome_col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[outcome_col]):
                corr = abs(series.astype('category').cat.codes.corr(df[outcome_col]))
            else:
                # Chi-square
                from scipy.stats import chi2_contingency
                table = pd.crosstab(series, df[outcome_col])
                chi2, p, _, _ = chi2_contingency(table)
                corr_score = 1 - p  # p càng nhỏ → mối quan hệ càng mạnh
        except:
            corr_score = 0.0

    score = 0.6 * balance + 0.4 * corr_score
    return max(0.0, min(1.0, score))

def score_time_candidate(df: pd.DataFrame, col: str) -> float:
    series = df[col].dropna()
    
    # Nếu column là numeric → bỏ qua
    if pd.api.types.is_numeric_dtype(series):
        ratio_ordered = ((series.sort_values().diff() >= 0).mean())
        if ratio_ordered > 0.9:
            return 0.8
    
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        valid_ratio = parsed.notna().mean()
        if valid_ratio < 0.8:
            return 0.0  
        
        # Kiểm tra range hợp lý
        min_date = parsed.min()
        max_date = parsed.max()
        if min_date.year < 1900 or max_date.year > 2100:
            return 0.3  
        
        return valid_ratio  # score [0.8–1.0]
    except:
        return 0.0

def score_instrument_candidate(df: pd.DataFrame, col: str, treatment_col: str, outcome_col: str) -> float:
    if not (treatment_col and outcome_col):
        return 0.0
    
    try:
        # Correlation với treatment
        corr_t = abs(df[col].corr(df[treatment_col]))
        # Correlation với outcome
        corr_o = abs(df[col].corr(df[outcome_col]))
        
        # Instrument: corr_t cao, corr_o thấp
        score = corr_t * (1 - corr_o)
        return max(0.0, min(1.0, score))
    except:
        return 0.0

def score_group_candidate(df: pd.DataFrame, col: str, outcome_col: str = None) -> float:
    if outcome_col is None:
        return 0.0

    series = df[col].dropna()
    nunq = series.nunique()
    
    # Medium cardinality: 2 <= nunq <= 20
    if nunq < 2 or nunq > 20:
        return 0.0

    # High entropy → good group
    try:
        freq = series.value_counts(normalize=True)
        entropy = -(freq * np.log2(freq + 1e-9)).sum()
        entropy_score = min(entropy / np.log2(nunq), 1.0)
    except:
        entropy_score = 0.0

    # Strong relationship with outcome → better
    relation_score = 0.0
    try:
        if pd.api.types.is_numeric_dtype(df[outcome_col]):
            # ANOVA
            groups = [df[df[col] == cat][outcome_col].dropna().values for cat in series.unique()]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                from scipy.stats import f_oneway
                f, p = f_oneway(*groups)
                relation_score = 1 - p  # p nhỏ → quan hệ mạnh
        else:
            # Chi-square
            from scipy.stats import chi2_contingency
            table = pd.crosstab(series, df[outcome_col])
            chi2, p, _, _ = chi2_contingency(table)
            relation_score = 1 - p
    except:
        relation_score = 0.0

    score = 0.5 * entropy_score + 0.5 * relation_score
    return max(0.0, min(1.0, score))

def score_id_candidate(df: pd.DataFrame, col: str) -> float:
    series = df[col].dropna()
    n = len(df)
    
    # 1. Tỷ lệ unique
    unique_ratio = series.nunique() / n
    uniqueness_score = min(unique_ratio, 1.0)
    
    # 2. Entropy phân phối
    try:
        freq = series.value_counts(normalize=True)
        entropy_score = -(freq * np.log2(freq + 1e-9)).sum()
        entropy_score /= np.log2(max(1, series.nunique()))
        entropy_score = min(entropy_score, 1.0)
    except:
        entropy_score = 0.0
    
    # 3. Numeric logic penalty dựa trên đặc tính dữ liệu (variance thấp, skew thấp)
    logic_penalty = 0.0
    if pd.api.types.is_numeric_dtype(series):
        var_norm = series.var() / (series.max() - series.min() + 1e-9)
        skew_abs = abs(series.skew())
        if var_norm < 0.01 and skew_abs < 0.3:
            logic_penalty = 0.3
    
    # 4. Combine
    score = 0.7 * uniqueness_score + 0.3 * entropy_score - 0.2 * logic_penalty
    return max(0.0, min(1.0, score))

def infer_variable_role(df: pd.DataFrame, col: str, outcome_col: str = None, treatment_col: str = None) -> dict:
    roles = ["outcome", "treatment", "id", "time", "instrument", "group"]
    scores = {}

    # Tính score cho từng vai trò
    scores["outcome"] = score_outcome_candidate(df, col)
    scores["treatment"] = score_treatment_candidate(df, col, outcome_col)
    scores["id"] = score_id_candidate(df, col)
    scores["time"] = score_time_candidate(df, col)  # chỉ object/string mới được score
    scores["instrument"] = score_instrument_candidate(df, col, treatment_col, outcome_col)
    scores["group"] = score_group_candidate(df, col, outcome_col)

    # Chọn vai trò có score cao nhất
    best_role = max(scores, key=scores.get)
    best_score = scores[best_role]

    # Confidence
    if best_score < 0.5:
        confidence = "very_low"
        suggestion = f"Column '{col}' không rõ vai trò — cần kiểm tra thủ công."
    elif best_score < 0.7:
        confidence = "low"
        suggestion = f"Column '{col}' có thể là '{best_role}' nhưng cần xác nhận."
    else:
        confidence = "high"
        suggestion = f"Column '{col}' rất có thể là '{best_role}'."

    return {
        "column": col,
        "role": best_role,
        "confidence_score": best_score,
        "confidence": confidence,
        "suggestion": suggestion,
        "all_scores": scores
    }

def detect_user_id(df: pd.DataFrame, outcome_col=None, treatment_col=None, time_col=None) -> str:
    cols = df.columns
    n = len(df)

    candidate_scores = {}
    for c in cols:
        nunq = df[c].nunique()
        nunq_ratio = nunq / n

        # Skip obvious non-ID
        if c in [outcome_col, treatment_col, time_col]:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if nunq < 0.5 * n:  # ID phải có nhiều giá trị gần bằng số dòng
            continue

        # Heuristic scoring
        entropy_score = _entropy(df[c]) / np.log2(max(2, nunq))
        # ID tốt: high unique_ratio, low entropy (mỗi user lặp lại không nhiều)
        score = 0.7 * nunq_ratio + 0.3 * (1 - entropy_score)
        candidate_scores[c] = score

    if not candidate_scores:
        return None

    return max(candidate_scores, key=candidate_scores.get)

def auto_detect_config(df: pd.DataFrame) -> dict:
    df = df.copy()
    cols = df.columns.tolist()

    # --- Infer schema ---
    schema = infer_schema_from_df(df)

    numeric_cols = [c for c, p in schema["properties"].items() if p["type"] in ["integer", "number"]]
    bool_cols = [c for c, p in schema["properties"].items() if p["type"] == "boolean" 
                 or (p["type"]=="string" and "enum" in p and len(p["enum"])==2)]
    datetime_cols = [c for c, p in schema["properties"].items() if p.get("format") == "date-time"]
    cat_cols = [c for c, p in schema["properties"].items() if p["type"] == "string"]

    n = len(df)

    # --- Parse datetime columns ---
    for c in datetime_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # --- Outcome ---
    outcome_col = detect_outcome(df, numeric_cols, top_k=1)

    # --- Treatment ---
    treatment_col = None
    if bool_cols:
        scores = {}
        for c in bool_cols:
            vc = df[c].value_counts(normalize=True)
            balance = 1 - abs(vc.iloc[0]-0.5)/0.5 if len(vc) >= 2 else 0
            corr = 0
            if outcome_col:
                try:
                    codes = pd.Categorical(df[c]).codes
                    corr = abs(pd.Series(codes).corr(df[outcome_col]))
                except:
                    pass
            scores[c] = 0.65 * balance + 0.35 * corr
        if scores:
            treatment_col = max(scores, key=scores.get)

    # --- Encode treatment_col if needed ---
    encoded_cols = {}
    if treatment_col and not pd.api.types.is_numeric_dtype(df[treatment_col]):
        df[treatment_col + "_enc"] = pd.Categorical(df[treatment_col]).codes
        treatment_col = treatment_col + "_enc"
        encoded_cols[treatment_col] = df[treatment_col]

    # --- Group ---
    group_col = None
    candidate_groups = [c for c in cat_cols if 2 <= df[c].nunique() <= 20]
    if candidate_groups:
        # chọn entropy cao nhất
        entropies = {c: _entropy(df[c]) for c in candidate_groups}
        group_col = max(entropies, key=entropies.get)
        # Encode
        df[group_col + "_enc"] = pd.Categorical(df[group_col]).codes
        group_col = group_col + "_enc"
        encoded_cols[group_col] = df[group_col]
    else:
        # fallback: tạo cột group giả lập 50/50
        df["group_col_auto"] = np.random.choice([0,1], size=n)
        group_col = "group_col_auto"
        encoded_cols[group_col] = df[group_col]

    # --- Instrument ---
    instrument_col = None
    if treatment_col:
        possible_inst = [c for c in cols if c not in [treatment_col, outcome_col] and df[c].nunique() >= 2]
        scores_inst = {}
        for c in possible_inst:
            codes = pd.Categorical(df[c]).codes
            corr_t = abs(pd.Series(codes).corr(df[treatment_col]))
            corr_o = abs(pd.Series(codes).corr(df[outcome_col])) if outcome_col else 0
            score = corr_t - corr_o
            if score > 0.05:
                scores_inst[c] = score
        if scores_inst:
            instrument_col = max(scores_inst, key=scores_inst.get)
            df[instrument_col + "_enc"] = pd.Categorical(df[instrument_col]).codes
            instrument_col = instrument_col + "_enc"
            encoded_cols[instrument_col] = df[instrument_col]

    # --- Time (cohort date_col) ---
    if datetime_cols:
        time_col = datetime_cols[0]
    else:
        # fallback: chọn duration_col numeric tăng dần
        dur_scores = {c: abs(df[c].dropna().skew()) for c in numeric_cols if len(df[c].dropna()) > 5 and df[c].min() >= 0}
        if dur_scores:
            time_col = max(dur_scores, key=dur_scores.get)
        else:
            # fallback: tạo cột giả lập tăng dần
            df["time_col_auto"] = np.arange(n)
            time_col = "time_col_auto"
            encoded_cols[time_col] = df[time_col]

    # --- User ID ---
    id_scores = {}
    for c in cols:
        nunq_ratio = df[c].nunique() / max(1, len(df))
        entropy_score = _entropy(df[c]) / np.log2(max(2, df[c].nunique()))
        id_scores[c] = nunq_ratio * 0.7 + entropy_score * 0.3
    user_col = detect_user_id(df, outcome_col=outcome_col, treatment_col=treatment_col, time_col=time_col)

    # --- Duration / Event ---
    duration_col = None
    dur_scores = {c: abs(df[c].dropna().skew()) for c in numeric_cols if len(df[c].dropna()) > 5 and df[c].min() >= 0}
    if dur_scores:
        duration_col = max(dur_scores, key=dur_scores.get)

    event_col = None
    if bool_cols:
        rare_scores = {c: 1 - (df[c].value_counts(normalize=True).min() * 2) for c in bool_cols}
        if rare_scores:
            event_col = max(rare_scores, key=rare_scores.get)

    # --- Covariates ---
    used = {outcome_col, treatment_col, user_col, duration_col, event_col, group_col, instrument_col, time_col}
    covariates = [c for c in numeric_cols if c not in used and c is not None]

    # --- Build config ---
    algo_config = {
        "cohort": {
            "user_col": user_col,
            "date_col": time_col,
            "group_col": group_col
        },
        "psm": {
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "covariates": covariates
        },
        "iv": {
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "instrument_col": instrument_col,
            "covariates": covariates
        },
        "did": {
            "group_col": group_col,
            "time_col": time_col,
            "outcome_col": outcome_col,
            "treatment_col": treatment_col
        }
    }

    return {"algo_config": algo_config, "encoded_cols": encoded_cols}

def explain_choice(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
    explanations = {}
    
    # Outcome
    if cfg.get("outcome_candidates"):
        best, score = cfg["outcome_candidates"][0]
        explanations["outcome"] = f"{best} chosen as outcome because it had highest variance/correlation score ({score:.3f})."
    else:
        explanations["outcome"] = "No numeric column strong enough for outcome."
    
    # Treatment
    if cfg.get("treatment_candidates"):
        best, score = cfg["treatment_candidates"][0]
        explanations["treatment"] = f"{best} chosen as treatment because it's binary and balance/outcome correlation score={score:.3f}."
    else:
        explanations["treatment"] = "No suitable binary treatment found."
    
    # Duration
    if cfg.get("duration_col"):
        explanations["duration"] = f"{cfg['duration_col']} chosen as duration because it's non-negative and skewed (typical for time-to-event)."
    else:
        explanations["duration"] = "No duration-like column found."
    
    # Event
    if cfg.get("event_col"):
        explanations["event"] = f"{cfg['event_col']} chosen as event because it's rare binary and correlates with duration."
    else:
        explanations["event"] = "No rare binary event column found."
    
    # Group
    if cfg.get("group_col"):
        explanations["group"] = f"{cfg['group_col']} chosen as group because it has 2–20 categories with decent entropy."
    else:
        explanations["group"] = "No categorical group detected."
    
    # Instrument
    if cfg.get("instrument_col"):
        explanations["instrument"] = f"{cfg['instrument_col']} chosen as instrument because it's correlated with treatment but not outcome."
    else:
        explanations["instrument"] = "No valid instrument candidate found."
    
    # Effect & Base
    if cfg.get("effect_col"):
        explanations["effect"] = f"{cfg['effect_col']} picked as effect proxy (highest variance numeric)."
    else:
        explanations["effect"] = "No effect proxy identified."
    if cfg.get("base_col"):
        explanations["base"] = f"{cfg['base_col']} picked as base proxy (largest total numeric)."
    else:
        explanations["base"] = "No base proxy identified."
    
    return explanations

# ----------------- Stubs & Safe Implementations -----------------
def cohort_retention(df: pd.DataFrame, user_col: str, date_col: str, period='M'):
    if user_col not in df.columns:
        raise ValueError(f"user_col '{user_col}' không tồn tại trong dataframe.")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' không tồn tại trong dataframe.")

    tmp = df[[user_col, date_col]].dropna().copy()

    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="raise")  

    # Tính cohort
    tmp["cohort_period"] = tmp[date_col].dt.to_period(period).dt.to_timestamp()
    tmp["activity_period"] = tmp[date_col].dt.to_period(period).dt.to_timestamp()

    first = tmp.groupby(user_col)["cohort_period"].min().reset_index().rename(columns={"cohort_period":"cohort"})
    tmp = tmp.merge(first, on=user_col, how="left")

    # Số kỳ (period_number)
    if period=='M':
        tmp["period_number"] = ((tmp["activity_period"].dt.year - tmp["cohort"].dt.year) * 12 +
                                (tmp["activity_period"].dt.month - tmp["cohort"].dt.month))
    else:
        tmp["period_number"] = ((tmp["activity_period"] - tmp["cohort"]) / np.timedelta64(1, 'D')).astype(int)

    # Tạo pivot table
    cohort_counts = tmp.groupby(["cohort", "period_number"])[user_col].nunique().reset_index()
    pivot = cohort_counts.pivot_table(index="cohort", columns="period_number", values=user_col, fill_value=0)

    size = pivot.iloc[:, 0] if pivot.shape[1] > 0 else pd.Series()
    retention = pivot.divide(size, axis=0).fillna(0)

    return {"cohort_matrix": pivot.round(0).to_dict(), "retention_matrix": retention.round(3).to_dict()}

def survival_km_cox(df: pd.DataFrame, duration_col: str, event_col: str, covariates=None):
    if duration_col not in df.columns or event_col not in df.columns:
        return with_hint("missing duration_col or event_col for survival", candidates=[duration_col, event_col])
    try:
        tmp = df[[duration_col, event_col] + (covariates or [])].dropna()
        tmp[duration_col] = pd.to_numeric(tmp[duration_col], errors="coerce")
        if LIFELINES_AVAILABLE:
            kmf = KaplanMeierFitter()
            kmf.fit(tmp[duration_col], event_observed=tmp[event_col])
            km = kmf.survival_function_.reset_index().rename(columns={"KM_estimate":"survival"})
            cph_summary = None
            if covariates:
                cph = CoxPHFitter()
                cph.fit(tmp.rename(columns={duration_col:"T", event_col:"E"}), duration_col="T", event_col="E")
                cph_summary = cph.summary.reset_index().to_dict(orient="records")
            return {"kaplan_meier": km.to_dict(orient="records"), "cox_summary": cph_summary}
        else:
            # naive KM estimator fallback
            df_sorted = tmp.sort_values(duration_col)
            at_risk = len(df_sorted)
            cum_surv = 1.0
            surv = []
            for t, g in df_sorted.groupby(duration_col):
                d = g[event_col].sum()
                n = at_risk
                if n > 0:
                    cum_surv *= (1 - d / n)
                    surv.append({"time": float(t), "survival": float(cum_surv)})
                at_risk -= g.shape[0]
            return {"kaplan_meier": surv, "cox_summary": None}
    except Exception as e:
        return {"error": str(e)}

def nonparametric_tests(df: pd.DataFrame, group_col: str, value_col: str, alternative="two-sided"):

    if group_col not in df.columns or value_col not in df.columns:
        return with_hint("missing group_col or value_col for nonparametric tests", candidates=[group_col, value_col])
    try:
        series = df[[group_col, value_col]].dropna()
        groups = [g[value_col].values for _, g in series.groupby(group_col)]
        res = {}
        try:
            if pd.api.types.is_numeric_dtype(series[group_col]):
                rho, p_rho = spearmanr(series[group_col], series[value_col])
                res["spearman"] = {"rho": float(rho), "p": float(p_rho)}
            else:
                res["spearman"] = None
        except Exception:
            res["spearman"] = None
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                res["kruskal"] = {"stat": float(stat), "p": float(p)}
            except Exception as e:
                res["kruskal"] = {"error": str(e)}
        if len(groups) == 2:
            try:
                u, p = mannwhitneyu(groups[0], groups[1], alternative=alternative)
                res["mannwhitney"] = {"u": float(u), "p": float(p)}
            except Exception as e:
                res["mannwhitney"] = {"error": str(e)}
        res["group_medians"] = series.groupby(group_col)[value_col].median().to_dict()
        return res
    except Exception as e:
        return {"error": str(e)}

def detect_effect_base(df: pd.DataFrame, min_corr=0.1, top_n=3):
    df_numeric = df.select_dtypes(include=[np.number])
    df_all = df.copy()
    base_candidates = []
    effect_candidates = []

    # 1️⃣ Base candidates
    for col in df_numeric.columns:
        series = df_numeric[col].dropna()
        if len(series) < 2:
            continue
        mean = series.mean()
        std = series.std()
        skew = series.skew()
        confidence = min(1.0, (abs(skew) / 5) + 0.5)
        base_candidates.append({
            "col": col,
            "mean": mean,
            "std": std,
            "skew": skew,
            "confidence": round(confidence, 2),
            "reason": "numeric, scale lớn, skewed"
        })

    base_candidates = sorted(base_candidates, key=lambda x: x["confidence"], reverse=True)[:top_n]

    # 2️⃣ Effect candidates + Pearson R
    for base in base_candidates:
        base_col = base["col"]
        for col in df_all.columns:
            if col == base_col:
                continue
            series = df_all[col].dropna()
            if len(series) < 2:
                continue

            corr = None
            conf_corr = 0.0

            if pd.api.types.is_numeric_dtype(series):
                try:
                    corr, _ = pearsonr(series, df_all[base_col])
                    conf_corr = abs(corr)
                except Exception:
                    corr = None
                    conf_corr = 0.1
            else:
                try:
                    series_encoded = pd.factorize(series)[0]
                    mi = mutual_info_classif(
                        series_encoded.reshape(-1,1), df_all[base_col], discrete_features=True
                    )
                    conf_corr = float(mi / (mi.max() + 1e-9))
                    corr = None
                except Exception:
                    corr = None
                    conf_corr = 0.1

            if conf_corr >= min_corr:
                effect_candidates.append({
                    "col": col,
                    "base_col": base_col,
                    "confidence": round(conf_corr, 2),
                    "correlation": round(corr, 4) if corr is not None else None,
                    "reason": "auto-detect via correlation/mutual info"
                })

    if len(effect_candidates) == 0:
        for base in base_candidates:
            for col in df_all.columns:
                if col != base["col"]:
                    effect_candidates.append({
                        "col": col,
                        "base_col": base["col"],
                        "confidence": 0.2,
                        "correlation": None,
                        "reason": "fallback heuristics"
                    })

    effect_candidates = sorted(effect_candidates, key=lambda x: x["confidence"], reverse=True)[:top_n]

    return {
        "base_candidates": base_candidates,
        "effect_candidates": effect_candidates,
        "significant_pairs": effect_candidates
    }

def effect_to_business(df: pd.DataFrame, effect_col: str = None, base_col: str = None, scale=1.0):
    is_auto_detected = not effect_col or not base_col or effect_col not in df.columns or base_col not in df.columns
    candidates = None

    # --- Auto detect if needed ---
    if is_auto_detected:
        candidates = detect_effect_base(df)
        pairs = candidates.get("significant_pairs", [])
        if len(pairs) > 0:
            effect_col = pairs[0]["col"]
            base_col = pairs[0]["base_col"]
        else:
            return {
                "error": "Không tìm thấy cặp biến có tương quan mạnh & ý nghĩa thống kê.",
                "all_candidates": candidates
            }

    try:
        tmp = df[[effect_col, base_col]].dropna().copy()
        tmp["absolute_impact"] = tmp[base_col] * tmp[effect_col] * scale
        tmp["new_value"] = tmp[base_col] + tmp["absolute_impact"]
        tmp["roi_estimate"] = tmp["absolute_impact"] / tmp[base_col].replace({0: pd.NA})

        summary = {
            "total_impact": float(tmp["absolute_impact"].sum()),
            "mean_roi": float(tmp["roi_estimate"].mean())
        }

        # --- Pearson for main pair ---
        pearson_r, p_value = None, None
        try:
            if len(tmp) >= 2 and pd.api.types.is_numeric_dtype(tmp[effect_col]):
                pearson_r, p_value = pearsonr(tmp[effect_col], tmp[base_col])
        except Exception:
            pearson_r, p_value = None, None

        # --- Prepare result ---
        result = {
            "used_cols": {
                "effect_col": effect_col,
                "base_col": base_col,
                "is_auto_detected": is_auto_detected
            },
            "sample": tmp.head(5).to_dict(orient="records"),
            "summary": summary,
            "note": f"Cặp biến được chọn: r={pearson_r:.4f}, p={'<0.05' if p_value and p_value<0.05 else f'{p_value:.5f}'}, confidence={pairs[0]['confidence']:.2f}" if pearson_r is not None else "⚠ Không thể tính Pearson R"
        }

        # --- Update all significant pairs with Pearson ---
        if is_auto_detected and candidates:
            for p in candidates.get("significant_pairs", []):
                try:
                    s = df[[p["col"], p["base_col"]]].dropna()
                    if len(s) >= 2 and pd.api.types.is_numeric_dtype(s[p["col"]]):
                        r, pv = pearsonr(s[p["col"]], s[p["base_col"]])
                        p["correlation"] = round(r, 4)
                        p["p_value"] = round(pv, 5)
                    else:
                        p["correlation"] = None
                        p["p_value"] = None
                except Exception:
                    p["correlation"] = None
                    p["p_value"] = None
            result["correlation_check"] = candidates.get("significant_pairs", [])
            result["all_candidates"] = candidates
        else:
            result["correlation_check"] = [{
                "col": effect_col,
                "base_col": base_col,
                "correlation": round(pearson_r, 4) if pearson_r is not None else None,
                "p_value": round(p_value, 5) if p_value is not None else None
            }]

        return result

    except Exception as e:
        return {"error": str(e)}
    
def root_cause_tree(problem: str, factors: dict):
    # Very simple storage of provided hypothesis tree
    return {"problem": problem or "Undefined problem", "factors": factors or {}}

def detect_counterintuitive(df: pd.DataFrame, metric_col: str, value_col: str, expected_sign_col: str):
    if metric_col not in df.columns or value_col not in df.columns or expected_sign_col not in df.columns:
        return with_hint("missing columns for detect_counterintuitive", candidates=[metric_col, value_col, expected_sign_col])
    surprising = []
    for _, row in df[[metric_col, value_col, expected_sign_col]].dropna().iterrows():
        val = row[value_col]
        exp = str(row[expected_sign_col]).lower()
        if exp in ("positive","+","pos") and val < 0:
            surprising.append(row.to_dict())
        elif exp in ("negative","-","neg") and val > 0:
            surprising.append(row.to_dict())
    return {"surprising": surprising, "count": len(surprising)}

def generate_counterfactual_row(
    df: pd.DataFrame,
    base_row: pd.Series,
    target_col: str,
    pct_change: float,
    knn_model,
    scaler,
    other_cols: List[str],
    original_df: pd.DataFrame
) -> pd.Series:
    """
    Tạo một hàng counterfactual:
    - Thay đổi target_col theo pct_change
    - Giữ các cột khác "gần giống" base_row nhất có thể (dùng KNN trên không gian other_cols)
    """
    new_row = base_row.copy()
    new_row[target_col] = base_row[target_col] * (1 + pct_change)

    # Vector hóa phần còn lại để tìm hàng gần nhất
    query_vec = scaler.transform(base_row[other_cols].to_frame().T)
    distances, indices = knn_model.kneighbors(query_vec, n_neighbors=1)
    nearest_idx = indices[0][0]
    nearest_row = original_df.iloc[nearest_idx]

    # Gán lại các cột khác từ hàng gần nhất → giữ "bối cảnh" giống thật
    for col in other_cols:
        new_row[col] = nearest_row[col]

    return new_row

def analyze_category_impact(
    df: pd.DataFrame,
    category_col: str,
    numeric_cols: List[str] = None,
    min_count: int = 10,
    effect_threshold: float = 0.5,
    compute_pvalue: bool = True
) -> Dict[str, Any]:


    if category_col not in df.columns:
        return {"error": f"category_col '{category_col}' not found in dataframe"}

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols = [c for c in numeric_cols if c in df.columns and c != category_col]
    if not numeric_cols:
        return {"error": "No numeric columns to analyze"}

    overall_means = df[numeric_cols].mean()
    value_effects = {}
    narratives = []

    for val, group in df.groupby(category_col):
        n = len(group)
        if n < min_count:
            continue

        group_means = group[numeric_cols].mean()
        effects = {}

        for col in numeric_cols:
            delta = float(group_means[col] - overall_means[col])

            # --- Cohen's d với pooled std ---
            other = df[df[category_col] != val][col].dropna()
            group_vals = group[col].dropna()

            if len(group_vals) < 2 or len(other) < 2:
                pooled_std = df[col].std()
            else:
                pooled_std = np.sqrt(
                    ((group_vals.var(ddof=1) * (len(group_vals) - 1)) +
                     (other.var(ddof=1) * (len(other) - 1))) /
                    (len(group_vals) + len(other) - 2)
                )

            cohens_d = delta / (pooled_std + 1e-9) if pooled_std > 0 else 0.0

            effect_info = {
                "delta": delta,
                "cohens_d": cohens_d,
                "count": n,
                "group_mean": float(group_means[col]),
                "overall_mean": float(overall_means[col]),
            }

            # --- p-value ---
            if compute_pvalue and len(group_vals) > 1 and len(other) > 1:
                _, pval = ttest_ind(group_vals, other, equal_var=False)
                effect_info["pvalue"] = float(pval)

            effects[col] = effect_info

            # --- Narrative ---
            if abs(cohens_d) >= effect_threshold:
                direction = "tăng" if delta > 0 else "giảm"
                pv_text = f", p={effect_info['pvalue']:.3f}" if "pvalue" in effect_info else ""
                narratives.append(
                    f"Khi {category_col} là '{val}'thì dẫn đến việc là giá trị của {col} {direction} {abs(delta):.2f} \n"
                    f"Vì (Cohen's d = {cohens_d:.2f}{pv_text}, n={n})"
                )

        value_effects[str(val)] = effects

    # --- Top impact per column ---
    top_impact = {}
    for col in numeric_cols:
        sorted_vals = sorted(
            [(v, eff[col]["cohens_d"]) for v, eff in value_effects.items() if col in eff],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        if sorted_vals:
            top_val, top_d = sorted_vals[0]
            top_impact[col] = {"value": top_val, "cohens_d": top_d}

    return {
        "category_col": category_col,
        "value_effects": value_effects,
        "top_impact_per_column": top_impact,
        "narratives": narratives,
        "min_count_threshold": min_count,
        "effect_threshold": effect_threshold,
    }

def scenario_simulation(
    df: pd.DataFrame,
    change_dict: dict = None,
    min_pct: float = -0.2,
    max_pct: float = 0.2,
    n_steps: int = 10,
    top_k: int = 20,
    n_bootstrap: int = 200,
    min_corr: float = 0.2,
    n_counterfactuals: int = 100,
    effect_threshold: float = 0.5,
    stats: dict = None,
    preprocessing_config: dict = None
) -> Dict[str, Any]:

    detected_info = detect_data_types(df, pipeline_config["data_type_detection"])
    detected_types = detected_info.get("detected_types", {})

    if not detected_types:
        return {"error": "Không thể phát hiện kiểu dữ liệu"}
    
    clean_detected_types = {
        k: v for k, v in detected_types.items()
        if not any(kw in k.lower() for kw in ["cluster", "prediction"])
    }
    if preprocessing_config:
        df_processed, preprocessing_log, updated_types = preprocess_data(
            df.copy(), clean_detected_types, preprocessing_config
        )
    else:
        df_processed = df.copy()
        updated_types = clean_detected_types.copy()

    results: Dict[str, Any] = {}

    if stats is None:
        stats = descriptive_statistics(df)

    numeric_cols = [
        c for c in df_processed.select_dtypes(include=[np.number]).columns
        if not any(kw in c.lower() for kw in ["cluster", "prediction"])
    ]
    categorical_cols = list(stats.get("categorical", {}).keys())
    datetime_cols = list(stats.get("datetime", {}).keys())

    if len(numeric_cols) < 2 and not categorical_cols:
        return {"error": "Cần >=2 cột numeric hoặc có categorical"}

    outcome_cols = auto_detect_target(df, clean_detected_types)
    if isinstance(outcome_cols, str):
        outcome_cols = [outcome_cols]
    elif not outcome_cols:
        return {"error": "Không tìm được outcome hợp lệ"}

    # === LOOP QUA TỪNG OUTCOME ===
    for outcome_col in outcome_cols:
        narratives = []
        scenario_df = df_processed.copy()
        impacted_summary = {}
        hypotheses: List[Dict[str, Any]] = []
        time_series_impacts: List[Dict[str, Any]] = []

        feature_cols = [
            c for c in df_processed.columns
            if c != outcome_col and updated_types.get(c) in ["numeric", "categorical"]
        ]
        
        numeric_features = [c for c in feature_cols if updated_types.get(c) == "numeric"]
        categorical_features = [c for c in feature_cols if updated_types.get(c) == "categorical"]

        if not feature_cols:
            results[outcome_col] = {"error": "Không có feature hợp lệ"}
            continue

        outcome_type = updated_types.get(outcome_col, "numeric")

        # Chuẩn bị dữ liệu
        X = scenario_df[feature_cols].copy()
        y = scenario_df[outcome_col].copy()

        # Encode categorical features
        for col in categorical_features:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        if outcome_type == "categorical":
            y = LabelEncoder().fit_transform(y.astype(str))

        # Train models
        if outcome_type == "numeric":
            lin = LinearRegression().fit(X.fillna(0), y)
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X.fillna(0), y)
            coefs = dict(zip(feature_cols, lin.coef_))
            importances = dict(zip(feature_cols, rf.feature_importances_))
        else:
            lin = LogisticRegression(max_iter=500).fit(X.fillna(0), y)
            rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.fillna(0), y)
            coefs = dict(zip(feature_cols, lin.coef_[0]))
            importances = dict(zip(feature_cols, rf.feature_importances_))

        # Correlation matrix
        corr = df[numeric_cols].corr().fillna(0) if len(numeric_cols) > 1 else pd.DataFrame()

        # Tạo % thay đổi
        pct_values = np.linspace(min_pct, max_pct, n_steps)

        # === Scenario simulation cho từng feature ===
        for col in numeric_features:
            for pct in pct_values:
                delta = df[col].mean() * pct
                scenario_col = f"{col}_scenario_{int(pct*100)}pct_{outcome_col}"
                scenario_df[scenario_col] = df[col] * (1 + pct)
                narratives.append(f"[{outcome_col}] Giả định {col} thay đổi {pct*100:.1f}% → tạo cột {scenario_col}.")

                # Correlation propagation
                for other in numeric_cols:
                    if other == col:
                        continue
                    r = corr.loc[col, other]
                    impact = r * delta * (df[other].std() / (df[col].std() + 1e-9))
                    if abs(impact) > 1e-6:
                        other_scenario_col = f"{other}_scenario_{int(pct*100)}pct"
                        scenario_df[other_scenario_col] = df[other] + impact
                        impacted_summary.setdefault(other, []).append((pct, impact))
                        if impact > 0:
                            trend = "tăng thêm"
                        elif impact == 0:
                            trend = "không đổi"
                        else:
                            trend = "giảm xuống"
                        if r > 0.5:
                            narratives.append(
                                f"Nếu {col} tăng thì {other} cũng tăng. "
                                f"(Chúng đi cùng nhau, giống như bóng với hình). "
                                f"Với thay đổi này, {other} dự kiến {trend} {impact:+.2f} "
                                f"so với giá trị trung bình {df[other].mean():.2f}, "
                                f"tương đương {impact/df[other].mean()*100:.1f}%."
                            )
                        elif r < -0.5:
                            narratives.append(
                                f"Nếu {col} tăng thì {other} lại giảm "
                                f"(Một bên lên thì bên kia xuống, như bập bênh). "
                                f"Với thay đổi này, {other} dự kiến {trend} {impact:+.2f} "
                                f"so với trung bình {df[other].mean():.2f}, "
                                f"tức khoảng {impact/df[other].mean()*100:.1f}%."
                            )
                        # else:
                        #     narratives.append(
                        #         f"{col} và {other} chỉ hơi liên quan "
                        #         f"(Giống như hai người quen xa xa). "
                        #         f"Tác động dự kiến chỉ {impact:+.2f}, "
                        #         f"so với trung bình {df[other].mean():.2f} là rất nhỏ "
                        #         f"({impact/df[other].mean()*100:.2f}%)."
                        #     )


                # ML-based effect
                pred_lin = coefs.get(col, 0) * delta
                pred_rf = importances.get(col, 0) * delta
                narratives.append(f"Linear/Logistic: {col} thay đổi {pct*100:.1f}% → {outcome_col} thay đổi {pred_lin:+.2f}")
                narratives.append(f"RandomForest: {col} importance={importances.get(col,0):.3f} → tác động {pred_rf:+.2f} tới {outcome_col}")

                cat_impacts = defaultdict(list)
                for cat_col in categorical_cols:
                    cat_result = analyze_category_impact(
                        df,
                        category_col=cat_col,
                        numeric_cols=numeric_cols,
                        effect_threshold=effect_threshold
                    )
                    for d in cat_result.get("details", []):
                        key = f"Nếu {cat_col} là '{d['category_value']}'"
                        desc = (
                            f"Thì giá trị của{ d['numeric_col']} thay đổi {d['effect']:+.2f} "
                            f"Vì (Cohen's d={d['cohen_d']:.2f}, p={d['p']:.3f}, n={d['n']})"
                        )
                        cat_impacts[key].append(desc)
                for key, effects in cat_impacts.items():
                    narratives.append(f"Xét [{key}]: " + "; ".join(effects))
        # === Hypothesis generation ===
        # Corr-based
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if any(kw in col1.lower() for kw in ["cluster", "prediction"]):
                    continue
                if any(kw in col2.lower() for kw in ["cluster", "prediction"]):
                    continue
                r = corr.loc[col1, col2] if not corr.empty else 0
                if abs(r) < min_corr:
                    continue
                signs = []
                for _ in range(n_bootstrap):
                    boot = resample(df[[col1, col2]].dropna())
                    if len(boot) > 5:
                        r_boot = boot[col1].corr(boot[col2])
                        signs.append(np.sign(r_boot))
                prob = np.mean([s == np.sign(r) for s in signs if s != 0])
                direction = "tăng" if r > 0 else "giảm"
                hypotheses.append({
                    "giả_thuyết": f"Nếu {col1} tăng thì {col2} có xu hướng {direction} (corr={r:.2f})",
                    "xác suất xảy ra": round(prob*100, 1)
                })

        # ML-based stability
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        coef_signs = {f: [] for f in feature_cols}
        X_df = pd.DataFrame(X)
        y_ser = pd.Series(y)
        for train_idx, test_idx in kf.split(X_df):
            lin_cv = LinearRegression().fit(X_df.iloc[train_idx], y_ser.iloc[train_idx])
            for f, c in zip(feature_cols, lin_cv.coef_):
                coef_signs[f].append(np.sign(c))
        for f, signs in coef_signs.items():
            if len(signs) == 0:
                continue
            stable = np.mean([s == np.sign(np.mean(signs)) for s in signs if s != 0])
            direction = "tăng" if np.mean(signs) > 0 else "giảm"
            hypotheses.append({
                "giả_thuyết": f"Nếu {f} tăng thì {outcome_col} có xu hướng {direction} (OLS)",
                "xác_suất": round(stable*100, 1)
            })

        # === Datetime analysis (Y/Q/M/W) ===
        for dt_col in datetime_cols:
            ts_impacts_for_col = []
            try:
                dt_info = stats["datetime"].get(dt_col, {})
                dayfirst_flag = dt_info.get("dayfirst_used", True)

                temp = df.copy()
                temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce", dayfirst=dayfirst_flag)
                temp = temp.dropna(subset=[dt_col])
                if temp.empty:
                    ts_impacts_for_col.append({
                        "datetime_col": dt_col,
                        "reason": f"Tất cả giá trị của {dt_col} đều NaT sau khi parse (NaN)",
                        "series": []
                    })
                    time_series_impacts.extend(ts_impacts_for_col)
                    continue

                for freq, label in [("Y", "theo năm"), ("Q", "theo quý"), ("M", "theo tháng"), ("W", "theo tuần")]:
                    temp_grouped = temp.copy()
                    temp_grouped[label] = temp_grouped[dt_col].dt.to_period(freq).astype(str)

                    # Lấy trung bình outcome theo chu kỳ
                    ts_summary = (
                        temp_grouped.groupby(label)[outcome_col]
                        .mean()
                        .reset_index()
                        .rename(columns={outcome_col: "mean_value"})
                    )

                    # Lưu block JSON cho chart
                    time_series_impacts.append({
                        "datetime_col": dt_col,
                        "frequency": freq,
                        "label": label,
                        "series": ts_summary.to_dict(orient="records")
                    })
                    
                    print("=== Time Series Impact Block ===")
                    print(time_series_impacts)
                    print("===============================")

                    # Narrative mô tả
                    dt_result = analyze_category_impact(
                        temp_grouped,
                        category_col=label,
                        numeric_cols=numeric_cols,
                        effect_threshold=effect_threshold
                    )
                    if "narratives" in dt_result and dt_result["narratives"]:
                        narratives.extend([f"[{outcome_col}] Xét {label}: {n}" for n in dt_result["narratives"]])

            except Exception as e:
                narratives.append(f"Lỗi khi phân tích datetime {dt_col}: {e}")

        # Lưu kết quả cho outcome này
        top_impacts = {
            col: f"{impacts[0][1]:+.2f}" if impacts else "0.00"
            for col, impacts in impacted_summary.items()
        }
        okr_draft = drivers_to_okrs(hypotheses, outcome_col)
        results[outcome_col] = {
            "narratives": narratives,
            "ảnh_hưởng": top_impacts,
            "ví_dụ_bản_ghi": scenario_df.head(5).to_dict(orient="records"),
            "cột_mới": [c for c in scenario_df.columns if "_scenario" in c],
            "ml_coefficients": coefs,
            "ml_importances": importances,
            "hypotheses": hypotheses,
            "okr_draft": okr_draft,
            "time_series_impacts": time_series_impacts
        }

    return results

def drivers_to_okrs(
    hypotheses: List[Dict],
    outcome: str,
    impacts: Dict[str, Any] = None,
    importances: Dict[str, float] = None,
    baselines: Dict[str, float] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    if not hypotheses:
        return {"Objective": f"Không tìm thấy giả thuyết đủ mạnh để định nghĩa OKR cho {outcome}"}

    # Xếp hạng drivers
    ranked_drivers = []
    if importances:
        ranked_drivers = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    elif impacts:
        ranked_drivers = sorted(impacts.items(), key=lambda x: abs(float(str(x[1]))), reverse=True)

    top_drivers = [d for d, _ in ranked_drivers[:top_k]]

    key_results = []
    for hyp in hypotheses:
        text = hyp.get("giả_thuyết", "")
        prob = hyp.get("xác_suất") or hyp.get("xác suất xảy ra")
        if not text:
            continue

        # Xác định driver
        driver = next((d for d in top_drivers if d in text), None)
        if not driver:  
            # fallback: lấy từ sau "Nếu"
            parts = text.split()
            driver = parts[1] if len(parts) > 1 else "UnknownDriver"

        # Hướng tác động
        if "giảm" in text:
            direction = "giảm"
        elif "tăng" in text:
            direction = "tăng"
        else:
            direction = "thay đổi"

        # Baseline & target
        baseline = baselines.get(driver) if baselines else None
        target = None
        if baseline:
            target = round(baseline * (0.9 if direction == "giảm" else 1.1), 2)

        # Tạo câu KR đúng với giả thuyết gốc
        if baseline and target:
            kr = f"{direction.capitalize()} {outcome} bằng cách {direction} {driver} (từ {baseline} → {target})"
        else:
            kr = f"{direction.capitalize()} {outcome} bằng cách {direction} {driver} (cần baseline/target)"

        if prob:
            kr += f" [Mức tin cậy {prob}%]"

        key_results.append({
            "KPI": driver,
            "KR": kr,
            "Baseline": baseline,
            "Target": target,
            "Nguồn": "CORE_HYPOTHESIS"
        })

    return {
        "Objective": f"Tối ưu {outcome} dựa trên CORE HYPOTHESES",
        "KeyResults": key_results
    }

def causal_psm(df, treatment_col, covariates, outcome_col, match_ratio=1, caliper=None, ipw=False, random_state=42):
    # Minimal safe PSM implementation: returns propensity, IPW ATE or matched ATT if possible
    if treatment_col not in df.columns or outcome_col not in df.columns:
        return with_hint("missing cols for causal_psm", candidates=[treatment_col, outcome_col])
    for c in covariates:
        if c not in df.columns:
            return with_hint(f"covariate {c} not in df", candidates=covariates)
    try:
        df_ = df[[treatment_col] + covariates + [outcome_col]].dropna().copy()
        X = pd.get_dummies(df_[covariates], drop_first=True)
        y = pd.to_numeric(df_[treatment_col], errors="coerce").fillna(0).astype(int)
        if X.shape[0] < 20:
            return with_hint("not enough rows to fit propensity model", candidates=list(df_.columns))
        lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
        lr.fit(X, y)
        ps = lr.predict_proba(X)[:, 1]
        df_ = df_.assign(propensity_score=ps, _treat=y.values)
        res = {"propensity_sample": safe_head(df_[["propensity_score", treatment_col, outcome_col]])}
        if ipw:
            # safe weights
            df_["weight"] = np.where(df_._treat==1, 1/np.clip(df_["propensity_score"], 1e-3, 1-1e-3), 1/np.clip(1-df_["propensity_score"], 1e-3,1-1e-3))
            ate = np.average((2*df_._treat-1) * df_[outcome_col], weights=df_["weight"])
            res["ipw_ate"] = float(ate)
        else:
            # nearest neighbor 1:1 by default
            treated = df_[df_._treat==1]
            control = df_[df_._treat==0]
            if control.shape[0] < 1 or treated.shape[0] < 1:
                res["error"] = "no treated or control rows after dropna"
                return res
            nbrs = NearestNeighbors(n_neighbors=min(match_ratio, max(1, control.shape[0]))).fit(control[["propensity_score"]])
            distances, indices = nbrs.kneighbors(treated[["propensity_score"]])
            matched_control_idx = control.iloc[indices.flatten()].index
            treated_idx = np.repeat(treated.index.values, match_ratio)[:len(matched_control_idx)]
            matched_idx = np.unique(np.concatenate([treated_idx, matched_control_idx]))
            matched = df_.loc[matched_idx]
            att = matched.loc[matched._treat==1, outcome_col].mean() - matched.loc[matched._treat==0, outcome_col].mean()
            res.update({"att": float(att), "matched_n": int(len(matched_idx)), "matched_sample": safe_head(matched)})
        return res
    except Exception as e:
        return {"error": str(e)}

def causal_iv_2sls(df, outcome, treatment, instrument, covariates=None):
    if outcome not in df.columns or treatment not in df.columns or instrument not in df.columns:
        return with_hint("missing columns for IV", candidates=[outcome, treatment, instrument])
    try:
        cols = [outcome, treatment, instrument] + (covariates or [])
        tmp = df[cols].dropna()
        X1 = pd.get_dummies(tmp[[instrument] + (covariates or [])], drop_first=True)
        X1 = sm.add_constant(X1, has_constant='add')
        y1 = tmp[treatment]
        stage1 = sm.OLS(y1, X1).fit()
        tmp["_pred_treat"] = stage1.predict(X1)
        X2 = pd.get_dummies(tmp[["_pred_treat"] + (covariates or [])], drop_first=True)
        X2 = sm.add_constant(X2, has_constant='add')
        y2 = tmp[outcome]
        stage2 = sm.OLS(y2, X2).fit(cov_type="HC1")
        fstat = None
        try:
            fstat = float(stage1.fvalue)
        except Exception:
            fstat = None
        return {"stage1": stage1.summary().as_text()[:200], "stage2": stage2.summary().as_text()[:200], "iv_coef": float(stage2.params.get("_pred_treat", np.nan)), "first_stage_f": fstat}
    except Exception as e:
        return {"error": str(e)}

def causal_did(df, time_col, group_col, outcome_col, treatment_time, treatment_col=None, covariates=None):
    try:
        df_ = df.copy()
        df_[time_col] = pd.to_datetime(df_[time_col], errors="coerce")
        df_ = df_.dropna(subset=[time_col, group_col, outcome_col])
        df_["post"] = (df_[time_col] >= pd.to_datetime(treatment_time)).astype(int)

        if treatment_col and treatment_col in df_.columns:
            df_["treated"] = df_[treatment_col].astype(int)
        elif df_[group_col].nunique() > 2:
            means = df_.groupby(group_col)[outcome_col].mean()
            treated_val = means.idxmax()
            df_["treated"] = (df_[group_col] == treated_val).astype(int)
        else:
            df_["treated"] = pd.to_numeric(df_[group_col], errors="coerce").fillna(0).astype(int)

        df_["did"] = df_["post"] * df_["treated"]

        outcome = f'Q("{outcome_col}")'
        rhs_terms = [f'Q("{t}")' for t in ["post", "treated", "did"]]
        if covariates:
            rhs_terms.extend([f'Q("{c}")' for c in covariates if c in df_.columns])
        formula = f"{outcome} ~ {' + '.join(rhs_terms)}"

        model = smf.ols(formula=formula, data=df_).fit(cov_type="HC1")

        return {
            "did_coef": float(model.params.get("Q(\"did\")", np.nan)),
            "did_pvalue": float(model.pvalues.get("Q(\"did\")", np.nan)),
            "n_obs": int(df_.shape[0])
        }
    except Exception as e:
        return {"error": str(e)}

def causal_did_safe(df, cfg, user_col=None, treatment_time=None, random_state=42, max_iter=5, top_covs=5, pval_threshold=0.05):
    try:
        df_ = df.copy()
        time_col = cfg.get("time_col")
        group_col = cfg.get("group_col")
        treatment_col = cfg.get("treatment_col")
        outcome_col = cfg.get("outcome_col")

        # Kiểm tra bắt buộc
        for col in [time_col, group_col, outcome_col, treatment_col]:
            if col not in df_.columns:
                return {"error": f"Missing required column: {col}"}

        df_[time_col] = pd.to_datetime(df_[time_col], errors="coerce")
        df_ = df_.dropna(subset=[time_col, group_col, outcome_col, treatment_col])
        n_obs = df_.shape[0]

        if treatment_time is None:
            treatment_time = df_[time_col].median()
        df_["post"] = (df_[time_col] >= pd.to_datetime(treatment_time)).astype(int)
        df_["treated"] = df_[treatment_col].astype(int)
        df_["did"] = df_["post"] * df_["treated"]

        # Candidate covariates numeric
        numeric_cols = df_.select_dtypes(include=[np.number]).columns.tolist()
        candidate_covs = [c for c in numeric_cols if c not in [outcome_col, treatment_col, "did", "post"] and df_[c].var() > 1e-6]

        best_result = None

        for iteration in range(max_iter):
            selected_covs = []

            # --- 1. Covariate selection: RF + Lasso ---
            if candidate_covs:
                X = df_[candidate_covs].fillna(0)
                y = df_[outcome_col]
                rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
                rf.fit(X, y)
                ranked = [c for c, imp in sorted(zip(candidate_covs, rf.feature_importances_), key=lambda x: -x[1])]
                top_features = ranked[:top_covs]

                if top_features:
                    lasso = LassoCV(cv=5, random_state=random_state).fit(X[top_features], y)
                    selected_covs = [c for c, coef in zip(top_features, lasso.coef_) if abs(coef) > 1e-6]

            # --- 2. Stabilized IPW ---
            if selected_covs:
                X_cov = df_[selected_covs]
                X_scaled = StandardScaler().fit_transform(X_cov)
                ps_model = LogisticRegression(solver="liblinear", random_state=random_state)
                ps_model.fit(X_scaled, df_["treated"])
                ps = ps_model.predict_proba(X_scaled)[:, 1]
            else:
                ps = np.full(df_.shape[0], df_["treated"].mean())

            treated_mean = df_["treated"].mean()
            df_["ipw"] = np.where(
                df_["treated"] == 1,
                treated_mean / np.clip(ps, 1e-3, 1-1e-3),
                (1 - treated_mean) / np.clip(1 - ps, 1e-3, 1-1e-3)
            )

            # --- 3. Fit OLS DiD với covariates selected ---
            if selected_covs:
                rhs_terms = ['Q("post")', 'Q("treated")', 'Q("did")'] + [f'Q("{c}")' for c in selected_covs]
            else:
                rhs_terms = ['Q("post")', 'Q("treated")', 'Q("did")']

            formula = f'Q("{outcome_col}") ~ ' + ' + '.join(rhs_terms)
            model = smf.ols(formula=formula, data=df_, weights=df_["ipw"])
            fit_res = model.fit(cov_type="HC1") if user_col is None else model.fit(cov_type="cluster", cov_kwds={"groups": df_[user_col]})

            # --- 4. Lọc covariates significant ---
            significant_covs = [c for c in selected_covs if fit_res.pvalues.get(f'Q("{c}")', 1) <= pval_threshold]

            # Nếu có covariates significant → refit
            if significant_covs:
                rhs_terms = ['Q("post")', 'Q("treated")', 'Q("did")'] + [f'Q("{c}")' for c in significant_covs]
                formula = f'Q("{outcome_col}") ~ ' + ' + '.join(rhs_terms)
                model = smf.ols(formula=formula, data=df_, weights=df_["ipw"])
                fit_res = model.fit(cov_type="HC1") if user_col is None else model.fit(cov_type="cluster", cov_kwds={"groups": df_[user_col]})

            did_coef = float(fit_res.params.get('Q("did")', np.nan))
            did_pval = float(fit_res.pvalues.get('Q("did")', np.nan))
            ci_lower, ci_upper = fit_res.conf_int().loc['Q("did")'].tolist()

            if best_result is None or (did_pval <= 0.05) or (abs(did_coef) > abs(best_result["did_coef"])):
                best_result = {
                    "did_coef": did_coef,
                    "did_pvalue": did_pval,
                    "ci_95": [ci_lower, ci_upper],
                    "selected_covariates": significant_covs,
                    "r_squared": fit_res.rsquared,
                    "iteration": iteration + 1
                }

            # Nếu ATT significant → dừng vòng lặp
            if did_pval <= 0.05:
                break

            # Thay đổi candidate covariates cho iteration tiếp theo
            if selected_covs:
                candidate_covs = [c for c in candidate_covs if c not in selected_covs[:1]]

        # --- 5. Pre-trend check ---
        grouped = df_.groupby([time_col, "treated"])[outcome_col].mean().reset_index()
        pre_trends, post_trends = [], []
        for t in sorted(df_[time_col].unique()):
            treatment_mean = grouped.loc[(grouped[time_col]==t) & (grouped["treated"]==1), outcome_col].values
            control_mean = grouped.loc[(grouped[time_col]==t) & (grouped["treated"]==0), outcome_col].values
            point = {
                "time": t,
                "treatment_mean": float(treatment_mean[0]) if len(treatment_mean)>0 else None,
                "control_mean": float(control_mean[0]) if len(control_mean)>0 else None
            }
            if t < pd.to_datetime(treatment_time):
                pre_trends.append(point)
            else:
                post_trends.append(point)

        pre_df = pd.DataFrame(pre_trends).dropna()
        pre_trend_warning = False
        if not pre_df.empty:
            t_vals = (pre_df["time"] - pre_df["time"].min()).dt.days.values.reshape(-1,1)
            slope_treatment = LinearRegression().fit(t_vals, pre_df["treatment_mean"]).coef_[0]
            slope_control = LinearRegression().fit(t_vals, pre_df["control_mean"]).coef_[0]
            threshold = 0.1 * (df_[outcome_col].max() - df_[outcome_col].min())
            pre_trend_warning = abs(slope_treatment - slope_control) > threshold

        interpretation = f"ATT estimate = {best_result['did_coef']:.4f}, p-value = {best_result['did_pvalue']:.4f}, 95% CI = [{best_result['ci_95'][0]:.4f}, {best_result['ci_95'][1]:.4f}]"

        result = {
            "did_coef": best_result['did_coef'],
            "did_pvalue": best_result['did_pvalue'],
            "n_obs": n_obs,
            "r_squared": best_result['r_squared'],
            "ci_95": best_result['ci_95'],
            "interpretation": interpretation,
            "pre_trends_warning": pre_trend_warning,
            "pre_trends": pre_trends,
            "post_trends": post_trends,
            "selected_covariates": best_result['selected_covariates']
        }

        return result

    except Exception as e:
        return {"error": str(e)}

def adaptive_storytelling_phd(df: pd.DataFrame, audience_col: str = None,
                             roi_col: str = None, effect_col: str = None,
                             method_col: str = None, driver_col: str = None,
                             assumptions_col: str = None, top_k=3):
    # clone
    df = df.copy()
    proxies = {}

    # ensure audience_col exists; if not create default audience "Analyst"
    if audience_col and audience_col in df.columns:
        aud_col = audience_col
    else:
        aud_col = "__audience__"
        df[aud_col] = "Analyst"
        proxies["audience_col"] = aud_col

    # ROI proxy: if roi_col provided and exists -> use; else try revenue & cost/profit
    if roi_col and roi_col in df.columns:
        roi_used = roi_col
    else:
        # try revenue & cost
        revenue_candidates = [c for c in df.columns if "revenue" in c.lower() or "total" in c.lower() or "sales" in c.lower()]
        cost_candidates = [c for c in df.columns if "cost" in c.lower() or "expense" in c.lower() or "cogs" in c.lower()]
        if revenue_candidates and cost_candidates:
            r = revenue_candidates[0]; c = cost_candidates[0]
            proxy_name = "__roi_proxy__"
            # avoid zero division
            df[proxy_name] = (df[r] - df[c]) / df[r].replace({0: np.nan})
            roi_used = proxy_name
            proxies["roi_proxy"] = (r, c, proxy_name)
        else:
            roi_used = None

    # Effect proxy: if effect_col exists use, else if treatment & outcome exist compute simple ATT per treatment
    if effect_col and effect_col in df.columns:
        effect_used = effect_col
    else:
        treat = None
        if "treatment" in df.columns:
            treat = "treatment"
        else:
            # find binary-like column as treatment heuristic
            bin_cols = [c for c in df.columns if df[c].nunique(dropna=True)==2]
            if bin_cols:
                treat = bin_cols[0]
        if treat:
            # compute ATT proxy if possible
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                outcome = max(numeric_cols, key=lambda c: df[c].var(skipna=True) if df[c].dropna().shape[0]>1 else 0)
                try:
                    treated_mean = df[df[treat]==1][outcome].mean()
                    control_mean = df[df[treat]==0][outcome].mean()
                    att_val = treated_mean - control_mean
                    rel = att_val / control_mean if control_mean and not np.isnan(control_mean) else att_val
                    df["_effect_att_proxy_rel"] = rel
                    effect_used = "_effect_att_proxy_rel"
                    proxies["effect_att_proxy"] = {"treatment_col": treat, "outcome_col": outcome, "att": att_val, "relative": rel}
                except Exception:
                    effect_used = None
            else:
                effect_used = None
        else:
            effect_used = None

    # For method_col/driver_col/assumptions_col: only use if present
    method_used = method_col if (method_col and method_col in df.columns) else None
    driver_used = driver_col if (driver_col and driver_col in df.columns) else None
    assumptions_used = assumptions_col if (assumptions_col and assumptions_col in df.columns) else None

    # Compose story per row
    def row_story(row):
        aud = row[aud_col] if aud_col in row else "Analyst"
        parts = []
        # High-level impact (CEO)
        if roi_used and roi_used in row and pd.notna(row[roi_used]):
            parts.append(f"Estimated ROI {row[roi_used]:.2%}")
        if effect_used and effect_used in row and pd.notna(row[effect_used]):
            # effect_used may be relative or absolute; format sensibly
            try:
                val = float(row[effect_used])
                if abs(val) < 10:  # likely relative like 0.05
                    parts.append(f"Effect ~{val:.2%}")
                else:
                    parts.append(f"Effect ~{val:.2f}")
            except Exception:
                parts.append(f"Effect: {row[effect_used]}")
        if driver_used and driver_used in row and pd.notna(row[driver_used]):
            parts.append(f"Driver: {row[driver_used]}")
        if method_used and method_used in row and pd.notna(row[method_used]):
            parts.append(f"Method: {row[method_used]}")
        if assumptions_used and assumptions_used in row and pd.notna(row[assumptions_used]):
            parts.append(f"Assumptions: {row[assumptions_used]}")

        # Build audience-targeted phrasing (PhD-style: concise, nuance, actions)
        if aud == "CEO":
            # pick top 2 parts: impact + driver
            impact = [p for p in parts if "ROI" in p or "Effect" in p]
            driver = [p for p in parts if p.startswith("Driver")]
            headline = " | ".join((impact + driver)[:2])
            if headline:
                return f"Executive: {headline}. Strategic recommendation: prioritize top driver and reallocate budget."
            else:
                return "Executive: No clear impact estimated. Recommendation: supply revenue/cost or effect columns."
        elif aud == "Analyst":
            # method + numeric nuance
            lines = []
            if method_used:
                lines.append(f"Method applied: {row.get(method_used,'N/A')}")
            if effect_used and effect_used in row:
                lines.append(f"Estimated effect: {row[effect_used]}")
            # include proxies note
            if proxies:
                lines.append(f"Proxies used: {list(proxies.keys())}")
            return "Analyst: " + " | ".join(lines) if lines else "Analyst: No technical signals available."
        elif aud == "Ops":
            # action-oriented
            if driver_used:
                return f"Ops: Focus on {row[driver_used]} — run A/B on operational fixes and track churn/retention."
            if roi_used:
                return f"Ops: ROI signals exist. Try targeted pilot to validate before broad rollout."
            return "Ops: No immediate operational action suggested; collect more diagnostic metrics."
        else:
            # general
            return " | ".join(parts) if parts else "No storyable signals found."

    # if dataset small, produce single-canonical story else per-row (limit rows returned)
    try:
        if df.shape[0] == 0:
            return {"error": "empty dataframe"}
        # produce stories for up to top_k rows and attach proxy-info
        sample = df.head(top_k).copy()
        sample["story"] = sample.apply(row_story, axis=1)
        return {"sample_stories": sample[["story"]].to_dict(orient="records"), "proxies": proxies}
    except Exception as e:
        return {"error": str(e)}

def robustness_checks(df, func, drop_top_percent=[0.01], subgroup_cols=None, n_bootstrap=5, time_col=None, **kwargs):
    results = {}
    try:
        results["baseline"] = func(df.copy(), **kwargs)
    except Exception as e:
        results["baseline"] = {"error": str(e)}

    for p in drop_top_percent:
        df2 = df.copy()
        for c in df2.select_dtypes(include=[np.number]).columns:
            cutoff = df2[c].quantile(1 - p)
            df2 = df2[df2[c] <= cutoff]
        try:
            results[f"drop_top_{int(p*100)}pct"] = func(df2, **kwargs)
        except Exception as e:
            results[f"drop_top_{int(p*100)}pct"] = {"error": str(e)}

    if subgroup_cols:
        for col in subgroup_cols:
            results[f"by_{col}"] = {}
            for val, sub in df.groupby(col):
                try:
                    results[f"by_{col}"][str(val)] = func(sub.copy(), **kwargs)
                except Exception as e:
                    results[f"by_{col}"][str(val)] = {"error": str(e)}

    results["bootstrap"] = {}
    for i in range(n_bootstrap):
        sample = df.sample(frac=0.8, replace=True, random_state=i)
        try:
            results["bootstrap"][f"iter_{i}"] = func(sample, **kwargs)
        except Exception as e:
            results["bootstrap"][f"iter_{i}"] = {"error": str(e)}

    if time_col and time_col in df.columns:
        results["rolling"] = {}
        try:
            df_ = df.dropna(subset=[time_col])
            df_[time_col] = pd.to_datetime(df_[time_col], errors="coerce")
            for y in sorted(df_[time_col].dt.year.unique()):
                subset = df_[df_[time_col].dt.year == y]
                if len(subset) > 30:
                    results["rolling"][str(y)] = func(subset, **kwargs)
        except Exception as e:
            results["rolling"]["error"] = str(e)
    return results

def explain_missing(name: str, cfg: dict, df: pd.DataFrame) -> dict:
    reqs = REQUIREMENTS.get(name, {})
    missing = []
    for k, desc in reqs.items():
        val = cfg.get(k)
        # Nếu là list → cho phép rỗng
        if isinstance(val, list):
            # OK nếu là list (dù rỗng)
            if not isinstance(val, list):
                missing.append(f"{k} → expected list, got {type(val)}")
            # Không coi [] là missing → KHÔNG làm gì cả
        else:
            # Với scalar (str, int...): phải tồn tại và không None/empty
            if not val:
                missing.append(f"{k} → required: {desc}")
            elif isinstance(val, str) and val not in df.columns:
                missing.append(f"{k} → column '{val}' not found in dataset")
    return {
        "error": f"Missing config for {name.upper()}",
        "requirements": reqs,
        "missing": missing
    }

def deep_merge(base: dict, extra: dict) -> dict:
    result = base.copy()
    for k, v in (extra or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def beyond_eda(df: pd.DataFrame, config: dict = None):
    print(">>> [beyond_eda] starting")
    
    if not isinstance(df, pd.DataFrame):
        return {"error": "Input must be a pandas DataFrame"}

    # --- Auto-detect config ---
    auto_cfg = auto_detect_config(df)
    encoded_cols = auto_cfg.get("encoded_cols", {})
    print(">>> [beyond_eda] auto-detected config:", auto_cfg)

    # --- Merge user config ---
    cfg = deep_merge(auto_cfg, config or {})
    algo_cfg = cfg.get("algo_config", {})
    print(">>> [beyond_eda] merged config:", cfg)

    # --- Apply encoded columns ---
    df_analysis = df.copy()
    for col_name, col_series in encoded_cols.items():
        df_analysis[col_name] = col_series

    results = {}

    # --- Cohort ---
    cohort_cfg = algo_cfg.get("cohort", {})
    try:
        user_col = cohort_cfg.get("user_col")
        date_col = cohort_cfg.get("date_col")
        if user_col in df_analysis.columns and date_col in df_analysis.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_analysis[date_col]):
                df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors="coerce")
            results["cohort"] = cohort_retention(
                df_analysis,
                user_col=user_col,
                date_col=date_col,
                period=cohort_cfg.get("period", "M")
            )
        else:
            results["cohort"] = explain_missing("cohort", cohort_cfg, df_analysis)
    except Exception as e:
        results["cohort"] = {"error": str(e)}

    # --- PSM ---
    psm_cfg = algo_cfg.get("psm", {})
    try:
        treatment_col = psm_cfg.get("treatment_col")
        outcome_col = psm_cfg.get("outcome_col")
        covariates = psm_cfg.get("covariates", [])
        if treatment_col in df_analysis.columns and outcome_col in df_analysis.columns:
            results["psm"] = causal_psm(
                df_analysis,
                treatment_col=treatment_col,
                covariates=covariates,
                outcome_col=outcome_col,
                ipw=True
            )
        else:
            results["psm"] = explain_missing("psm", psm_cfg, df_analysis)
    except Exception as e:
        results["psm"] = {"error": str(e)}

    # --- DiD (tích hợp nâng cấp causal_did_safe) ---
    did_cfg = algo_cfg.get("did", {})
    try:
        did_group_col = did_cfg.get("group_col")
        did_time_col = did_cfg.get("time_col")
        did_outcome_col = did_cfg.get("outcome_col")
        did_treatment_col = did_cfg.get("treatment_col")
        if (
            did_group_col in df_analysis.columns and
            did_time_col in df_analysis.columns and
            did_outcome_col in df_analysis.columns and
            did_treatment_col in df_analysis.columns
        ):
            # Sử dụng phiên bản nâng cấp: tự động chọn covariates significant
            results["did"] = causal_did_safe(
                df_analysis,
                did_cfg,
                user_col=user_col
            )
        else:
            results["did"] = explain_missing("did", did_cfg, df_analysis)
    except Exception as e:
        results["did"] = {"error": str(e)}

    # --- IV ---
    iv_cfg = algo_cfg.get("iv", {})
    try:
        instrument_col = iv_cfg.get("instrument_col")
        iv_treatment_col = iv_cfg.get("treatment_col")
        iv_outcome_col = iv_cfg.get("outcome_col")
        covariates = iv_cfg.get("covariates", [])
        if (
            instrument_col in df_analysis.columns and
            iv_treatment_col in df_analysis.columns and
            iv_outcome_col in df_analysis.columns
        ):
            results["iv"] = causal_iv_2sls(
                df_analysis,
                outcome=iv_outcome_col,
                treatment=iv_treatment_col,
                instrument=instrument_col,
                covariates=covariates
            )
        else:
            results["iv"] = explain_missing("iv", iv_cfg, df_analysis)
    except Exception as e:
        results["iv"] = {"error": str(e)}

    # --- Robustness ---
    try:
        if treatment_col in df_analysis.columns and outcome_col in df_analysis.columns:
            results["robustness"] = robustness_checks(
                df_analysis,
                func=lambda d, **kw: causal_psm(
                    d,
                    treatment_col=treatment_col,
                    covariates=covariates,
                    outcome_col=outcome_col,
                    ipw=True
                ),
                drop_top_percent=[0.01, 0.05],
                subgroup_cols=[treatment_col],
                n_bootstrap=5,
                time_col=did_cfg.get("time_col")
            )
        else:
            results["robustness"] = explain_missing("robustness", psm_cfg, df_analysis)
    except Exception as e:
        results["robustness"] = {"error": str(e)}

    # --- Survival ---
    try:
        surv_cfg = cfg.get("survival", {})
        if surv_cfg.get("duration_col") and surv_cfg.get("event_col"):
            results["survival"] = survival_km_cox(
                df_analysis,
                duration_col=surv_cfg["duration_col"],
                event_col=surv_cfg["event_col"],
                covariates=surv_cfg.get("covariates", [])
            )
        else:
            results["survival"] = explain_missing("survival", surv_cfg, df)
    except Exception as e:
        results["survival"] = {"error": str(e)}

    # --- Nonparametric ---
    try:
        nonparam_cfg = cfg.get("nonparam", {})
        if nonparam_cfg.get("group_col") and nonparam_cfg.get("outcome_col"):
            results["nonparam"] = nonparametric_tests(
                df_analysis,
                group_col=nonparam_cfg["group_col"],
                value_col=nonparam_cfg["outcome_col"]
            )
        else:
            results["nonparam"] = explain_missing("nonparam", nonparam_cfg, df)
    except Exception as e:
        results["nonparam"] = {"error": str(e)}

    # --- Effect to business ---
    try:
        eff_cfg = cfg.get("effect", {})
        result_effect = effect_to_business(
            df_analysis,
            effect_col=eff_cfg.get("effect_col"),
            base_col=eff_cfg.get("base_col"),
            scale=eff_cfg.get("scale", 1.0)
        )
        # Nếu auto-detect, thêm confidence từ detect_effect_base
        if "all_candidates" in result_effect:
            sig_pairs = result_effect["all_candidates"].get("significant_pairs", [])
            if len(sig_pairs) > 0:
                result_effect["effect_confidence_pct"] = sig_pairs[0].get("confidence", 0.0) * 100
                result_effect["base_confidence_pct"] = next(
                    (b.get("confidence",0.0) for b in result_effect["all_candidates"].get("base_candidates",[])
                     if b["col"]==sig_pairs[0]["base_col"]), 0.0
                ) * 100
            else:
                result_effect["effect_confidence_pct"] = 0.0
                result_effect["base_confidence_pct"] = 0.0
        results["effect"] = result_effect
    except Exception as e:
        results["effect"] = {"error": str(e)}

    # --- Root cause ---
    try:
        rc_cfg = cfg.get("root_cause", {})
        results["root_cause"] = root_cause_tree(
            problem=rc_cfg.get("problem", "Undefined problem"),
            factors=rc_cfg.get("factors", {})
        )
    except Exception as e:
        results["root_cause"] = {"error": str(e)}

    # --- Counterintuitive ---
    try:
        ci_cfg = cfg.get("counterintuitive", {})
        if ci_cfg.get("metric_col") and ci_cfg.get("value_col") and ci_cfg.get("expected_sign_col"):
            results["counterintuitive"] = detect_counterintuitive(
                df_analysis,
                metric_col=ci_cfg["metric_col"],
                value_col=ci_cfg["value_col"],
                expected_sign_col=ci_cfg["expected_sign_col"]
            )
        else:
            results["counterintuitive"] = explain_missing("counterintuitive", ci_cfg, df)
    except Exception as e:
        results["counterintuitive"] = {"error": str(e)}
    
    # --- Scenario ---
    try:
        results["scenario"] = scenario_simulation(df, change_dict=cfg.get("change_dict"))
    except Exception as e:
        results["scenario"] = {"error": str(e)}
    pprint(results)
    # --- Storytelling ---
    try:
        results["storytelling"] = adaptive_storytelling_phd(
            df,
            audience_col=cfg.get("audience_col"),
            roi_col=cfg.get("roi_col"),
            effect_col=cfg.get("effect_col"),
            method_col=cfg.get("method_col"),
            driver_col=cfg.get("driver_col"),
            assumptions_col=cfg.get("assumptions_col"),
            top_k=5
        )
    except Exception as e:
        results["storytelling"] = {"error": str(e)}


    print(">>> [beyond_eda] finished")
    return {"config": cfg, "results": results}
