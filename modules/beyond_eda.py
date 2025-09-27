# modules/beyond_eda.py
from typing import Any, Dict, List
import warnings
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from modules.eda import analyze_column
from scipy.stats import ttest_ind
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

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
def _scale_01(s: pd.Series) -> pd.Series:
    if s.size == 0:
        return s
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def auto_detect_config(df: pd.DataFrame, top_k: int = 3) -> Dict[str, Any]:
    """
    Stat-driven auto-detection (no hard keyword reliance).
    Returns dict with:
      - best guesses (outcome_col, treatment_col, ...)
      - candidate lists with scores
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")

    n = len(df)
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- outcome scoring ---
    outcome_candidates = []
    if numeric_cols:
        var = df[numeric_cols].var(skipna=True)
        var_s = _scale_01(var)
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs().fillna(0)
            corr_mean = {c: corr.loc[c, [o for o in numeric_cols if o != c]].mean() for c in numeric_cols}
            corr_s = _scale_01(pd.Series(corr_mean))
        else:
            corr_s = pd.Series(0.0, index=numeric_cols)

        score = 0.7 * var_s + 0.3 * corr_s
        score = score.sort_values(ascending=False)
        outcome_candidates = [(c, float(score[c])) for c in score.index[:top_k]]
    outcome_col = outcome_candidates[0][0] if outcome_candidates else None

    # --- treatment scoring ---
    bin_cols = [c for c in cols if df[c].nunique(dropna=True) == 2]
    treat_scores = {}
    for c in bin_cols:
        vc = df[c].value_counts(normalize=True, dropna=True)
        maj = float(vc.iloc[0]) if len(vc) else 1.0
        balance = 1 - abs(maj - 0.5) / 0.5
        corr_with_outcome = 0.0
        if outcome_col and pd.api.types.is_numeric_dtype(df[outcome_col]):
            try:
                codes = pd.Categorical(df[c]).codes
                corr_with_outcome = abs(pd.Series(codes).corr(df[outcome_col]))
                if pd.isna(corr_with_outcome):
                    corr_with_outcome = 0.0
            except Exception:
                corr_with_outcome = 0.0
        treat_scores[c] = 0.65 * balance + 0.35 * corr_with_outcome
    treatment_candidates = sorted(treat_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    treatment_col = treatment_candidates[0][0] if treatment_candidates else None

    # --- ID (high cardinality) ---
    id_scores = {c: df[c].nunique()/max(1, n) for c in cols}
    id_candidates = [c for c, score in id_scores.items() if score >= 0.8 or df[c].nunique() >= max(50, 0.5*n)]
    user_col = id_candidates[0] if id_candidates else None

    # --- duration (non-negative, skewed) ---
    dur_candidates = []
    for c in numeric_cols:
        ser = df[c].dropna()
        if len(ser) < 5:
            continue
        if ser.min() >= 0:
            dur_candidates.append((c, float(ser.skew())))
    dur_candidates.sort(key=lambda x: x[1], reverse=True)
    duration_col = dur_candidates[0][0] if dur_candidates else None

    # --- event (rare binary) ---
    event_candidates = []
    for c in bin_cols:
        freqs = df[c].value_counts(normalize=True, dropna=True)
        if len(freqs) == 0: 
            continue
        minority = float(freqs.min())
        rarity = 1 - (minority * 2)
        corr_to_duration = 0.0
        if duration_col:
            try:
                codes = pd.Categorical(df[c]).codes
                corr_to_duration = abs(pd.Series(codes).corr(df[duration_col]))
                if pd.isna(corr_to_duration): corr_to_duration = 0.0
            except Exception:
                pass
        score = 0.6 * rarity + 0.4 * corr_to_duration
        event_candidates.append((c, score))
    event_candidates.sort(key=lambda x: x[1], reverse=True)
    event_col = event_candidates[0][0] if event_candidates else None

    # --- group (categorical with 2..20 levels) ---
    group_candidates = []
    for c in cols:
        if c == user_col: continue
        nunq = df[c].nunique(dropna=True)
        if 2 <= nunq <= 20:
            probs = df[c].value_counts(normalize=True, dropna=True)
            ent = -(probs * np.log2(probs+1e-9)).sum()
            group_candidates.append((c, ent))
    group_candidates.sort(key=lambda x: x[1], reverse=True)
    group_col = group_candidates[0][0] if group_candidates else None

    # --- instrument (corr with treatment not outcome) ---
    instrument_candidates = []
    if treatment_col:
        treat_enc = pd.Categorical(df[treatment_col]).codes
        for c in cols:
            if c in (treatment_col, outcome_col): continue
            if df[c].nunique(dropna=True) < 2: continue
            try:
                inst_codes = pd.Categorical(df[c]).codes
                corr_t = abs(pd.Series(inst_codes).corr(pd.Series(treat_enc)))
                corr_o = 0.0
                if outcome_col and pd.api.types.is_numeric_dtype(df[outcome_col]):
                    corr_o = abs(pd.Series(inst_codes).corr(df[outcome_col]))
                score = corr_t - corr_o
                if score > 0.05:
                    instrument_candidates.append((c, float(score)))
            except Exception:
                continue
    instrument_candidates.sort(key=lambda x: x[1], reverse=True)
    instrument_col = instrument_candidates[0][0] if instrument_candidates else None

    # --- effect & base ---
    effect_col = None
    base_col = None
    if numeric_cols:
        effect_col = sorted([(c, df[c].var(skipna=True)) for c in numeric_cols], key=lambda x: x[1], reverse=True)[0][0]
        base_col = sorted([(c, df[c].sum(skipna=True)) for c in numeric_cols], key=lambda x: x[1], reverse=True)[0][0]

    used = {outcome_col, treatment_col, user_col, duration_col, event_col, group_col, instrument_col, effect_col, base_col}
    covariates = [c for c in numeric_cols if c not in used]

    return {
        "outcome_col": outcome_col,
        "outcome_candidates": outcome_candidates,
        "treatment_col": treatment_col,
        "treatment_candidates": treatment_candidates,
        "user_col": user_col,
        "duration_col": duration_col,
        "event_col": event_col,
        "group_col": group_col,
        "instrument_col": instrument_col,
        "effect_col": effect_col,
        "base_col": base_col,
        "covariates": covariates
    }

# ---- explain why choices were made ----
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
    """
    Safe cohort: fallback returns structure with hint if columns missing.
    """
    if user_col not in df.columns or date_col not in df.columns:
        return with_hint("missing user_col or date_col for cohort_retention", candidates=[user_col, date_col])
    try:
        tmp = df[[user_col, date_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if tmp[date_col].isna().all():
            return with_hint("date_col could not be parsed as datetime", candidates=[date_col])
        tmp["cohort_period"] = tmp[date_col].dt.to_period(period).dt.to_timestamp()
        tmp["activity_period"] = tmp[date_col].dt.to_period(period).dt.to_timestamp()
        first = tmp.groupby(user_col)["cohort_period"].min().reset_index().rename(columns={"cohort_period":"cohort"})
        tmp = tmp.merge(first, on=user_col, how="left")
        if period=='M':
            tmp["period_number"] = ((tmp["activity_period"].dt.year - tmp["cohort"].dt.year) * 12 + (tmp["activity_period"].dt.month - tmp["cohort"].dt.month))
        else:
            tmp["period_number"] = ((tmp["activity_period"] - tmp["cohort"]) / np.timedelta64(1, 'D')).astype(int)
        cohort_counts = tmp.groupby(["cohort", "period_number"])[user_col].nunique().reset_index()
        pivot = cohort_counts.pivot_table(index="cohort", columns="period_number", values=user_col, fill_value=0)
        size = pivot.iloc[:, 0] if pivot.shape[1]>0 else pd.Series()
        retention = pivot.divide(size, axis=0).fillna(0)
        return {"cohort_matrix": pivot.round(0).to_dict(), "retention_matrix": retention.round(3).to_dict()}
    except Exception as e:
        return {"error": str(e)}

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
    from scipy.stats import spearmanr, kruskal, mannwhitneyu
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

def effect_to_business(df: pd.DataFrame, effect_col: str, base_col: str, scale=1.0):
    if effect_col not in df.columns or base_col not in df.columns:
        return with_hint("missing effect_col or base_col for effect_to_business", candidates=[effect_col, base_col])
    try:
        tmp = df[[effect_col, base_col]].dropna().copy()
        tmp["absolute_impact"] = tmp[base_col] * tmp[effect_col] * scale
        tmp["new_value"] = tmp[base_col] + tmp["absolute_impact"]
        tmp["roi_estimate"] = tmp["absolute_impact"] / tmp[base_col].replace({0:np.nan})
        return {"sample": safe_head(tmp), "summary": {"total_impact": float(tmp["absolute_impact"].sum()), "mean_roi": float(tmp["roi_estimate"].mean())}}
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
                    f"Khi {category_col} = '{val}', {col} {direction} {abs(delta):.2f} "
                    f"(Cohen's d = {cohens_d:.2f}{pv_text}, n={n})"
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
    effect_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Scenario simulation cho tất cả numeric columns:
    1. Thay đổi feature trong khoảng min_pct → max_pct.
    2. Tính correlation propagation.
    3. Tính ML-based effect (LinearRegression + RandomForest).
    4. Xuất narratives + bảng kết quả scenario + hypotheses.
    """
    scenario_df = df.copy()
    narratives = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need >=2 numeric columns"}

    # Outcome mặc định là cột có variance lớn nhất
    outcome_col = df[numeric_cols].var().idxmax()
    feature_cols = [c for c in numeric_cols if c != outcome_col]

    if not feature_cols:
        return {"error": "No valid feature columns"}

    # Train ML models
    X = df[feature_cols].fillna(0)
    y = df[outcome_col].fillna(0)
    lin = LinearRegression().fit(X, y)
    coefs = dict(zip(feature_cols, lin.coef_))
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    importances = dict(zip(feature_cols, rf.feature_importances_))

    # Correlation matrix
    corr = df[numeric_cols].corr().fillna(0)

    # Tạo danh sách % thay đổi
    pct_values = np.linspace(min_pct, max_pct, n_steps)
    impacted_summary = {}

    # === Scenario simulation cho từng feature ===
    for col in feature_cols:
        for pct in pct_values:
            delta = df[col].mean() * pct
            scenario_col = f"{col}_scenario_{int(pct*100)}pct"
            scenario_df[scenario_col] = df[col] * (1 + pct)
            narratives.append(f"Giả định {col} thay đổi {pct*100:.1f}% → tạo cột {scenario_col}.")

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
            narratives.append(f"LinearRegression: {col} thay đổi {pct*100:.1f}% → {outcome_col} thay đổi {pred_lin:+.2f} (coef={coefs[col]:+.3f})")
            pred_rf = importances.get(col, 0) * delta
            narratives.append(f"RandomForest: {col} importance={importances[col]:.3f} → tác động khoảng {pred_rf:+.2f} tới {outcome_col}")
            for cat_col in categorical_cols: 
                cat_result = analyze_category_impact( df, category_col=cat_col, numeric_cols=numeric_cols, effect_threshold=effect_threshold ) 
                if "narratives" in cat_result and cat_result["narratives"]: 
                    narratives.extend([f"[{cat_col}] {n}" for n in cat_result["narratives"]])
    # Tóm tắt top impacted cols (lấy impact đầu tiên trong list)
    top_impacts = {
        col: f"{impacts[0][1]:+.2f}" if impacts else "0.00"
        for col, impacts in impacted_summary.items()
    }

    # === Hypothesis generation ===
    hypotheses: List[Dict[str, Any]] = []

    # Corr-based hypotheses
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            r = corr.loc[col1, col2]
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
                "xác_suất": round(prob*100, 1)
            })

    # ML-based hypotheses (cross-validation OLS stability)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    coef_signs = {f: [] for f in feature_cols}
    for train_idx, test_idx in kf.split(X):
        lin_cv = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
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

    return {
        "scenario": {
            "narratives": narratives,
            "ảnh_hưởng": top_impacts,
            "ví_dụ_bản_ghi": scenario_df.head(5).to_dict(orient="records"),
            "cột_mới": [c for c in scenario_df.columns if "_scenario" in c],
            "ml_outcome": outcome_col,
            "ml_coefficients": coefs,
            "ml_importances": importances
        },
        "hypotheses": hypotheses
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
    """
    Difference-in-Differences estimator.
    """
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
        rhs = "post + treated + did"
        if covariates:
            rhs += " + " + " + ".join(covariates)
        formula = f"{outcome_col} ~ {rhs}"
        model = smf.ols(formula=formula, data=df_).fit(cov_type="HC1")

        return {
            "did_coef": float(model.params.get("did", np.nan)),
            "did_pvalue": float(model.pvalues.get("did", np.nan)),
            "n_obs": int(df_.shape[0])
        }
    except Exception as e:
        return {"error": str(e)}

def causal_did_safe(df, cfg):
    if not cfg.get("time_col") or not cfg.get("group_col") or not cfg.get("outcome_col"):
        return with_hint("missing config for DiD", candidates=[cfg.get("time_col"), cfg.get("group_col"), cfg.get("outcome_col")])
    try:
        treatment_time = cfg.get("treatment_time") or pd.to_datetime(df[cfg["time_col"]], errors="coerce").median()
        return causal_did(
            df,
            time_col=cfg["time_col"],
            group_col=cfg["group_col"],
            outcome_col=cfg["outcome_col"],
            treatment_time=treatment_time,
            treatment_col=cfg.get("treatment_col"),
            covariates=cfg.get("covariates", [])
        )
    except Exception as e:
        return {"error": str(e)}

def adaptive_storytelling_phd(df: pd.DataFrame, audience_col: str = None,
                             roi_col: str = None, effect_col: str = None,
                             method_col: str = None, driver_col: str = None,
                             assumptions_col: str = None, top_k=3):
    """
    PhD-like storytelling:
    - Validate columns exist; if missing, try to create proxies:
        * If roi_col missing but base_col & cost available: compute ROI proxy
        * If effect_col missing but treatment & outcome: compute simple ATT per group and fill effect_col
    - Build multi-audience narratives (CEO: concise impact; Analyst: methods & robustness; Ops: actions)
    - Return DataFrame with 'story' + 'audience' (if given) and debug 'proxies' used
    """
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

def beyond_eda(df: pd.DataFrame, config: dict = None):
    """
    Master pipeline:
    - auto-detect config
    - attempt to run each module; when missing inputs, return hint candidates rather than crash
    - where possible, create proxies and fill result with debug info (PhD-style reasoning)
    """
    # safety: ensure DataFrame
    if not isinstance(df, pd.DataFrame):
        return {"error": "input must be a pandas DataFrame"}

    auto_cfg = auto_detect_config(df)
    cfg = {**auto_cfg, **(config or {})}
    results = {}
    # 1. PSM (try IPW + matching)
    try:
        if cfg.get("treatment_col") and cfg.get("outcome_col") and cfg.get("covariates"):
            results["psm"] = causal_psm(df, treatment_col=cfg["treatment_col"], covariates=cfg.get("covariates",[]), outcome_col=cfg["outcome_col"], ipw=True)
        else:
            candidates = [c for c in df.columns if df[c].nunique(dropna=True)==2]
            results["psm"] = with_hint("missing config for PSM", candidates=candidates)
    except Exception as e:
        results["psm"] = {"error": str(e)}

    # 2. DiD
    try:
        results["did"] = causal_did_safe(df, cfg)
    except Exception as e:
        results["did"] = {"error": str(e)}

    # 3. IV
    try:
        if cfg.get("instrument_col") and cfg.get("treatment_col") and cfg.get("outcome_col"):
            results["iv"] = causal_iv_2sls(df, outcome=cfg["outcome_col"], treatment=cfg["treatment_col"], instrument=cfg["instrument_col"], covariates=cfg.get("covariates",[]))
        else:
            results["iv"] = with_hint("missing config for IV", candidates=[cfg.get("instrument_col")])
    except Exception as e:
        results["iv"] = {"error": str(e)}

    # 4. Robustness (bootstrap + drop-tops + rolling if time exists)
    try:
        if cfg.get("treatment_col") and cfg.get("outcome_col") and cfg.get("covariates"):
            results["robustness"] = robustness_checks(
                df,
                func=lambda d, **kw: causal_psm(d, treatment_col=cfg["treatment_col"], covariates=cfg.get("covariates",[]), outcome_col=cfg["outcome_col"], ipw=True),
                drop_top_percent=[0.01, 0.05],
                subgroup_cols=[cfg["treatment_col"]] if cfg.get("treatment_col") in df.columns else None,
                n_bootstrap=5,
                time_col=cfg.get("time_col")
            )
        else:
            results["robustness"] = with_hint("missing config for Robustness", candidates=[cfg.get("treatment_col"), cfg.get("outcome_col")])
    except Exception as e:
        results["robustness"] = {"error": str(e)}

    # 5. Cohort
    try:
        if cfg.get("user_col") and cfg.get("time_col"):
            results["cohort"] = cohort_retention(df, user_col=cfg["user_col"], date_col=cfg["time_col"], period=cfg.get("period","M"))
        else:
            results["cohort"] = with_hint("missing config for Cohort", candidates=[cfg.get("user_col"), cfg.get("time_col")])
    except Exception as e:
        results["cohort"] = {"error": str(e)}

    # 6. Survival
    try:
        if cfg.get("duration_col") and cfg.get("event_col"):
            results["survival"] = survival_km_cox(df, duration_col=cfg["duration_col"], event_col=cfg["event_col"], covariates=cfg.get("covariates",[]))
        else:
            results["survival"] = with_hint("missing config for Survival", candidates=[cfg.get("duration_col"), cfg.get("event_col")])
    except Exception as e:
        results["survival"] = {"error": str(e)}

    # 7. Nonparam
    try:
        if cfg.get("group_col") and cfg.get("outcome_col"):
            results["nonparam"] = nonparametric_tests(df, group_col=cfg["group_col"], value_col=cfg["outcome_col"])
        else:
            results["nonparam"] = with_hint("missing config for Nonparametric tests", candidates=[cfg.get("group_col"), cfg.get("outcome_col")])
    except Exception as e:
        results["nonparam"] = {"error": str(e)}

    # 8. Effect->business (try to compute proxies if missing)
    try:
        if cfg.get("effect_col") and cfg.get("base_col") and cfg["effect_col"] in df.columns and cfg["base_col"] in df.columns:
            results["effect"] = effect_to_business(df, effect_col=cfg["effect_col"], base_col=cfg["base_col"], scale=cfg.get("scale",1.0))
        else:
            # attempt simple auto-proxy: if treatment+outcome exist compute relative ATT
            if cfg.get("treatment_col") in df.columns and cfg.get("outcome_col") in df.columns:
                tmp = causal_psm(df, treatment_col=cfg["treatment_col"], covariates=cfg.get("covariates",[]), outcome_col=cfg["outcome_col"], ipw=True)
                results["effect"] = {"note": "effect_col/base_col missing; providing ATT/IPW proxy", "proxy": tmp}
            else:
                results["effect"] = with_hint("missing config for Effect-to-business", candidates=[cfg.get("effect_col"), cfg.get("base_col")])
    except Exception as e:
        results["effect"] = {"error": str(e)}

    # 9. Root cause (store provided or empty)
    try:
        results["root_cause"] = root_cause_tree(problem=cfg.get("problem","Undefined problem"), factors=cfg.get("factors",{}))
    except Exception as e:
        results["root_cause"] = {"error": str(e)}

    # 10. Counterintuitive
    try:
        if cfg.get("metric_col") and cfg.get("value_col") and cfg.get("expected_sign_col"):
            results["counterintuitive"] = detect_counterintuitive(df, metric_col=cfg["metric_col"], value_col=cfg["value_col"], expected_sign_col=cfg["expected_sign_col"])
        else:
            results["counterintuitive"] = with_hint("missing config for Counterintuitive", candidates=[cfg.get("metric_col"), cfg.get("value_col")])
    except Exception as e:
        results["counterintuitive"] = {"error": str(e)}

    # 11. Scenario
    try:
        results["scenario"] = scenario_simulation(df, change_dict=cfg.get("change_dict"))
    except Exception as e:
        results["scenario"] = {"error": str(e)}

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

    return {"config": cfg, "results": results}
