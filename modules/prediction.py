import os
from fastapi import  UploadFile, HTTPException
import pandas as pd
from io import BytesIO
import json
from typing import List, Optional, Tuple, Dict, Any, Union
from sklearn.model_selection import train_test_split
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, root_mean_squared_error, silhouette_score

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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "pipelineAutoML.json")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
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
                for dayfirst in [True, False]:
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)
                        if parsed.notna().mean() > 0.5:
                            detected[col] = "time_series"
                            break  # thoát ngay khi parse được
                    except Exception:
                        continue
                if detected.get(col) == "time_series":
                    continue

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

def auto_detect_target(df: pd.DataFrame, detected_types: Dict[str, str]) -> List[str]:
    candidate_cols = [c for c in df.columns if "cluster" not in c.lower()]
    
    numeric_cols = [c for c in candidate_cols if detected_types.get(c) == "numeric"]
    categorical_cols = [c for c in candidate_cols if detected_types.get(c) == "categorical"]

    targets = []

    # 1. Numeric liên tục (nhiều giá trị) → regression target
    for c in numeric_cols:
        if df[c].nunique() > 20:
            targets.append(c)

    # 2. Categorical có số lớp nhỏ (2–10) → classification target
    for c in categorical_cols:
        nunique = df[c].nunique()
        if 2 <= nunique <= 10:
            targets.append(c)

    # 3. Trường hợp chỉ có 1 cột numeric → có thể là target
    if len(numeric_cols) == 1 and numeric_cols[0] not in targets:
        targets.append(numeric_cols[0])

    # 4. Trường hợp chỉ có 1 categorical với số lớp nhỏ → có thể là target
    if len(categorical_cols) == 1:
        c = categorical_cols[0]
        nunique = df[c].nunique()
        if 2 <= nunique <= 10 and c not in targets:
            targets.append(c)

    return targets

def select_models_for_training(
    df: pd.DataFrame,
    detected_types: Dict[str, str],
    model_config: Dict[str, Any],
    target_col: Union[str, List[str], None] = None
) -> List[Dict[str, Any]]:

    if not target_col:
        target_col = auto_detect_target(df, detected_types)

    if isinstance(target_col, str):
        target_cols = [target_col]
    elif isinstance(target_col, list):
        target_cols = target_col
    else:
        target_cols = []

    results = []

    for t_col in target_cols:
        # --- 1. Xác định loại bài toán ---
        if t_col and t_col in df.columns:
            y = df[t_col]
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
        n_features = df.shape[1] - (1 if t_col else 0)
        dataset_size = df.shape[0]
        context = {
            "problem_type": problem_type,
            "n_features": int(n_features),
            "dataset_size": int(dataset_size),
            "has_imbalanced_target": False,
            "interpretability_required": False,
            "has_seasonality": "time_series" in detected_types.values(),
            "has_lag_features": any("lag" in c for c in df.columns),
            "contains_text_features": any(v == "text" for v in detected_types.values()),
            "contains_categorical_features": any(v == "categorical" for v in detected_types.values()),
            "n_unique_categories": max(
                [df[c].nunique() for c, t in detected_types.items() if t == "categorical"],
                default=0,
            ),
            "has_arbitrary_shapes": False,
        }

        # --- 3. Check imbalance ---
        if t_col and problem_type == "classification":
            y_counts = df[t_col].value_counts(normalize=True)
            if y_counts.min() < 0.2:
                context["has_imbalanced_target"] = True

        # --- 4. Rule-based auto-select ---
        selected_models, justification = None, None
        auto_select_cfg = model_config.get("auto_select", {})
        justification_cfg = auto_select_cfg.get("justification", {})
        rules = justification_cfg.get("rules", [])

        if auto_select_cfg.get("enabled", False):
            for rule in rules:
                try:
                    safe_globals = {
                        "__builtins__": {},
                        "true": True, "false": False,
                        "True": True, "False": False,
                        "None": None,
                    }
                    result = eval(rule["condition"], safe_globals, context)
                    if result:
                        selected_models = rule.get("selected_models", ["RandomForest"])
                        justification = rule.get("justification", "Rule-based selection.")
                        break
                except Exception as e:
                    print(f"⚠️ Rule eval error: {rule.get('condition')} → {e}")

        # --- 5. Fallback ---
        if not selected_models:
            model_sel_cfg = model_config.get("model_selection", {})
            pipeline_defaults = model_sel_cfg.get("problem_type", {})

            if problem_type in pipeline_defaults and pipeline_defaults[problem_type]:
                selected_models = pipeline_defaults[problem_type]
                justification = f"Không rule nào match → chọn toàn bộ model mặc định cho {problem_type} từ pipeline."
            elif "auto_select" in model_sel_cfg:
                fallback_cfg = (
                    model_sel_cfg["auto_select"].get("justification", {}).get("fallback", {})
                )
                if fallback_cfg:
                    selected_models = fallback_cfg.get("selected_models", ["RandomForest"])
                    justification = fallback_cfg.get(
                        "justification",
                        f"Không có rule match và pipeline không định nghĩa {problem_type} → dùng fallback."
                    )

        if not selected_models:
            if problem_type in ["classification", "regression"]:
                selected_models = ["RandomForest"]
            elif problem_type == "clustering":
                selected_models = ["KMeans"]
            elif problem_type == "time_series":
                selected_models = ["Prophet"]
            else:
                selected_models = ["RandomForest"]
            justification = f"⚠️ Chưa có cấu hình pipeline/fallback → tạm hardcode {selected_models[0]} (cần cập nhật config)."

        # Lưu kết quả cho target này
        results.append({
            "target_col": t_col,
            "problem_type": problem_type,
            "selected_models": selected_models,
            "justification": justification,
            "context": context,
        })

    return results

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
    study.optimize(objective, n_trials=1,
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

# def train_models(df, model_selection_log, training_config):
#     target_col = model_selection_log.get("target_col")
#     problem_type = model_selection_log.get("problem_type")
#     model_candidates = model_selection_log.get("selected_models", [])

#     if target_col and target_col in df.columns:
#         X = df.drop(columns=target_col)
#         y = df[target_col]
#     else:
#         # Clustering không có target
#         X, y = df, None

#     split_cfg = training_config.get("train_test_split", {})
#     if problem_type != "clustering" and target_col:
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y,
#             test_size=split_cfg.get("test_size", 0.2),
#             random_state=split_cfg.get("random_state", 42),
#             stratify=y if problem_type == "classification" else None
#         )
#     else:
#         X_train, X_test, y_train, y_test = X, None, y, None

#     results = {}
#     for model_name in model_candidates:
#         print(f"Training with Optuna: {model_name}")
#         model_cls = get_model_class(model_name, problem_type)
#         if not model_cls:
#             print(f"⚠️ Model {model_name} chưa được hỗ trợ")
#             continue

#         # Optuna tuning
#         best_params = {}
#         if problem_type in ["classification", "regression"]:
#             try:
#                 best_params = production_phase_tuning(
#                     model_cls, X_train, y_train, problem_type, training_config
#                 )
#             except Exception as e:
#                 print(f"⚠️ Optuna tuning failed for {model_name}: {e}")

#         # Train final model
#         try:
#             model = model_cls(**best_params)
#         except TypeError:
#             model = model_cls()

#         try:
#             if problem_type != "clustering":
#                 model.fit(X_train, y_train)
#             else:
#                 model.fit(X)
#         except Exception as e:
#             print(f"⚠️ Training failed for {model_name}: {e}")
#             continue

#         # Evaluate test_score
#         test_score, preds = None, None
#         try:
#             if problem_type == "classification":
#                 preds = model.predict(X_test)
#                 test_score = accuracy_score(y_test, preds)

#             elif problem_type == "regression":
#                 preds = model.predict(X_test)
#                 test_score = root_mean_squared_error(y_test, preds)

#             elif problem_type == "clustering":
#                 if hasattr(model, "labels_"):
#                     preds = model.labels_
#                     if X.shape[0] > 1 and X.shape[0] > X.shape[1]:
#                         test_score = silhouette_score(X, preds)

#             elif problem_type == "time_series":
#                 if X_test is not None:
#                     preds = model.predict(len(X_test))
#                     test_score = mean_squared_error(y_test, preds) ** 0.5

#             elif problem_type == "text":
#                 preds = model.predict(X_test)
#                 test_score = f1_score(y_test, preds, average="weighted")
#         except Exception as e:
#             print(f"⚠️ Evaluation failed for {model_name}: {e}")
#             test_score = None

#         # Save model
#         os.makedirs("models", exist_ok=True)
#         model_path = f"models/{model_name}.joblib"
#         try:
#             joblib.dump(model, model_path)
#         except Exception as e:
#             print(f"⚠️ Saving model {model_name} failed: {e}")

#         results[model_name] = {
#             "best_params": best_params,
#             "model_path": model_path,
#             "test_score": test_score,
#             "preds": preds.tolist() if preds is not None else None
#         }

#     return results

# def model_comparison(results: Dict[str, Any], model_selection_log: Dict[str, Any], training_config: Dict[str, Any]) -> Dict[str, Any]:
#     problem_type = model_selection_log.get("problem_type")
#     selection_metric = training_config.get("model_comparison", {}).get("selection_metric", "auto")

#     metric = "accuracy" if (selection_metric == "auto" and problem_type == "classification") else "rmse"

#     # Lọc ra những model có test_score hợp lệ
#     valid_results = {m: r for m, r in results.items() if r.get("test_score") is not None}
#     if not valid_results:
#         raise ValueError("No valid models with test_score found. Check training pipeline.")

#     if problem_type == "classification":
#         best_model = max(valid_results, key=lambda m: valid_results[m]["test_score"])
#     else:
#         best_model = min(valid_results, key=lambda m: valid_results[m]["test_score"])

#     best_score = valid_results[best_model]["test_score"]

#     # Explanation
#     explanation_cfg = training_config.get("model_comparison", {}).get("explanation_template", {})
#     explanation = explanation_cfg.get("winning_model", "{model_name} wins with {metric_name}={score}").format(
#         model_name=best_model,
#         metric_name=metric,
#         score=best_score,
#         alpha=training_config.get("model_comparison", {}).get("alpha", 0.05)
#     )
#     justification = model_selection_log.get("justification", "")

#     return {
#         "best_model": best_model,
#         "best_score": best_score,
#         "p_value": None,  # Wilcoxon bỏ qua tạm
#         "explanation": f"{explanation}\n→ {justification}"
#     }

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
    # training_results = []
    # res = train_models(
    #     df_fs,
    #     model_selection_log=model_selection_log,
    #     training_config=pipeline_config["model_training"]
    # )

    # # --- Step 7: Model comparison ---
    # comparison = model_comparison(
    #     res,
    #     model_selection_log=model_selection_log,
    #     training_config=pipeline_config["model_training"]
    # )

    # training_results = [{
    #     "target_col": model_selection_log["target_col"],
    #     "problem_type": model_selection_log["problem_type"],
    #     "results": res,
    #     "comparison": comparison
    # }]

    # --- Final output ---
    return {
        "detected_types": updated_types,
        "preprocessing_log": preprocessing_log,
        "feature_engineering_log": fe_log,
        "feature_selection_log": fs_log,
        "model_selection_log": model_selection_log,
        # "training_results": training_results,
        # "processed_data_preview": df_fs.head(5).to_dict(orient="records"),
    }


##=================
# def powerset(iterable):
#     """Sinh tất cả tổ hợp con không rỗng"""
#     s = list(iterable)
#     return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

# def auto_detect_target(df: pd.DataFrame, detected_types: Dict[str, str]) -> List[Union[str, List[str]]]:
#     candidate_cols = [
#         c for c in df.columns 
#         if "cluster" not in c.lower() and "id" not in c.lower()
#     ]

#     # Sinh powerset
#     multi_candidates = [list(x) for x in powerset(candidate_cols)]
#     return multi_candidates

# def select_models_for_training(
#     df: pd.DataFrame,
#     detected_types: Dict[str, str],
#     model_config: Dict[str, Any],
#     target_col: str = None
# ) -> Dict[str, Any]:
#     if not target_col:
#         target_candidates = auto_detect_target(df, detected_types)
#         print(f"Target cols {target_candidates}")
#     else:
#         target_candidates = [target_col] if isinstance(target_col, str) else [target_col]

#     results = []
#     if not target_candidates:
#         context = {
#             "problem_type": "clustering",
#             "n_features": df.shape[1],
#             "dataset_size": df.shape[0],
#             "contains_text_features": any(v == "text" for v in detected_types.values()),
#             "contains_categorical_features": any(v == "categorical" for v in detected_types.values()),
#         }
#         results.append({
#             "target_col": None,
#             "problem_type": "clustering",
#             "selected_models": ["KMeans"],
#             "justification": "Không có target column → clustering.",
#             "context": context,
#         })
#         return results

#     for tgt in target_candidates:
#         # --- 1. Lấy target data ---
#         if isinstance(tgt, str):
#             y = df[tgt]
#         else:  # multi-target
#             y = df[tgt]

#     # --- 1. Xác định loại bài toán ---
#     if isinstance(tgt, list) and len(tgt) > 1:
#             problem_type = "multi_output"
#     else:
#         if pd.api.types.is_numeric_dtype(y):
#             if y.nunique() <= 20 and str(y.dtype).startswith("int"):
#                 problem_type = "classification"
#             else:
#                 problem_type = "regression"
#         else:
#             problem_type = "classification"

#     # --- 2. Tạo context dataset ---
#     n_features = df.shape[1] - (1 if target_col else 0)
#     dataset_size = df.shape[0]
#     context = {
#         "problem_type": problem_type,
#         "n_features": int(n_features),
#         "dataset_size": int(dataset_size),
#         "has_imbalanced_target": False,
#         "interpretability_required": False,
#         "has_seasonality": "time_series" in detected_types.values(),
#         "has_lag_features": any("lag" in c for c in df.columns),
#         "contains_text_features": any(v == "text" for v in detected_types.values()),
#         "contains_categorical_features": any(v == "categorical" for v in detected_types.values()),
#         "n_unique_categories": max(
#             [df[c].nunique() for c, t in detected_types.items() if t == "categorical"],
#             default=0,
#         ),
#         "has_arbitrary_shapes": False,
#     }

#     # --- 3. Check imbalance ---
#     if target_col and problem_type == "classification":
#         y_counts = df[target_col].value_counts(normalize=True)
#         if y_counts.min() < 0.2:
#             context["has_imbalanced_target"] = True

#     # --- 4. Rule-based auto-select ---
#     selected_models, justification = None, None
#     auto_select_cfg = model_config.get("auto_select", {})
#     justification_cfg = auto_select_cfg.get("justification", {})
#     rules = justification_cfg.get("rules", [])
   

#     print("AUTO_SELECT_CFG:", auto_select_cfg)
#     print("JUSTIFICATION_CFG:", justification_cfg)
#     print("RULES LOADED:", rules)

#     if auto_select_cfg.get("enabled", False):
#         print("CONTEXT:", context)
#         for rule in rules:
#             try:
#                 safe_globals = {
#                     "__builtins__": {},
#                     "true": True, "false": False,
#                     "True": True, "False": False,
#                     "None": None,
#                 }
#                 result = eval(rule["condition"], safe_globals, context)
#                 print(f"CHECK RULE: {rule['condition']} → {result}")
#                 print("RULE:", rule["condition"])
#                 print("EVAL CONTEXT:", context)
#                 print("RESULT:", eval(rule["condition"], safe_globals, context))

#                 if result:
#                     selected_models = rule.get("selected_models", ["RandomForest"])
#                     justification = rule.get("justification", "Rule-based selection.")
#                     break
#             except Exception as e:
#                 print(f"⚠️ Rule eval error: {rule.get('condition')} → {e}")


#     # --- 5. Fallback ---
#     if not selected_models:
#         model_sel_cfg = model_config.get("model_selection", {})
#         pipeline_defaults = model_sel_cfg.get("problem_type", {})

#         # Ưu tiên 1: pipeline định nghĩa model cho problem_type
#         if problem_type in pipeline_defaults and pipeline_defaults[problem_type]:
#             selected_models = pipeline_defaults[problem_type]
#             justification = (
#                 f"Không rule nào match → chọn toàn bộ model mặc định cho {problem_type} từ pipeline."
#             )

#         # Ưu tiên 2: fallback trong config
#         elif "auto_select" in model_sel_cfg:
#             fallback_cfg = (
#                 model_sel_cfg["auto_select"].get("justification", {}).get("fallback", {})
#             )
#             if fallback_cfg:
#                 selected_models = fallback_cfg.get("selected_models", ["RandomForest"])
#                 justification = fallback_cfg.get(
#                     "justification",
#                     f"Không có rule match và pipeline không định nghĩa {problem_type} → dùng fallback."
#                 )

#         if not selected_models:
#             if problem_type in ["classification", "regression"]:
#                 selected_models = ["RandomForest"]
#             elif problem_type == "clustering":
#                 selected_models = ["KMeans"]
#             elif problem_type == "time_series":
#                 selected_models = ["Prophet"]
#             else:
#                 selected_models = ["RandomForest"]
#             justification = (
#                 f"⚠️ Chưa có cấu hình pipeline/fallback → tạm hardcode {selected_models[0]} (cần cập nhật config)."
#             )

#     return {
#         "target_col": target_col,
#         "problem_type": problem_type,
#         "selected_models": selected_models,
#         "justification": justification,
#         "context": context,
#     }
