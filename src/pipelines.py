import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from joblib import Memory
memory = Memory("cache_dir", verbose=0)

from feature_engineering import build_preprocessor

# === Utilidad para separar columnas ===
def split_features(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return cat_cols, num_cols

# === Pipeline para Regresión Logística ===
def logistic_pipeline(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", penalty="l2", n_jobs=-1))
    ])

# === Pipeline para SVM ===
def svm_pipeline(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ])

# === Pipeline para XGBoost ===
def xgboost_pipeline(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))  # XGBoost soporta NaN, pero imputar da más estabilidad
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Ajustar según balance de clases
            eval_metric="auc"
        ))
    ])

# === Pipeline v2 para Regresión Logística ===
def logistic_pipeline_v2(df, target_col="isFraud"):
    """
    Pipeline con feature_engineering para Regresión Logística.
    Incluye:
    - Imputación por mediana (numéricas) y moda (categóricas)
    - Reducción de colinealidad numérica
    - Escalado estándar
    - OneHot (categóricas de baja cardinalidad)
    - FrequencyEncoder o TargetMeanEncoder (alta cardinalidad)
    """
    preprocessor, _ = build_preprocessor(
        df=df,
        target_col=target_col,
        low_card_max_unique=10,
        high_card_strategy="frequency",   # seguro para CV
        numeric_scale=True,
        numeric_scaler="standard",
        reduce_corr=True,
        corr_threshold=0.90,
        add_numeric_missing_flags=True
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            penalty="l2",
            n_jobs=-1
        ))
    ], memory=memory)


# === Pipeline v2 para SVM ===
def svm_pipeline_v2(df, target_col="isFraud"):
    """
    Pipeline con feature_engineering para SVM.
    Similar al de regresión logística, requiere escalado y reducción de colinealidad.
    """
    preprocessor, _ = build_preprocessor(
        df=df,
        target_col=target_col,
        low_card_max_unique=20,
        high_card_strategy="frequency",
        numeric_scale=True,
        numeric_scaler="standard",
        reduce_corr=True,
        corr_threshold=0.90,
        add_numeric_missing_flags=True
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True
        ))
    ], memory=memory)


# === Pipeline v2 para XGBoost ===
def xgboost_pipeline_v2(df, target_col="isFraud"):
    """
    Pipeline con feature_engineering para XGBoost.
    - Numéricas imputadas por mediana
    - Opcional reducción de colinealidad (puede desactivarse)
    - Categóricas bajas → OneHot
    - Categóricas altas → FrequencyEncoder (por defecto)
    Nota: XGBoost maneja NaN de forma nativa, pero imputamos para mayor estabilidad.
    """
    preprocessor, _ = build_preprocessor(
        df=df,
        target_col=target_col,
        low_card_max_unique=20,
        high_card_strategy="frequency",  # también puede usarse "target"
        numeric_scale=False,             # XGB no requiere escalado
        reduce_corr=False,               # opcional: activar para datasets enormes
        add_numeric_missing_flags=True
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",      # optimizado para grandes datasets
            n_jobs=-1,
            eval_metric="auc"
        ))
    ], memory=memory)