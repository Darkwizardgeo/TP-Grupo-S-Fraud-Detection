import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
from joblib import Memory

memory = Memory("cache_dir", verbose=0)

def split_features(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return cat_cols, num_cols

def reduce_dataset_complexity_pipeline(df):
    cat_cols, num_cols = split_features(df)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var_th", VarianceThreshold(threshold=0.01)),
        ("scaler", StandardScaler())
    ])

    cat_low = [c for c in cat_cols if df[c].nunique() <= 10]
    cat_high = [c for c in cat_cols if df[c].nunique() > 10]

    cat_low_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    cat_high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("freq", OneHotEncoder(handle_unknown="ignore", max_categories=50))  # también puedes usar FrequencyEncoder
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat_low", cat_low_pipe, cat_low),
        ("cat_high", cat_high_pipe, cat_high)
    ])

    # === Selección con XGBoost (reduce dimensionalidad antes de LogReg/SVM) ===
    selector = SelectFromModel(
        xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="auc"
        ),
        threshold="1.5*mean"
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("feature_sel", selector)  # selecciona features más útiles
    ])

# === Pipeline para Regresión Logística ===
def logistic_pipeline(df, target_col="isFraud"):
    pipeline_reduced = reduce_dataset_complexity_pipeline(df.drop(columns=[target_col]))
    return Pipeline([
        ("reduce", pipeline_reduced),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            penalty="l2",
            solver="saga",
            C=0.05,
            n_jobs=-1
        ))
    ], memory=memory)

# === Pipeline para SVM ===
def svm_pipeline(df, target_col="isFraud"):
    pipeline_reduced = reduce_dataset_complexity_pipeline(df.drop(columns=[target_col]))
    svm_base = LinearSVC(class_weight="balanced", max_iter=5000)

    return Pipeline([
        ("reduce", pipeline_reduced),
        ("model", CalibratedClassifierCV(svm_base, method="sigmoid", cv=3))  # para tener predict_proba
    ], memory=memory)
    # return Pipeline([
    #     ("preprocess", preprocessor),
    #     ("model", SVC(kernel="rbf", class_weight="balanced", probability=True))
    # ])

# === Pipeline para XGBoost ===
def xgboost_pipeline_reduced(df, target_col="isFraud"):
    pipeline_reduced = reduce_dataset_complexity_pipeline(df.drop(columns=[target_col]))

    return Pipeline([
        ("reduce", pipeline_reduced),
        ("model", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Ajustar según balance de clases
            eval_metric="auc"
        ))
    ], memory=memory)

# === Pipeline v2 para XGBoost ===
def xgboost_pipeline_raw():
    return Pipeline([
        ("model", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",      # optimizado para grandes datasets
            n_jobs=-1,
            eval_metric="auc"
        ))
    ], memory=memory)