# feature_engineering.py
# ------------------------------------------------------------
# Utilidades de ingeniería de características para datasets grandes
# con mezcla de variables numéricas y categóricas, NaNs y alta colinealidad.
# ------------------------------------------------------------

from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, MissingIndicator

# ============================================================
# --------------- Detectores y Transformadores ----------------
# ============================================================

class CorrelationReducer(BaseEstimator, TransformerMixin):
    """
    Elimina columnas numéricas altamente correlacionadas (|ρ| > threshold).
    Mantiene la primera que aparece y elimina las subsiguientes del triángulo superior.
    Devuelve un DataFrame con columnas reducidas para preservar nombres.
    """
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.keep_cols_: List[str] = []

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        if X_df.shape[1] <= 1:
            self.keep_cols_ = list(X_df.columns)
            return self

        corr = X_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_cols_ = [c for c in X_df.columns if c not in drop_cols]
        return self

    def transform(self, X):
        X_df = self._to_df(X)
        cols = [c for c in self.keep_cols_ if c in X_df.columns]
        return X_df[cols]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Codificación de alta cardinalidad por frecuencia relativa.
    Seguro, rápido y sin usar y (evita fuga de target).
    """
    def __init__(self):
        self.maps_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        self.columns_ = list(X_df.columns)
        n = len(X_df)
        for c in self.columns_:
            freq = X_df[c].astype("object").value_counts(dropna=False) / n
            self.maps_[c] = freq.to_dict()
        return self

    def transform(self, X):
        X_df = self._to_df(X)
        out = pd.DataFrame(index=X_df.index)
        for c in self.columns_:
            m = self.maps_[c]
            out[f"{c}__freq"] = X_df[c].astype("object").map(m).fillna(0.0).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}__freq" for c in self.columns_])

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Codificación media del target con suavizado para alta cardinalidad.
    Requiere y. Útil con modelos lineales/árboles; usar dentro de CV para evitar fuga.
    mean_enc = (n_i * mean_i + m * global_mean) / (n_i + m)
    """
    def __init__(self, m: float = 50.0):
        self.m = m
        self.global_mean_: float = 0.0
        self.maps_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X, y):
        X_df = self._to_df(X)
        self.columns_ = list(X_df.columns)
        y = pd.Series(y).astype(float)
        self.global_mean_ = float(y.mean())

        for c in self.columns_:
            g = pd.DataFrame({"x": X_df[c].astype("object"), "y": y})
            stats = g.groupby("x")["y"].agg(["mean", "count"])
            enc = (stats["count"] * stats["mean"] + self.m * self.global_mean_) / (stats["count"] + self.m)
            self.maps_[c] = enc.to_dict()

        return self

    def transform(self, X):
        X_df = self._to_df(X)
        out = pd.DataFrame(index=X_df.index)
        for c in self.columns_:
            m = self.maps_[c]
            out[f"{c}__tmean"] = X_df[c].astype("object").map(m).fillna(self.global_mean_).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}__tmean" for c in self.columns_])

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


# ============================================================
# --------------- Helper de separación de columnas -----------
# ============================================================

def split_features(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    low_card_max_unique: int = 20,
    id_like_threshold: float = 0.95,
    id_regex: str = r"(?:^|_)(id|uuid|guid|token|hash)(?:$|_)"
) -> Dict[str, List[str]]:
    """
    Separa columnas en numéricas, categóricas de baja/alta cardinalidad y "ID-like".
    """
    X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    n = len(X)
    cat_low, cat_high, id_like = [], [], []

    for c in cat_cols:
        nunique = X[c].nunique(dropna=False)
        unique_ratio = nunique / max(n, 1)

        if re.search(id_regex, c, flags=re.IGNORECASE) or unique_ratio >= id_like_threshold:
            id_like.append(c)
        elif nunique <= low_card_max_unique:
            cat_low.append(c)
        else:
            cat_high.append(c)

    return {
        "num_cols": num_cols,
        "cat_low": cat_low,
        "cat_high": cat_high,
        "id_like": id_like
    }


# ============================================================
# ------------------- Preprocesadores ------------------------
# ============================================================

def numeric_preprocessor(
    scale: bool = True,
    scaler: str = "standard",
    reduce_corr: bool = True,
    corr_threshold: float = 0.90,
    add_missing_flags: bool = True
) -> Pipeline:
    """
    Pipeline para variables NUMÉRICAS:
    - Imputación por mediana
    - (opcional) Reducción de colinealidad por correlación
    - (opcional) Escalado (standard o robust)
    - (opcional) Indicadores de missing
    """
    steps: List[Tuple[str, TransformerMixin]] = []
    steps.append(("imputer", SimpleImputer(strategy="median")))

    if reduce_corr:
        steps.append(("corr_reducer", CorrelationReducer(threshold=corr_threshold)))

    if scale:
        if scaler == "robust":
            steps.append(("scaler", RobustScaler()))
        else:
            steps.append(("scaler", StandardScaler()))

    pipe = Pipeline(steps)

    if add_missing_flags:
        # Nota: también podríamos usar SimpleImputer(add_indicator=True),
        # pero añadimos indicadores como rama paralela en el ColumnTransformer.
        pass

    return pipe


def categorical_low_preprocessor() -> Pipeline:
    """
    Pipeline para CATEGÓRICAS de BAJA cardinalidad:
    - Imputación por moda
    - One-Hot Encoder con manejo de categorías desconocidas
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None))
    ])


def categorical_high_preprocessor(strategy: str = "frequency", tmean_m: float = 50.0) -> Pipeline:
    """
    Pipeline para CATEGÓRICAS de ALTA cardinalidad:
    - Imputación por moda
    - Encoder de alta cardinalidad: 'frequency' (seguro) o 'target' (media suavizada)
    """
    if strategy not in ("frequency", "target"):
        raise ValueError("strategy debe ser 'frequency' o 'target'.")

    encoder = FrequencyEncoder() if strategy == "frequency" else TargetMeanEncoder(m=tmean_m)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])


def build_preprocessor(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    low_card_max_unique: int = 20,
    id_like_threshold: float = 0.95,
    high_card_strategy: str = "frequency",
    numeric_scale: bool = True,
    numeric_scaler: str = "standard",
    reduce_corr: bool = True,
    corr_threshold: float = 0.90,
    add_numeric_missing_flags: bool = True
) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    """
    Construye un ColumnTransformer listo para plug-and-play en pipelines de modelado.
    Devuelve (preprocessor, splits) donde splits describe las columnas usadas.
    """
    splits = split_features(
        df=df,
        target_col=target_col,
        low_card_max_unique=low_card_max_unique,
        id_like_threshold=id_like_threshold
    )
    num_cols = splits["num_cols"]
    cat_low = splits["cat_low"]
    cat_high = splits["cat_high"]
    id_like = splits["id_like"]  # típicamente se excluyen fuera del preprocesador

    transformers = []

    if num_cols:
        transformers.append(("num", numeric_preprocessor(
            scale=numeric_scale,
            scaler=numeric_scaler,
            reduce_corr=reduce_corr,
            corr_threshold=corr_threshold
        ), num_cols))

        if add_numeric_missing_flags:
            transformers.append(("num_missing", MissingIndicator(features="all"), num_cols))

    if cat_low:
        transformers.append(("cat_low", categorical_low_preprocessor(), cat_low))

    if cat_high:
        transformers.append(("cat_high", categorical_high_preprocessor(strategy=high_card_strategy), cat_high))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    return pre, splits