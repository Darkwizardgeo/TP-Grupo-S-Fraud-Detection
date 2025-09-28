# evaluate.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def evaluate_pipeline(pipeline, X, y, cv_splits=5, random_state=42):
    """
    Evalúa un pipeline de clasificación con Stratified K-Fold.

    Parámetros
    ----------
    pipeline : sklearn Pipeline
        Pipeline ya configurado (ej. logistic_pipeline, svm_pipeline, xgboost_pipeline)
    X : pd.DataFrame
        Features
    y : pd.Series o np.array
        Target binario (0/1, ej. isFraud)
    cv_splits : int
        Número de folds para Stratified K-Fold
    random_state : int
        Semilla para reproducibilidad

    Retorna
    -------
    results : dict
        Métricas promedio y desviación estándar
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    aucs, pr_aucs, f1s = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Predicciones de probabilidad y etiquetas
        y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(X_val)
        y_pred = pipeline.predict(X_val)

        # Calcular métricas
        aucs.append(roc_auc_score(y_val, y_proba))
        pr_aucs.append(average_precision_score(y_val, y_proba))
        f1s.append(f1_score(y_val, y_pred, pos_label=1))

    results = {
        "ROC-AUC": (np.mean(aucs), np.std(aucs)),
        "PR-AUC": (np.mean(pr_aucs), np.std(pr_aucs)),
        "F1-Score": (np.mean(f1s), np.std(f1s))
    }

    return results


def evaluate_multiple(pipelines, X, y, cv_splits=5):
    """
    Evalúa múltiples pipelines y devuelve resultados en un DataFrame.

    pipelines : dict
        Diccionario con nombre -> pipeline
    X : DataFrame
    y : Series

    Retorna
    -------
    pd.DataFrame
    """
    all_results = {}
    for name, pipe in pipelines.items():
        res = evaluate_pipeline(pipe, X, y, cv_splits=cv_splits)
        all_results[name] = {metric: f"{val:.4f} ± {std:.4f}" for metric, (val, std) in res.items()}
    return pd.DataFrame(all_results).T