import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def calcular_nulos(df):
    nulos = df['isFraud'].isna().sum()
    print(f"Cantidad de valores nulos en isFraud: {nulos}")

def fraud_licit_ratio(df):
    conteo = df['isFraud'].value_counts()  # 0 = no fraude, 1 = fraude
    porcentaje = df['isFraud'].value_counts(normalize=True) * 100
    
    print("\nDistribución de isFraud:")
    for clase in conteo.index:
        print(f"Clase {clase}: {conteo[clase]} registros ({porcentaje[clase]:.2f}%)")

def split_by_suffix(df):
    id_cols = [col for col in df.columns if col.startswith('id_')]
    card_cols = [col for col in df.columns if col.startswith('card')]
    v_cols = [col for col in df.columns if col.startswith('V')]

    return id_cols, card_cols, v_cols

def nan_ratio(df,df_description):
    nan_ratio = df.isna().mean().sort_values(ascending=False)

    print(f"Top 10 columns with highest NaN ratio in {df_description}:")
    print(nan_ratio.head(10))

    cols_majority_nan  = nan_ratio[nan_ratio > 0.5].index.tolist()

    print(f"\nColumns with >50% NaN in {df_description}:", cols_majority_nan)
    
def visualize_missing_values(dataset, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.isna(), cbar=False, cmap='viridis')
    plt.title(title)
    plt.show()

def missing_audit(df, target_col, drop_threshold=0.7, lift_threshold=1.05):
    """
    Analiza columnas con valores faltantes en relación al target binario (ej. fraude/no fraude).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo
    target_col : str
        Nombre de la columna target (binaria: 0/1)
    drop_threshold : float
        Porcentaje mínimo de NaN para considerar la columna como candidata a eliminar
    lift_threshold : float
        Umbral para decidir si los NaN aportan señal (lift significativo)

    Retorna
    -------
    result : pd.DataFrame
        Métricas por columna
    drop_candidates : list
        Listado de columnas que podrían eliminarse
    """
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    def compute_stats(col):
        miss = X[col].isna()
        p_miss = miss.mean()
        if p_miss in (0, 1):
            return (col, p_miss, np.nan, np.nan, np.nan)
        fr_miss = y[miss].mean()
        fr_present = y[~miss].mean()
        lift = (fr_miss + 1e-9) / (fr_present + 1e-9)
        return (col, p_miss, fr_miss, fr_present, lift)

    stats = [compute_stats(c) for c in X.columns]
    result = pd.DataFrame(stats, columns=[
        "col", "pct_missing", "fraud_rate_if_missing",
        "fraud_rate_if_present", "lift_missing"
    ])

    # Identificar columnas candidatas a eliminar
    drop_candidates = result.query(
        "pct_missing >= @drop_threshold and (lift_missing.isna() or abs(lift_missing - 1) < (@lift_threshold - 1))"
    )["col"].tolist()

    return result.sort_values(["pct_missing", "lift_missing"], ascending=[False, False]), drop_candidates
