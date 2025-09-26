def calcular_nulos(df):
    nulos = df['isFraud'].isna().sum()
    print(f"Cantidad de valores nulos en isFraud: {nulos}")

def fraud_licit_ratio(df):
    # Calcular proporción de clases
    conteo = df['isFraud'].value_counts()  # 0 = no fraude, 1 = fraude
    porcentaje = df['isFraud'].value_counts(normalize=True) * 100
    
    # Mostrar resumen
    print("\nDistribución de isFraud:")
    for clase in conteo.index:
        print(f"Clase {clase}: {conteo[clase]} registros ({porcentaje[clase]:.2f}%)")

def split_by_suffix(df):
    id_cols = [col for col in df.columns if col.startswith('id_')]
    card_cols = [col for col in df.columns if col.startswith('card')]
    v_cols = [col for col in df.columns if col.startswith('V')]

    return id_cols, card_cols, v_cols

def nan_ratio(df,df_description):
    # Calculate % of NaN values per column
    nan_ratio = df.isna().mean().sort_values(ascending=False)

    print(f"Top 10 columns with highest NaN ratio in {df_description}:")
    print(nan_ratio.head(10))

    # Select columns where more than 50% are NaN
    cols_majority_nan  = nan_ratio[nan_ratio > 0.5].index.tolist()

    print(f"\nColumns with >50% NaN in {df_description}:", cols_majority_nan)
