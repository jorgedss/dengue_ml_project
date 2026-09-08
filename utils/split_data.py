"""
Splits dengue_treated.parquet into train (2020–2023) and test (2024) sets
using temporal stratification by `ano`. Saves X/y parquet files and config.json
to data/features/baseline/.
"""
import json
import os
from datetime import date

import pandas as pd

try:  # execucao via pacote (python main.py)
    from utils.clean_data import METADATA_COLS
except ImportError:  # execucao direta (python utils/split_data.py)
    from clean_data import METADATA_COLS

INPUT_PATH = "data/processed/dengue_treated.parquet"
OUTPUT_DIR = "data/features/baseline"

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS  = [2024]
TARGET_COL  = "target"
SPLIT_COL   = "year"

# Metadados: saem em meta_{train,test}.parquet, alinhados linha a linha com
# X/y, e nunca em X. Os notebooks montam o ColumnTransformer com
# remainder='passthrough' e so removem `year` antes do fit, entao qualquer
# coluna extra deixada em X iria direto para o estimador.
NON_FEATURE_COLS = list(METADATA_COLS)


def assert_no_metadata_in_X(X: pd.DataFrame, nome: str) -> None:
    """Falha alto se algum metadado escapar para X."""
    proibidas = [c for c in X.columns if c in NON_FEATURE_COLS]
    assert not proibidas, f"colunas de metadado presentes em {nome}: {proibidas}"

    # Pega tambem colunas de data futuras que ninguem lembrou de listar.
    datas = [
        c for c in X.columns
        if c.startswith("DT_") or pd.api.types.is_datetime64_any_dtype(X[c])
    ]
    assert not datas, f"colunas de data presentes em {nome}: {datas}"


def main():
    print("=" * 55)
    print("Gerando split treino/teste temporal")
    print("=" * 55)

    print(f"\nLendo {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
    print(f"Total: {len(df):,} registros | Colunas: {len(df.columns)}")

    if SPLIT_COL not in df.columns:
        raise ValueError(f"Coluna '{SPLIT_COL}' não encontrada. Execute treat_data.py.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna '{TARGET_COL}' não encontrada. Execute treat_data.py.")

    train = df[df[SPLIT_COL].isin(TRAIN_YEARS)].copy()
    test  = df[df[SPLIT_COL].isin(TEST_YEARS)].copy()

    # year é mantido em X para permitir expanding window nos notebooks de modelo
    # deve ser dropada antes do fit - não é feature
    excluidas = {TARGET_COL, *NON_FEATURE_COLS}
    feature_cols = [c for c in df.columns if c not in excluidas]
    meta_cols    = [c for c in NON_FEATURE_COLS if c in df.columns]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COL]

    meta_train = train[meta_cols]
    meta_test  = test[meta_cols]

    assert_no_metadata_in_X(X_train, "X_train")
    assert_no_metadata_in_X(X_test, "X_test")

    print(f"\nTreino ({TRAIN_YEARS}): {len(X_train):,} registros")
    print(f"  Óbitos: {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)")
    print(f"\nTeste  ({TEST_YEARS}): {len(X_test):,} registros")
    print(f"  Óbitos: {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)")
    print(f"\nFeatures: {len(feature_cols)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_parquet(os.path.join(OUTPUT_DIR, "X_train.parquet"), index=False, engine="pyarrow")
    y_train.to_frame().to_parquet(os.path.join(OUTPUT_DIR, "y_train.parquet"), index=False, engine="pyarrow")
    X_test.to_parquet(os.path.join(OUTPUT_DIR, "X_test.parquet"),  index=False, engine="pyarrow")
    y_test.to_frame().to_parquet(os.path.join(OUTPUT_DIR, "y_test.parquet"),  index=False, engine="pyarrow")

    meta_train.to_parquet(os.path.join(OUTPUT_DIR, "meta_train.parquet"), index=False, engine="pyarrow")
    meta_test.to_parquet(os.path.join(OUTPUT_DIR, "meta_test.parquet"),  index=False, engine="pyarrow")

    config = {
        "strategy":    "temporal",
        "train_years": TRAIN_YEARS,
        "test_years":  TEST_YEARS,
        "train_size":  len(X_train),
        "test_size":   len(X_test),
        "target_col":  TARGET_COL,
        "split_col":   SPLIT_COL,
        "features":    feature_cols,
        "n_features":  len(feature_cols),
        "non_feature_cols": meta_cols,
        "train_events": int(y_train.sum()),
        "test_events":  int(y_test.sum()),
        "created_at":  str(date.today()),
    }

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nMetadados fora de X ({len(meta_cols)}): {', '.join(meta_cols)}")

    print(f"\nArtefatos salvos em: {OUTPUT_DIR}/")
    print("  X_train.parquet | y_train.parquet | meta_train.parquet")
    print("  X_test.parquet  | y_test.parquet  | meta_test.parquet")
    print("  config.json")


if __name__ == "__main__":
    main()
