"""
Splits dengue_treated.parquet into train (2020–2023) and test (2024) sets
using temporal stratification by `ano`. Saves X/y parquet files and config.json
to data/features/baseline/.
"""
import json
import os
from datetime import date

import pandas as pd

INPUT_PATH = "data/processed/dengue_treated.parquet"
OUTPUT_DIR = "data/features/baseline"

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS  = [2024]
TARGET_COL  = "target"
SPLIT_COL   = "year"


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
    # deve ser dropada antes do fit — não é feature
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COL]

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
        "created_at":  str(date.today()),
    }

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nArtefatos salvos em: {OUTPUT_DIR}/")
    print("  X_train.parquet | y_train.parquet")
    print("  X_test.parquet  | y_test.parquet")
    print("  config.json")


if __name__ == "__main__":
    main()
