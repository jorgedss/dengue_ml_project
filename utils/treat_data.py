"""
Applies all feature transformations to dengue_hospitalized.parquet:
builds the binary target, decodes age, extracts date features, and converts
SINAN 1/2 symptom encoding to binary 0/1. Output: data/processed/dengue_treated.parquet
"""
import os

import numpy as np
import pandas as pd
from pysus.preprocessing.decoders import decodifica_idade_SINAN

INPUT_PATH = "data/raw/dengue_hospitalized.parquet"
OUTPUT_PATH = "data/processed/dengue_treated.parquet"

# Columns that use SINAN 1=Sim / 2=Não / 9=Ignorado — remapped to 1 / 0 / NaN
BINARY_SINAN_COLS = [
    "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
    "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
    "LEUCOPENIA", "LACO", "DOR_RETRO", "DIABETES", "HEMATOLOG",
    "HEPATOPAT", "RENAL", "HIPERTENSA", "AUTO_IMUNE",
    # Sinais de alarme
    "ALRM_HIPOT", "ALRM_PLAQ", "ALRM_VOM", "ALRM_SANG", "ALRM_HEMAT",
    "ALRM_ABDOM", "ALRM_LETAR", "ALRM_HEPAT", "ALRM_LIQ",
    # Sinais de gravidade
    "GRAV_PULSO", "GRAV_CONV", "GRAV_ENCH", "GRAV_INSUF", "GRAV_TAQUI",
    "GRAV_EXTRE", "GRAV_HIPOT", "GRAV_HEMAT", "GRAV_MELEN", "GRAV_METRO",
    "GRAV_SANG", "GRAV_AST", "GRAV_MIOC", "GRAV_CONSC", "GRAV_ORGAO",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


def build_target(df: pd.DataFrame) -> pd.Series:
    """EVOLUCAO '1' (cura) → 0, '2' (óbito por dengue) → 1."""
    evolucao = df["EVOLUCAO"].astype(str).str.strip()
    target = pd.Series(np.nan, index=df.index, dtype="float32")
    target[evolucao == "1"] = 0
    target[evolucao == "2"] = 1
    return target


def remap_binary_sinan(series: pd.Series) -> pd.Series:
    """1=Sim → 1 | 2=Não → 0 | 9/outros → NaN."""
    s = _to_numeric(series)
    result = pd.Series(np.nan, index=series.index, dtype="float32")
    result[s == 1] = 1
    result[s == 2] = 0
    return result


def treat(df: pd.DataFrame) -> pd.DataFrame:
    # ── 1. Target ─────────────────────────────────────────────────────────────
    df["target"] = build_target(df)
    df = df.drop(columns=["EVOLUCAO"])

    # ── 2. Sintomas / alarmes / gravidade: 1/2/9 → 1/0/NA ────────────────────
    for col in BINARY_SINAN_COLS:
        if col in df.columns:
            df[col] = remap_binary_sinan(df[col])

    # ── 3. Decodificação de idade ──────────────────────────────────────────────
    if "NU_IDADE_N" in df.columns:
        df["age_years"] = decodifica_idade_SINAN(df["NU_IDADE_N"], "Y")
        df.loc[df["age_years"] > 120, "age_years"] = np.nan
        df = df.drop(columns=["NU_IDADE_N"])

    # ── 4. ano e epi_week derivados de SEM_PRI (formato YYYYWW) ──────────────
    if "SEM_PRI" in df.columns:
        sem = _to_numeric(df["SEM_PRI"])
        df["year"]      = (sem // 100).astype("Int16")
        df["epi_week"] = (sem % 100).astype("float32")
        df = df.drop(columns=["SEM_PRI"])

    # ── 5. Códigos ignorado → NA ──────────────────────────────────────────────
    if "CS_SEXO" in df.columns:
        df["CS_SEXO"] = df["CS_SEXO"].astype(str).str.strip()
        df.loc[df["CS_SEXO"] == "I", "CS_SEXO"] = np.nan

    for col in ["CS_GESTANT", "CS_RACA", "CS_ESCOL_N"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
            df.loc[df[col] == 9, col] = np.nan

    # ── 6. Strings vazias → NA ────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace("", np.nan)

    return df


def main():
    print("=" * 55)
    print("Tratando dengue_hospitalized.parquet")
    print("=" * 55)

    print(f"\nLendo {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
    print(f"Registros: {len(df):,} | Colunas: {len(df.columns)}")

    df = treat(df)

    n_cura = (df["target"] == 0).sum()
    n_obito = (df["target"] == 1).sum()
    pct_obito = n_obito / len(df) * 100

    print("\nApós tratamento:")
    print(f"  Total:     {len(df):,}")
    print(f"  Cura  (0): {n_cura:,}")
    print(f"  Óbito (1): {n_obito:,}  ({pct_obito:.2f}%)")
    print(f"  Colunas:   {len(df.columns)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nSalvando em: {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print("Concluído.")


if __name__ == "__main__":
    main()
