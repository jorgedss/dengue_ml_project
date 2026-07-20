"""
    Filters raw yearly parquet files to keep only hospitalized patients with a known outcome
    (EVOLUCAO 1 = recovery, 2 = dengue death) and consolidates all years into dengue_hospitalized.parquet.
"""
import gc
import os

import pandas as pd
import pyarrow.parquet as pq

RAW_DIR = "data/raw"
OUTPUT_PATH = os.path.join(RAW_DIR, "dengue_hospitalized.parquet")
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

EVOLUCAO_VALID = {"1", "2"}  # 1 = cura, 2 = óbito por dengue

FEATURES = [
    "CS_SEXO", "CS_GESTANT", "CS_RACA", "CS_ESCOL_N", "SG_UF", "SEM_PRI", "NU_IDADE_N",
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

# Columns read from the file: features + filter columns
READ_COLS = FEATURES + ["HOSPITALIZ", "EVOLUCAO"]


def filter_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if "HOSPITALIZ" not in df.columns or "EVOLUCAO" not in df.columns:
        return pd.DataFrame()

    df["HOSPITALIZ"] = df["HOSPITALIZ"].astype(str).str.strip()
    df["EVOLUCAO"] = df["EVOLUCAO"].astype(str).str.strip()

    df = df[df["HOSPITALIZ"] == "1"]
    df = df[df["EVOLUCAO"].isin(EVOLUCAO_VALID)]

    return df.drop(columns=["HOSPITALIZ"])


def process_year(year: int) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, f"dengue_{year}.parquet")

    if not os.path.exists(path):
        print(f"  AVISO: arquivo não encontrado — {path}")
        return None

    print(f"\n  Lendo {year}...")
    available = set(pq.read_schema(path).names)
    cols = [c for c in READ_COLS if c in available]
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")
    n_raw = len(df)

    df = filter_chunk(df)

    n_filtered = len(df)
    pct = n_filtered / n_raw * 100 if n_raw > 0 else 0
    print(f"  Brutos: {n_raw:,} | Internados c/ desfecho: {n_filtered:,} ({pct:.2f}%)")
    print(f"  EVOLUCAO: {df['EVOLUCAO'].value_counts().to_dict()}")

    return df if not df.empty else None


def main():
    print("=" * 55)
    print("Gerando dengue_hospitalized.parquet")
    print("=" * 55)

    list_dfs = []

    for year in YEARS:
        df = process_year(year)
        if df is not None:
            list_dfs.append(df)
        gc.collect()

    if not list_dfs:
        print("\nERRO: nenhum dado encontrado.")
        return

    df_final = pd.concat(list_dfs, ignore_index=True)
    del list_dfs
    gc.collect()

    evolucao_dist = df_final["EVOLUCAO"].value_counts().to_dict()

    print(f"\n{'='*55}")
    print(f"Total consolidado: {len(df_final):,} registros")
    print(f"Colunas: {len(df_final.columns)}")
    print(f"EVOLUCAO: {evolucao_dist}")
    print(f"Salvando em: {OUTPUT_PATH}")

    df_final.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print("Concluído.")


if __name__ == "__main__":
    main()
