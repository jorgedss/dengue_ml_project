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

# Criterio de inclusao: dengue confirmada. Ficam de fora 5 (descartado),
# 8 (inconclusivo), 9 (ignorado) e o campo em branco.
CLASSI_CONFIRMED = {"10", "11", "12"}

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

# Metadados: viajam junto com a coorte para permitir auditoria e analise de
# features, mas NUNCA entram em X. A exclusao e feita em split_data.py, que
# importa METADATA_COLS daqui.
DATE_COLS = [
    "DT_NOTIFIC", "DT_SIN_PRI", "DT_ALRM", "DT_GRAV", "DT_INVEST",
    "DT_ENCERRA", "DT_OBITO", "DT_INTERNA", "DT_DIGITA",
]

# CLASSI_FIN e metadado pelo mesmo motivo que as datas, e por um adicional:
# e atribuida no encerramento do caso, e o codigo 12 tem 52% de letalidade.
# Como preditor seria vazamento direto do alvo.
METADATA_COLS = DATE_COLS + ["CLASSI_FIN"]

# Columns read from the file: features + metadata + filter columns
READ_COLS = FEATURES + METADATA_COLS + ["HOSPITALIZ", "EVOLUCAO"]


def filter_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coorte + quantos registros havia antes do filtro de CLASSI_FIN."""
    obrigatorias = {"HOSPITALIZ", "EVOLUCAO", "CLASSI_FIN"}
    if not obrigatorias.issubset(df.columns):
        return pd.DataFrame(), 0

    for col in obrigatorias:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["HOSPITALIZ"] == "1"]
    df = df[df["EVOLUCAO"].isin(EVOLUCAO_VALID)]
    n_desfecho = len(df)

    df = df[df["CLASSI_FIN"].isin(CLASSI_CONFIRMED)]

    return df.drop(columns=["HOSPITALIZ"]), n_desfecho


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

    df, n_desfecho = filter_chunk(df)

    n_final = len(df)
    n_removidos = n_desfecho - n_final
    pct_rem = n_removidos / n_desfecho * 100 if n_desfecho > 0 else 0
    print(f"  Brutos: {n_raw:,} | Internados c/ desfecho: {n_desfecho:,}")
    print(f"  Dengue confirmada: {n_final:,} | Removidos por CLASSI_FIN: {n_removidos:,} ({pct_rem:.2f}%)")
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
