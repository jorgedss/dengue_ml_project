"""
    Downloads raw SINAN dengue datasets (2020–2025) and saves each year as a parquet file in data/raw/.
"""
import gc
import os

import pandas as pd
from pysus.online_data.SINAN import SINAN

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_year(sinan, year: int, partitions_limit: int = 60):
    print(f"\n{'='*55}")
    print(f"Baixando {year}...")

    files = sinan.get_files("DENG", year)
    folder_cache = str(sinan.download(files))

    partitions = sorted([f for f in os.listdir(folder_cache) if f.endswith(".parquet")])
    total_partitions = len(partitions)
    print(f"  Partições encontradas: {total_partitions}")

    list_dfs = []

    for i in range(0, total_partitions, partitions_limit):
        batch = partitions[i : i + partitions_limit]
        print(f"  > Batch {i}–{min(i + partitions_limit, total_partitions)}")

        frames = [pd.read_parquet(os.path.join(folder_cache, p)) for p in batch]
        df_batch = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        list_dfs.append(df_batch)
        del df_batch
        gc.collect()

    if not list_dfs:
        print(f"  AVISO: nenhum registro encontrado em {year}.")
        return

    df_final = pd.concat(list_dfs, ignore_index=True)
    del list_dfs
    gc.collect()

    dest = os.path.join(OUTPUT_DIR, f"dengue_{year}.parquet")
    df_final.to_parquet(dest, index=False, engine="pyarrow")
    print(f"  Registros: {len(df_final):,} | Colunas: {len(df_final.columns)} | Salvo: {dest}")

    del df_final
    gc.collect()


def main():
    sinan = SINAN().load()
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    for year in years:
        download_year(sinan, year)


if __name__ == "__main__":
    main()
