"""
Orquestra o pipeline de dados completo do projeto, na ordem de dependência:

  1. download_data  -> data/raw/dengue_{ano}.parquet          (SINAN 2020-2025, em chunks)
  2. clean_data     -> data/raw/dengue_hospitalized.parquet   (internados c/ desfecho, todos os anos)
  3. treat_data     -> data/processed/dengue_treated.parquet  (feature engineering, dataset unico)
  4. split_data     -> data/features/baseline/{X,y}_{train,test}.parquet (split temporal 2020-2023 / 2024)

Todos os anos (2020-2025) sao consolidados num unico dataset tratado; a coluna `year`
e preservada e usada como parametro de split (treino 2020-2023, teste 2024, validacao 2025).

Execute a partir da raiz do projeto:  python main.py
"""
from utils import download_data, clean_data, treat_data, split_data


def main():
    print("\n" + "#" * 60)
    print("# PIPELINE DE DADOS - dengue_ml_project")
    print("#" * 60)

    print("\n[1/4] Download dos dados brutos do SINAN (2020-2025)...")
    download_data.main()

    print("\n[2/4] Filtragem de internados com desfecho conhecido...")
    clean_data.main()

    print("\n[3/4] Tratamento e feature engineering...")
    treat_data.main()

    print("\n[4/4] Split temporal treino/teste...")
    split_data.main()

    print("\n" + "#" * 60)
    print("# PIPELINE CONCLUIDO")
    print("#" * 60)


if __name__ == "__main__":
    main()
