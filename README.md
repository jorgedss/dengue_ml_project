# dengue_ml_project

Pipeline de aprendizado de máquina supervisionado para **predição de óbito por dengue**
a partir de dados do SINAN (Sistema de Informação de Agravos de Notificação), desenvolvido
como parte da dissertação de mestrado de José Jorge de Souza Silva.

O objetivo é a **avaliação comparativa da capacidade preditiva** de cinco algoritmos
que cobrem o espectro interpretabilidade↔complexidade: Regressão Logística, Árvore de
Decisão (CART), Random Forest, XGBoost e LightGBM.

## Estrutura do projeto

```
dengue_ml_project/
├── main.py                     # Orquestra o pipeline de dados completo
├── utils/                      # Pipeline de dados (download → tratamento → split)
│   ├── download_data.py        # Baixa os dados brutos do SINAN (2020–2025) em chunks
│   ├── clean_data.py           # Filtra internados com desfecho conhecido (todos os anos)
│   ├── treat_data.py           # Feature engineering + construção do alvo (dataset único)
│   └── split_data.py           # Split temporal: treino 2020–2023 / teste 2024
└── notebooks/
    ├── 01_models/              # Um notebook por modelo (treino + tuning + avaliação)
    ├── 02_evaluation/          # Comparação, bootstrap (IC 95%), calibração, validação 2025
    ├── 03_interpretation/      # Importância das variáveis (5 modelos)
    └── 04_fairness/            # Análise de equidade estratificada
```

## Pipeline de dados

Todos os anos de **2020 a 2025** são consolidados num **único dataset tratado**. A coluna
`year` é preservada e usada como parâmetro de split temporal (treino 2020–2023, teste 2024,
validação externa 2025).

Execute o pipeline completo a partir da raiz do projeto:

```bash
python main.py
```

Etapas (também executáveis individualmente):

1. `utils/download_data.py` → `data/raw/dengue_{ano}.parquet`
2. `utils/clean_data.py` → `data/raw/dengue_hospitalized.parquet`
3. `utils/treat_data.py` → `data/processed/dengue_treated.parquet`
4. `utils/split_data.py` → `data/features/baseline/{X,y}_{train,test}.parquet`

> Os diretórios `data/` e `output/` não são versionados; são regenerados pela execução
> do pipeline e dos notebooks.

## Notebooks

- **01_models/** — cada notebook contém o fluxo completo de um modelo: carregamento,
  pré-processamento, validação temporal (*expanding window*), otimização de
  hiperparâmetros (GridSearchCV), treino final e avaliação.
- **02_evaluation/** — comparação dos modelos (baseline 2024), bootstrap para intervalos
  de confiança 95%, calibração (Brier Score / BSS) e validação externa no ano de 2025.
- **03_interpretation/** — importância das variáveis dos cinco modelos: MDI/ganho para os
  modelos de árvore e coeficientes/razões de chances para a Regressão Logística.
- **04_fairness/** — análise de equidade estratificada (ΔAUPRC entre subgrupos).

## Ambiente

Projeto gerenciado com [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Requer Python ≥ 3.12.
