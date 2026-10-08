"""
Applies all feature transformations to dengue_hospitalized.parquet:
builds the binary target, decodes age, extracts date features, and converts
SINAN 1/2 symptom encoding to binary 0/1. Output: data/processed/dengue_treated.parquet
"""
import os

import numpy as np
import pandas as pd
from pysus.preprocessing.decoders import decodifica_idade_SINAN

try:  # execucao via pacote (python main.py)
    from utils.clean_data import DATE_COLS
except ImportError:  # execucao direta (python utils/treat_data.py)
    from clean_data import DATE_COLS

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

# Blocos de sinais, derivados de BINARY_SINAN_COLS para nao duplicar a lista.
ALRM_COLS = [c for c in BINARY_SINAN_COLS if c.startswith("ALRM_")]
GRAV_COLS = [c for c in BINARY_SINAN_COLS if c.startswith("GRAV_")]

# Codificacoes aceitas por treat():
#   'antiga'    -> comportamento historico, inalterado (padrao)
#   'corrigida' -> acrescenta os indicadores de bloco avaliado
CODIFICACOES = ("antiga", "corrigida")

BLOCO_COLS = ["bloco_alarme_avaliado", "bloco_gravidade_avaliado"]

# Macrorregiao do IBGE, derivada do primeiro digito do codigo da UF em SG_UF.
# Substitui SG_UF como preditor nas codificacoes nao publicadas: as 27 indicadoras
# de UF renderam ganho de 0,012 na validacao cruzada e perda de 0,016 no teste
# temporal da regressao logistica, invertendo o padrao dos outros quatro
# algoritmos, e o diagnostico afastou subrregularizacao como explicacao.
COL_MACRORREGIAO = "macrorregiao"
MACRORREGIAO_POR_DIGITO = {
    "1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste",
}


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


def macrorregiao_da_uf(series: pd.Series) -> pd.Series:
    """Primeiro digito do codigo do IBGE em SG_UF -> macrorregiao.

    Codigo ausente ou fora da tabela vira NaN: nao ha categoria de despejo, e o
    ausente segue para o mesmo tratamento das demais categoricas.
    """
    digito = series.astype(str).str.strip().str[0]
    return digito.map(MACRORREGIAO_POR_DIGITO).astype(object)


def _exigir_macrorregiao_completa(uf: pd.Series, macrorregiao: pd.Series) -> None:
    """Interrompe se alguma SG_UF nao mapear para uma das cinco macrorregioes.

    As bases do DATASUS sao revisadas retroativamente e o artigo declara a data da
    extracao. Uma reextracao que traga codigo novo, ausente ou fora da tabela do IBGE
    tem de falhar aqui, visivelmente e no momento da extracao, em vez de virar uma
    linha de modelo sem informacao geografica.
    """
    sem_mapa = macrorregiao.isna()
    if not sem_mapa.any():
        return
    # Os dois casos pedem decisoes diferentes e por isso sao contados a parte: sem UF
    # informada e um problema de preenchimento da ficha; codigo fora da tabela e um
    # problema de tabela. Listar os dois juntos mandaria quem le procurar 'nan' entre
    # os codigos do IBGE.
    faltantes = uf[sem_mapa]
    sem_uf = faltantes.isna()
    partes = []
    if sem_uf.any():
        partes.append(f"{int(sem_uf.sum()):,} sem UF informada")
    fora = faltantes[~sem_uf].astype(str).value_counts()
    if len(fora):
        detalhe = ", ".join(f"{codigo}: {n:,}" for codigo, n in fora.items())
        partes.append(
            f"{int((~sem_uf).sum()):,} com codigo fora da tabela do IBGE ({detalhe})"
        )
    raise ValueError(
        f"SG_UF sem macrorregiao correspondente em {int(sem_mapa.sum()):,} registros: "
        f"{'; '.join(partes)}. O primeiro digito do codigo tem de estar em "
        f"{sorted(MACRORREGIAO_POR_DIGITO)}. Se o IBGE mudou a tabela, atualizar "
        f"MACRORREGIAO_POR_DIGITO; se a reextracao trouxe registro sem UF, decidir o "
        f"tratamento antes de seguir."
    )


def treat(df: pd.DataFrame, codificacao: str = "antiga") -> pd.DataFrame:
    """`codificacao='antiga'` (padrao) preserva o comportamento historico.

    `codificacao='corrigida'` acrescenta `bloco_alarme_avaliado`,
    `bloco_gravidade_avaliado` e `macrorregiao`. O passo 2 ja preserva o ausente
    como NaN nos 24 campos, e a fusao de "nao" com "nao avaliado" acontece
    depois, no ColumnTransformer dos notebooks de treino.

    `SG_UF` continua na saida nas duas codificacoes: na corrigida ela deixa de ser
    preditora, mas segue acompanhando a base como origem da macrorregiao e como
    chave das analises por unidade federativa.
    """
    if codificacao not in CODIFICACOES:
        raise ValueError(
            f"codificacao invalida: {codificacao!r}. Use uma de {CODIFICACOES}."
        )

    # ── 0. Datas: 'YYYYMMDD' → datetime64 ────────────────────────────────────
    # Antes de tudo: o passo 6 normaliza colunas object e corromperia as datas
    # se elas ainda fossem string.
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
            )

    # ── 1. Target ─────────────────────────────────────────────────────────────
    df["target"] = build_target(df)
    df = df.drop(columns=["EVOLUCAO"])

    # ── 1b. Blocos avaliados (somente codificacao='corrigida') ───────────────
    # Calculadas a partir do codigo bruto, obrigatoriamente antes do passo 2:
    # respondido = codigo 1 (sim) ou 2 (nao); 9/branco/outros = nao avaliado.
    if codificacao == "corrigida":
        for nome, campos in zip(BLOCO_COLS, (ALRM_COLS, GRAV_COLS)):
            presentes = [c for c in campos if c in df.columns]
            if not presentes:
                df[nome] = np.nan
                continue
            respondido = pd.DataFrame(
                {c: _to_numeric(df[c]).isin([1, 2]) for c in presentes},
                index=df.index,
            )
            df[nome] = respondido.any(axis=1).astype("int8")

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
    # O `astype(str)` normaliza o conteudo, mas convertia tambem a ausencia: NaN
    # virava o texto 'nan' e None virava 'None', e o registro deixava de ser
    # detectavel por isna(). Atingia os 187 registros de CS_SEXO com codigo "I", que
    # o passo 5 acabara de tornar ausentes, e deixava SG_UF e CLASSI_FIN expostas ao
    # mesmo efeito. A mascara guarda quem ja estava ausente e o devolve ao NaN depois
    # da normalizacao, sem mudar o tratamento de nenhum valor presente.
    #
    # A correcao nao se propaga a data/processed/dengue_treated_antiga.parquet, que
    # nao e regenerado. Aquele arquivo existe para reproduzir os modelos publicados, a
    # reproducao esta verificada por sha256 coluna a coluna, e o defeito e inocuo ali:
    # o OrdinalEncoder do sexo trata o texto 'nan' como desconhecido e o converte em
    # NaN, de modo que os dois caminhos convergem antes de chegar ao estimador.
    # Regenera-lo trocaria uma reproducao verificada por outra a reverificar.
    for col in df.select_dtypes(include="object").columns:
        ausente = df[col].isna()
        df[col] = df[col].astype(str).str.strip().replace("", np.nan)
        df.loc[ausente, col] = np.nan

    # ── 7. Macrorregiao (somente codificacao='corrigida') ────────────────────
    # Depois do passo 6, e nao antes: o `astype(str)` de la converte o ausente de
    # uma coluna object no texto 'nan'. O codificador nao reconhece essa string como
    # nivel ausente e, com `handle_unknown='ignore'`, deixaria o registro sair com as
    # cinco indicadoras em 0 -- indistinguivel de um registro valido. Derivada aqui, a
    # coluna fica fora do alcance da conversao e o ausente dela e NaN de verdade.
    if codificacao == "corrigida" and "SG_UF" in df.columns:
        df[COL_MACRORREGIAO] = macrorregiao_da_uf(df["SG_UF"])
        _exigir_macrorregiao_completa(df["SG_UF"], df[COL_MACRORREGIAO])

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
