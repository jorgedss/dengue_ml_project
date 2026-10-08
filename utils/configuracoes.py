"""
Defines the three predictor configurations compared in the reanalysis: which columns
enter X and how missing values in the ALRM_*/GRAV_* blocks are meant to be handled
downstream.
"""
try:  # execucao via pacote (python main.py, import utils.configuracoes)
    from utils.clean_data import FEATURES
    from utils.treat_data import (
        ALRM_COLS, BLOCO_COLS, COL_MACRORREGIAO, GRAV_COLS, MACRORREGIAO_POR_DIGITO,
    )
except ImportError:  # execucao direta (python utils/configuracoes.py)
    from clean_data import FEATURES
    from treat_data import (
        ALRM_COLS, BLOCO_COLS, COL_MACRORREGIAO, GRAV_COLS, MACRORREGIAO_POR_DIGITO,
    )

# Colunas que acompanham as bases mas nunca entram em X.
COLUNAS_NAO_PREDITORAS = ["target", "year"]

COL_UF = "SG_UF"

# SG_UF deixa de ser preditora nas configuracoes corrigida e notificacao, mas
# continua acompanhando as bases: e a origem da macrorregiao e a chave das analises
# de equidade e de estabilidade geografica. Nas configuracoes em que ela ja e
# preditora, entra so uma vez.
COLUNAS_CARREGADAS = [COL_UF]

# Os cinco niveis da macrorregiao, na ordem do codigo do IBGE, para quem precisa
# declara-los ao codificador em vez de aprende-los da particao de treino.
MACRORREGIOES = list(MACRORREGIAO_POR_DIGITO.values())

# ── Blocos de preditores, na ordem de data/features/baseline/config.json ──────
DEMOGRAFICOS = ["CS_SEXO", "CS_GESTANT", "CS_RACA", "CS_ESCOL_N", "SG_UF"]

# Sinais, sintomas e comorbidades: o que sobra de FEATURES depois de tirar os
# blocos de alarme/gravidade, os demograficos e os campos que treat_data consome
# para derivar age_years e epi_week.
_CONSUMIDOS = set(DEMOGRAFICOS) | set(ALRM_COLS) | set(GRAV_COLS) | {"NU_IDADE_N", "SEM_PRI"}
SINTOMAS = [c for c in FEATURES if c not in _CONSUMIDOS]

DERIVADOS = ["age_years", "epi_week"]

# 51 preditores da analise publicada (as 52 features de config.json menos `year`,
# que e coluna de particao e sai antes do fit).
PREDITORES_ANTIGA = DEMOGRAFICOS + SINTOMAS + ALRM_COLS + GRAV_COLS + DERIVADOS

BLOCO_ALARME, BLOCO_GRAVIDADE = BLOCO_COLS


def _com_macrorregiao(preds: list[str]) -> list[str]:
    """Troca SG_UF pela macrorregiao, na mesma posicao da lista.

    As 27 indicadoras de UF renderam ganho de 0,012 na validacao cruzada e perda de
    0,016 no teste temporal da regressao logistica, invertendo o padrao dos outros
    quatro algoritmos; o diagnostico afastou subrregularizacao. As cinco indicadoras
    de macrorregiao mantem o eixo geografico com um vinte-e-dois avos das colunas.
    """
    return [COL_MACRORREGIAO if c == COL_UF else c for c in preds]

# Codificacao das categoricas, lida por utils/preprocessamento.py:
#   'publicada' -> UF ordinal e escolaridade numa unica escala com o codigo 10 dentro,
#                  como no estudo submetido;
#   'corrigida' -> UF em one-hot nas cinco familias, codigo 10 da escolaridade
#                  separado num indicador proprio, e nominais em one-hot tambem
#                  nas familias de reforco.
CODIFICACAO_CATEGORICAS = ("publicada", "corrigida")


def _configuracoes() -> dict[str, dict]:
    return {
        "antiga": {
            "nome": "antiga",
            "descricao": (
                "Codificacao publicada. Linha de base: ausente e 'nao' chegam ao "
                "modelo como o mesmo valor 0."
            ),
            "codificacao_treat": "antiga",
            "codificacao_categoricas": "publicada",
            "preditores": list(PREDITORES_ANTIGA),
            "tratamento_ausente": {
                "blocos_alrm_grav": "constante 0 (funde 'nao' com 'nao avaliado')",
                "sintomas": "moda",
                "continuas": "mediana",
                "categoricas": "moda",
            },
        },
        "corrigida": {
            "nome": "corrigida",
            "descricao": (
                "Ausente preservado como NaN nos 24 campos clinicos, com dois "
                "indicadores binarios de bloco avaliado; UF trocada por macrorregiao."
            ),
            "codificacao_treat": "corrigida",
            "codificacao_categoricas": "corrigida",
            "preditores": _com_macrorregiao(
                list(PREDITORES_ANTIGA) + [BLOCO_ALARME, BLOCO_GRAVIDADE]
            ),
            "tratamento_ausente": {
                "blocos_alrm_grav": "NaN preservado; disponibilidade vai nos indicadores de bloco",
                "sintomas": "moda",
                "continuas": "mediana",
                "categoricas": "moda",
            },
        },
        # Os 9 sinais de alarme entram nesta configuracao: DT_ALRM fica em mediana
        # um dia ANTES de DT_NOTIFIC, com 78% dos obitos em data igual ou anterior
        # a notificacao (analysis/results_temporalidade/defasagem_sinal_vs_notificacao.csv).
        # Os 15 criterios de gravidade ficam de fora, com o indicador do bloco.
        "notificacao": {
            "nome": "notificacao",
            "descricao": (
                "Campos disponiveis na notificacao: igual a corrigida, sem os 15 "
                "criterios de gravidade e sem o indicador do bloco de gravidade."
            ),
            "codificacao_treat": "corrigida",
            "codificacao_categoricas": "corrigida",
            "preditores": _com_macrorregiao([
                c
                for c in list(PREDITORES_ANTIGA) + [BLOCO_ALARME, BLOCO_GRAVIDADE]
                if c not in set(GRAV_COLS) | {BLOCO_GRAVIDADE}
            ]),
            "tratamento_ausente": {
                "blocos_alrm_grav": "NaN preservado nos ALRM_*; disponibilidade no indicador de alarme",
                "sintomas": "moda",
                "continuas": "mediana",
                "categoricas": "moda",
            },
        },
    }


def configuracao(nome: str) -> dict:
    todas = _configuracoes()
    if nome not in todas:
        raise ValueError(
            f"configuracao invalida: {nome!r}. Use uma de {tuple(todas)}."
        )
    return todas[nome]


def preditores(nome: str) -> list[str]:
    return list(configuracao(nome)["preditores"])


def codificacao_categoricas(nome: str) -> str:
    return configuracao(nome)["codificacao_categoricas"]


def todas_configuracoes() -> dict[str, dict]:
    return _configuracoes()


NOMES = ("antiga", "corrigida", "notificacao")

# Chave interna -> rotulo publicado.
#
#   "antiga"      -> "configuracao original"
#   "corrigida"   -> "configuracao final"
#   "notificacao" -> "configuracao sem criterios de gravidade"
#
# As chaves sao identificadores e nao acompanham o rotulo: elas nomeiam os arquivos
# data/processed/dengue_treated_<chave>.parquet, os artefatos de output/ e as colunas
# das tabelas de analysis/. Renomea-las quebraria a verificacao de reproducao por
# sha256 da configuracao original, ancorada em data/features/baseline/.
# Os notebooks fazem a traducao na exibicao, pelo dicionario ROTULO_CONFIG de cada um.
ROTULO_PUBLICADO = {
    "antiga": "original",
    "corrigida": "final",
    "notificacao": "sem criterios de gravidade",
}


def main():
    print("=" * 60)
    print("Configuracoes de preditores")
    print("=" * 60)
    for nome, cfg in _configuracoes().items():
        print(f"\n{nome} — {len(cfg['preditores'])} preditores")
        print(f"  {cfg['descricao']}")
        print(f"  treat(codificacao={cfg['codificacao_treat']!r})")
        print(f"  categoricas: {cfg['codificacao_categoricas']}")
        print(f"  ausente nos blocos: {cfg['tratamento_ausente']['blocos_alrm_grav']}")


if __name__ == "__main__":
    main()
