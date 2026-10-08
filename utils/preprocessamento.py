"""
Definicao unica do pre-processamento dos modelos: o `ColumnTransformer` de cada
familia de algoritmo, montado a partir da lista de preditores da configuracao.

Existia antes dentro de `notebooks/09_retreino`, o que deixava a definicao sujeita
a copias divergentes entre notebooks. Aqui ela e a mesma para todo o pipeline, e a
codificacao das categoricas segue a chave `codificacao_categoricas` da configuracao:

  'publicada' (configuracao antiga, linha de base do estudo submetido)
      UF ordinal pelo codigo do IBGE; escolaridade numa unica escala ordinal com o
      codigo 10 ("nao se aplica") dentro dela; nas familias de
      reforco, raca/cor, gestacao e escolaridade chegam como codigo numerico bruto.

  'corrigida' (configuracoes corrigida e notificacao)
      Macrorregiao no lugar da UF, em one-hot nas cinco familias; raca/cor e gestacao
      tambem em one-hot, com os niveis declarados e o ausente como nivel proprio;
      escolaridade em tres colunas -- escala ordinal de 0 a 8, indicadora do codigo 10
      e indicadora de ausente. Sexo segue ordinal: sendo binaria, ordinal equivale a
      one-hot com categoria de referencia.

      Nenhuma demografica e imputada aqui. Onde o estimador percorre o ausente
      nativamente (arvore e reforco) ele segue como NaN; na familia linear, que exige
      valor finito, a escala ordinal e o sexo recebem constante fora da escala, e as
      indicadoras dizem de que ausencia se trata. A unica imputacao estatistica que
      resta no pre-processamento e a moda dos sintomas gerais.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler,
)

try:  # execucao via pacote (import utils.preprocessamento)
    from utils import configuracoes
except ImportError:  # execucao direta (python utils/preprocessamento.py)
    import configuracoes

# Sinais, sintomas e comorbidades: recebem a moda do treino.
SINTOMAS_GERAIS = [
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA',
    'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA', 'PETEQUIA_N',
    'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'DIABETES', 'HEMATOLOG',
    'HEPATOPAT', 'RENAL', 'HIPERTENSA', 'AUTO_IMUNE',
]

COLS_BLOCO = [configuracoes.BLOCO_ALARME, configuracoes.BLOCO_GRAVIDADE]

FAMILIAS = ('linear', 'arvore', 'boosting')

# Categoricas nominais: sem ordem entre as categorias, logo one-hot na
# codificacao corrigida. Sexo fica de fora por ser binaria. SG_UF continua aqui
# porque segue preditora na configuracao antiga, onde recebe codificacao ordinal.
NOMINAIS = ('CS_RACA', 'CS_GESTANT', configuracoes.COL_UF,
            configuracoes.COL_MACRORREGIAO)

# CS_ESCOL_N: 0 a 8 sao niveis crescentes de estudo, e o codigo 10 nao pertence a
# essa escala: significa "nao se aplica". Na codificacao publicada
# ele entrava como 10, duas unidades acima de superior completo.
CODIGO_ESCOLARIDADE_NAO_SE_APLICA = 10.0
COL_ESCOLARIDADE_NAO_SE_APLICA = 'escolaridade_nao_se_aplica'
COL_ESCOLARIDADE_AUSENTE = 'escolaridade_ausente'

# Sexo: binaria, declarada no `OrdinalEncoder`. Tudo fora deste par -- inclusive o
# ausente -- e desconhecido para o codificador.
CATEGORIAS_SEXO = ['F', 'M']
COL_SEXO_AUSENTE = 'sexo_ausente'

# Familias cujo estimador percorre o ausente nativamente: arvore de decisao e random
# forest desde o scikit-learn 1.4, XGBoost e LightGBM desde sempre. So a familia
# linear exige valor finito, e e por isso que a regra tem duas implementacoes e nao
# cinco. O pino do ambiente e scikit-learn==1.8.0.
FAMILIAS_COM_NAN_NATIVO = ('arvore', 'boosting')

# Valor dado a escala ordinal da escolaridade e ao sexo na familia linear, que nao
# aceita NaN. Fica fora da escala 0-8 e fora do par 0/1 do sexo, e as indicadoras
# dizem qual ausencia e qual: "nao se aplica" ou "sem informacao".
CONSTANTE_FORA_DA_ESCALA = -1.0

# Niveis declarados das categoricas nominais, em vez de aprendidos da particao de
# treino. Sem `categories=`, um nivel que nao aparecesse no treino sairia zerado sem
# aviso, inclusive o proprio nivel ausente. O NaN tem de ser o ultimo da lista.
CATEGORIAS_RACA = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]
CATEGORIAS_GESTANT = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]
# A macrorregiao nao tem ausente na coorte: sem nivel NaN, sem coluna morta.
CATEGORIAS_MACRORREGIAO = list(configuracoes.MACRORREGIOES)


def aceita_nan(familia):
    '''Se o estimador da familia trata o ausente sem que o pre-processador preencha.'''
    return familia in FAMILIAS_COM_NAN_NATIVO


def _escolaridade_ordinal(X):
    '''Escala ordinal de escolaridade sem o codigo 10, que sai da escala e vira NaN.

    A coluna fica com NaN em dois casos diferentes: quem respondeu o codigo 10 e quem
    nao informou nada. As duas indicadoras que acompanham esta coluna e que separam
    um do outro; sozinha, ela nao distingue.
    '''
    a = np.array(X, dtype=float, copy=True).reshape(len(X), -1)
    a[a == CODIGO_ESCOLARIDADE_NAO_SE_APLICA] = np.nan
    return a


def _escolaridade_indicador(X):
    '''1 no codigo 10 ("nao se aplica"), 0 no resto.

    Escolaridade ausente nao e "nao se aplica": fica 0 aqui e 1 na indicadora de
    ausente. Antes esta docstring dizia que a ausencia viajava na coluna ordinal, o
    que a imputacao por mediana desmentia -- ela apagava a ausencia antes do modelo.
    '''
    a = np.asarray(X, dtype=float).reshape(len(X), -1)
    return (a == CODIGO_ESCOLARIDADE_NAO_SE_APLICA).astype(float)


def _escolaridade_ausente(X):
    '''1 onde a escolaridade nao foi informada, 0 no resto.

    Lida do codigo bruto, antes de `_escolaridade_ordinal` transformar o codigo 10 em
    NaN: aqui o NaN e so o campo vazio.
    '''
    a = np.asarray(X, dtype=float).reshape(len(X), -1)
    return np.isnan(a).astype(float)


def _sexo_ausente(X):
    '''1 onde o sexo nao e F nem M, 0 no resto.

    Espelha exatamente o que o `OrdinalEncoder` de sexo considera desconhecido. Na
    familia linear a coluna ordinal recebe a constante fora da escala, e e esta
    indicadora que diz que aquele valor nao e um nivel de sexo -- sem ela o
    coeficiente unico da regressao teria de valer para F, para M e para o ausente na
    mesma escala, fazendo o ausente entrar como extrapolacao.
    '''
    a = np.asarray(X, dtype=object).reshape(len(X), -1)
    return (~np.isin(a, CATEGORIAS_SEXO)).astype(float)


def _nome_ausente_sexo(transformador, nomes_entrada):
    return np.asarray([COL_SEXO_AUSENTE], dtype=object)


def _nome_indicador_escolaridade(transformador, nomes_entrada):
    return np.asarray([COL_ESCOLARIDADE_NAO_SE_APLICA], dtype=object)


def _nome_ausente_escolaridade(transformador, nomes_entrada):
    return np.asarray([COL_ESCOLARIDADE_AUSENTE], dtype=object)


def _one_hot_publicado():
    '''One-hot da codificacao publicada: niveis aprendidos do treino e ausente
    preenchido com a moda.

    Existe so para a configuracao antiga, que e a linha de base do estudo submetido e
    por isso nao muda. Nas demais, `_one_hot` declara os niveis e guarda o ausente.
    '''
    return Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('enc', OneHotEncoder(handle_unknown='ignore',
                                           sparse_output=False))])


def _one_hot(categorias, desconhecido='ignore'):
    '''One-hot dos niveis declarados, sem imputacao e sem categoria de referencia.

    Quando o NaN esta na lista de `categorias`, o ausente vira um nivel proprio com
    coluna propria, em vez de ser preenchido.

    `desconhecido` decide o que fazer com um valor fora da lista. Em 'ignore' ele sai
    com todas as indicadoras em 0, o que para o modelo equivale a uma categoria de
    referencia implicita -- justamente o que o desenho sem `drop` quer evitar. Use
    'error' quando a lista de niveis for exaustiva por construcao e um valor fora dela
    significar que a base mudou: ai a execucao para em vez de seguir com a linha sem
    informacao. Em 'ignore' ficam as variaveis que ja declaram o proprio NaN como
    nivel, onde o ausente tem coluna e nao depende deste parametro.
    '''
    return OneHotEncoder(categories=[list(categorias)], handle_unknown=desconhecido,
                         sparse_output=False)


def _transformadores_categoricos(cfg, familia):
    '''Demograficas categoricas, na codificacao declarada pela configuracao.

    A familia decide duas coisas, e so duas: o z-score, que vale so para a linear e
    so na escala ordinal da escolaridade; e se o ausente pode seguir como NaN ate o
    estimador ou precisa de um valor finito. Nenhuma imputacao estatistica e decidida
    aqui -- na codificacao corrigida, nenhuma destas variaveis recebe imputacao.
    '''
    corrigida = configuracoes.codificacao_categoricas(cfg) == 'corrigida'
    escalar = familia == 'linear'

    def num(est):
        return Pipeline([('imp', est), ('scl', StandardScaler())]) if escalar else est

    if not corrigida:
        # Codificacao publicada: preservada tal e qual, imputacao por moda e por
        # mediana inclusive. E a linha de base do estudo submetido.
        return [
            ('sexo', Pipeline([
                ('enc', OrdinalEncoder(categories=[['F', 'M']],
                                       handle_unknown='use_encoded_value',
                                       unknown_value=np.nan)),
                ('imp', SimpleImputer(strategy='most_frequent')),
            ]), ['CS_SEXO']),
            ('escol', num(SimpleImputer(strategy='median')), ['CS_ESCOL_N']),
            ('raca', _one_hot_publicado(), ['CS_RACA']),
            ('gestant', _one_hot_publicado(), ['CS_GESTANT']),
            ('uf', OrdinalEncoder(handle_unknown='use_encoded_value',
                                  unknown_value=-1), ['SG_UF']),
        ]

    # Nas familias que percorrem o ausente nativamente ele segue como NaN; na linear,
    # que exige valor finito, recebe uma constante fora da escala. Duas implementacoes
    # da mesma regra, nao cinco.
    fecho = ([] if aceita_nan(familia) else
             [('const', SimpleImputer(strategy='constant',
                                      fill_value=CONSTANTE_FORA_DA_ESCALA))])

    # Na familia linear o ausente do sexo vira a constante fora da escala, e precisa
    # da indicadora para nao ser lido como um terceiro nivel na mesma reta -- o mesmo
    # par de colunas ja dado a escala ordinal da escolaridade. Onde o NaN e nativo a
    # indicadora seria redundante: o algoritmo ja separa o ausente sozinho.
    sexo_ausente = [] if aceita_nan(familia) else [
        ('sexo_ausente', FunctionTransformer(
            _sexo_ausente, feature_names_out=_nome_ausente_sexo), ['CS_SEXO']),
    ]

    return [
        ('sexo', Pipeline([
            ('enc', OrdinalEncoder(categories=[CATEGORIAS_SEXO],
                                   handle_unknown='use_encoded_value',
                                   unknown_value=np.nan)),
        ] + fecho), ['CS_SEXO']),
    ] + sexo_ausente + [
        # Escolaridade em tres colunas: a escala ordinal, a indicadora do codigo 10 e
        # a indicadora de ausente. Sem mediana: o ausente chega ao modelo declarado.
        ('escol', Pipeline([
            ('sep', FunctionTransformer(_escolaridade_ordinal,
                                        feature_names_out='one-to-one')),
        ] + fecho + ([('scl', StandardScaler())] if escalar else [])), ['CS_ESCOL_N']),
        ('escol_na', FunctionTransformer(
            _escolaridade_indicador,
            feature_names_out=_nome_indicador_escolaridade), ['CS_ESCOL_N']),
        ('escol_ausente', FunctionTransformer(
            _escolaridade_ausente,
            feature_names_out=_nome_ausente_escolaridade), ['CS_ESCOL_N']),
        # Raca/cor e gestacao: o ausente e nivel proprio do codificador, com coluna
        # propria, em todas as cinco familias. Nenhuma imputacao por moda.
        ('raca', _one_hot(CATEGORIAS_RACA), ['CS_RACA']),
        ('gestant', _one_hot(CATEGORIAS_GESTANT), ['CS_GESTANT']),
        # Os cinco niveis do IBGE esgotam a macrorregiao, e `treat_data` ja exige que
        # toda SG_UF mapeie para um deles. Um valor fora da lista chegando ate aqui
        # significa base mudada, entao para a execucao em vez de zerar as cinco.
        # Raca/cor e gestacao ficam em 'ignore': o ausente delas tem coluna propria.
        ('macro', _one_hot(CATEGORIAS_MACRORREGIAO, desconhecido='error'),
         [configuracoes.COL_MACRORREGIAO]),
    ]


def campos_clinicos(preds):
    '''Os 24 campos de alarme e gravidade presentes na configuracao.'''
    return [c for c in preds if c.startswith('ALRM_') or c.startswith('GRAV_')]


def preprocessador(cfg, familia):
    '''familia: 'linear' (z-score), 'arvore' (sem escala) ou 'boosting' (NaN nativo).'''
    if familia not in FAMILIAS:
        raise ValueError(f'familia invalida: {familia!r}. Use uma de {FAMILIAS}.')
    preds = configuracoes.preditores(cfg)
    clinicos = campos_clinicos(preds)
    sintomas = [c for c in SINTOMAS_GERAIS if c in preds]
    blocos   = [c for c in COLS_BLOCO if c in preds]
    corrigida = configuracoes.codificacao_categoricas(cfg) == 'corrigida'

    if familia == 'boosting':
        if corrigida:
            # Mesma codificacao das demais familias para as categoricas; o que sobra
            # passa direto, com o ausente indo para o tratamento nativo do algoritmo.
            transformadores = _transformadores_categoricos(cfg, familia)
        else:
            transformadores = [
                ('sexo', OrdinalEncoder(categories=[['F', 'M']],
                                        handle_unknown='use_encoded_value',
                                        unknown_value=-1), ['CS_SEXO']),
                ('uf', OrdinalEncoder(handle_unknown='use_encoded_value',
                                      unknown_value=-1), ['SG_UF']),
            ]
        if not blocos:
            # Configuração antiga: sem colunas de bloco, o ausente dos campos clínicos
            # é preenchido com 0, exatamente como no estudo publicado.
            transformadores.insert(0, ('clinicos',
                                       SimpleImputer(strategy='constant', fill_value=0),
                                       clinicos))
        # Nas demais, o ausente segue como NaN para o tratamento nativo do algoritmo;
        # a disponibilidade do bloco vai nos indicadores, que passam direto.
        return ColumnTransformer(transformadores, remainder='passthrough',
                                 verbose_feature_names_out=False), clinicos

    escalar = familia == 'linear'
    def num(est):
        return Pipeline([('imp', est), ('scl', StandardScaler())]) if escalar else est

    transformadores = [
        # Campos de alarme e gravidade: ausente preenchido com 0.
        ('clinicos', SimpleImputer(strategy='constant', fill_value=0), clinicos),
        # Sintomas gerais e comorbidades: UNICA imputacao estatistica que sobra no
        # pre-processamento. A moda sai do `fit`, isto e, so da particao de treino --
        # dentro do pipeline, de modo que cada dobra da validacao cruzada calcula a
        # sua. Todas as demograficas da codificacao corrigida deixaram de ser
        # imputadas: o ausente delas chega ao modelo declarado.
        ('sintomas', SimpleImputer(strategy='most_frequent'), sintomas),
        # Discretas: mediana do treino.
        ('continuas', num(SimpleImputer(strategy='median')), ['age_years', 'epi_week']),
    ] + _transformadores_categoricos(cfg, familia)
    if blocos:
        transformadores.append(
            ('bloco', SimpleImputer(strategy='constant', fill_value=0), blocos))

    cobertas = [c for _, _, cols in transformadores for c in cols]
    fora = sorted(set(preds) - set(cobertas))
    assert not fora, f'{cfg}/{familia}: preditores fora do pré-processador — {fora}'
    return ColumnTransformer(transformadores, remainder='drop',
                             verbose_feature_names_out=False), clinicos


# ── Auditoria: que codificacao cada preditor recebe em cada familia ──────────
# As codificacoes que a auditoria compara ficam em constantes: a adequacao ao tipo
# do preditor e decidida por identidade com estes valores, e um literal repetido a
# mao em cada ponto de uso deixaria a comparacao falhar em silencio se um deles
# fosse reescrito sozinho.
COD_ONE_HOT = 'one-hot'
COD_ORDINAL_CORRIGIDA = ('ordinal de 0 a 8 mais indicadoras de "não se aplica" '
                         'e de ausente')
COD_UF_ORDINAL = 'ordinal pelo código do IBGE'

CODIFICACAO_DO_TRANSFORMADOR = {
    'clinicos':  'binária, ausente preenchido com 0',
    'sintomas':  'binária, ausente na moda',
    'continuas': 'numérica, ausente na mediana',
    'sexo':      f'ordinal binária (equivale a {COD_ONE_HOT} com referência)',
    'escol':     'ordinal',
    'escol_na':  'indicadora binária',
    'escol_ausente': 'indicadora binária',
    'sexo_ausente': 'indicadora binária',
    'raca':      COD_ONE_HOT,
    'gestant':   COD_ONE_HOT,
    # SG_UF so aparece na configuracao antiga, e sempre pelo OrdinalEncoder; a
    # macrorregiao e que carrega o eixo geografico nas demais, em one-hot.
    'uf':        COD_UF_ORDINAL,
    'macro':     COD_ONE_HOT,
    'bloco':     'binária, ausente preenchido com 0',
    'remainder': 'passagem direta do código bruto',
}

# Natureza de cada preditor, que e o que decide se a codificacao esta certa:
# so as nominais exigem one-hot; binarias e numericas ordenadas nao.
TIPO_NOMINAL   = 'nominal'
TIPO_ORDINAL   = 'ordinal'
TIPO_BINARIA   = 'binária'
TIPO_NUMERICA  = 'numérica ordenada'


def tipo_do_preditor(coluna):
    if coluna in NOMINAIS:
        return TIPO_NOMINAL
    if coluna == 'CS_ESCOL_N':
        return TIPO_ORDINAL
    if coluna in ('age_years', 'epi_week'):
        return TIPO_NUMERICA
    return TIPO_BINARIA


def tratamento_por_preditor(cfg, familia):
    '''Uma linha por preditor: transformador que o recebe, codificacao resultante e
    se a codificacao e adequada ao tipo do preditor.'''
    ct, _ = preprocessador(cfg, familia)
    preds = configuracoes.preditores(cfg)
    # Um preditor pode alimentar mais de um transformador: e o caso da escolaridade
    # corrigida, que sai numa escala ordinal e numa indicadora.
    de_transformador = {}
    for nome, _, cols in ct.transformers:
        for c in cols:
            de_transformador.setdefault(c, []).append(nome)

    linhas = []
    for c in preds:
        nomes = de_transformador.get(c, ['remainder'])
        if nomes == ['escol', 'escol_na', 'escol_ausente']:
            codificacao = COD_ORDINAL_CORRIGIDA
        else:
            nome = nomes[0]
            codificacao = CODIFICACAO_DO_TRANSFORMADOR[nome]
            if nome == 'escol':
                codificacao = 'ordinal com o código 10 dentro da escala'
        tipo = tipo_do_preditor(c)
        if tipo == TIPO_NOMINAL:
            adequada = codificacao == COD_ONE_HOT
        elif tipo == TIPO_ORDINAL:
            adequada = codificacao == COD_ORDINAL_CORRIGIDA
        else:
            adequada = True
        linhas.append({'preditor': c, 'tipo': tipo,
                       'transformador': ' + '.join(nomes),
                       'codificacao': codificacao,
                       'adequada_ao_tipo': 'sim' if adequada else 'não'})
    return linhas
