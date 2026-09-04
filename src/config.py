"""
config.py
=========

Configuração central do projeto.

Este arquivo existe para que, se o CSV do FBref mudar de formato no futuro
(colunas novas, nomes diferentes, etc.), seja necessário ajustar as coisas
em UM lugar só, em vez de procurar por todo o código.

O CSV real que analisamos (data/raw/mls_players.csv) é a tabela "Standard
Stats" do FBref para jogadores da MLS. Ela vem com um cabeçalho duplo
(duas linhas de header), então depois de carregada e "achatada" pelo
data_loader.py, as colunas originais ficam assim:

    Rk, Player, Nation, Pos, Squad, Age, Born,
    Playing Time_MP, Playing Time_Starts, Playing Time_Min, Playing Time_90s,
    Performance_Gls, Performance_Ast, Performance_G+A, Performance_G-PK,
    Performance_PK, Performance_PKatt, Performance_CrdY, Performance_CrdR,
    Per 90 Minutes_Gls, Per 90 Minutes_Ast, Per 90 Minutes_G+A,
    Per 90 Minutes_G-PK, Per 90 Minutes_G+A-PK,
    Matches, -additional_-9999

IMPORTANTE: este é um dataset de estatísticas BÁSICAS (gols, assistências,
cartões, minutos). Ele NÃO contém xG, xAG, chutes, passes progressivos,
desarmes, interceptações, etc. Por isso as análises e o clustering deste
projeto usam apenas o que realmente existe no arquivo. Se um dia você usar
um CSV mais completo do FBref (com mais tabelas de estatísticas), basta
atualizar os dicionários abaixo.
"""

# ---------------------------------------------------------------------------
# MAPA DE RENOMEAÇÃO DE COLUNAS
# ---------------------------------------------------------------------------
# Chave: nome da coluna depois de "achatar" o cabeçalho duplo do CSV.
# Valor: nome interno, mais curto e fácil de usar no código (snake_case).
#
# Isso é feito em UM lugar só (clean_column_names, em data_cleaning.py usa
# este dicionário) para documentar claramente o mapeamento pedido no
# projeto: "nome original -> nome interno".
COLUMN_RENAME_MAP = {
    "Rk": "rk",
    "Player": "player",
    "Nation": "nation",
    "Pos": "pos",
    "Squad": "squad",
    "Age": "age",
    "Born": "born_year",
    "Playing Time_MP": "mp",
    "Playing Time_Starts": "starts",
    "Playing Time_Min": "min",
    "Playing Time_90s": "nineties",
    "Performance_Gls": "gls",
    "Performance_Ast": "ast",
    "Performance_G+A": "g_a",
    "Performance_G-PK": "g_minus_pk",
    "Performance_PK": "pk",
    "Performance_PKatt": "pkatt",
    "Performance_CrdY": "crdy",
    "Performance_CrdR": "crdr",
    "Per 90 Minutes_Gls": "gls_per90",
    "Per 90 Minutes_Ast": "ast_per90",
    "Per 90 Minutes_G+A": "g_a_per90",
    "Per 90 Minutes_G-PK": "g_minus_pk_per90",
    "Per 90 Minutes_G+A-PK": "g_a_minus_pk_per90",
}

# Colunas que não carregam informação estatística útil e podem ser
# descartadas na limpeza (link de texto "Matches" e um hash interno do
# FBref usado apenas para montar URLs).
COLUMNS_TO_DROP = ["Matches", "-additional_-9999"]

# ---------------------------------------------------------------------------
# COLUNAS "BASE" USADAS PARA GERAR MÉTRICAS POR 90 MINUTOS
# ---------------------------------------------------------------------------
# Formato: (coluna_bruta, nome_da_nova_coluna_por_90)
# A função build_per_90_metrics (data_cleaning.py) só cria a métrica se
# AMBAS as colunas (a bruta e "min") existirem no DataFrame.
#
# Gls, Ast e G+A já vêm prontos por 90 no CSV original (gls_per90 etc.),
# então aqui geramos por 90 apenas para estatísticas que NÃO vêm prontas:
# cartões e pênaltis. Isso demonstra o cálculo de forma real, sem duplicar
# o que o FBref já fornece.
PER_90_BASE_COLUMNS = [
    ("g_minus_pk", "g_minus_pk_per90_calc"),
    ("pk", "pk_per90"),
    ("pkatt", "pkatt_per90"),
    ("crdy", "crdy_per90"),
    ("crdr", "crdr_per90"),
]

# ---------------------------------------------------------------------------
# FEATURES USADAS NA ANÁLISE DE JOGADORES (rankings, comparações, etc.)
# ---------------------------------------------------------------------------
# Apenas nomes que EXISTEM (depois da limpeza) devem ser usados de fato;
# quem filtra isso é a função check_required_columns (data_cleaning.py).
PLAYER_ANALYSIS_FEATURES = [
    "gls",
    "ast",
    "g_a",
    "gls_per90",
    "ast_per90",
    "g_a_per90",
    "crdy",
    "crdr",
    "min",
]

# ---------------------------------------------------------------------------
# FEATURES USADAS NO CLUSTERING (K-Means / PCA / Similaridade)
# ---------------------------------------------------------------------------
# Como o CSV não tem estatísticas de criação/progressão/defesa, o
# agrupamento é feito com o que está disponível: produção ofensiva por 90
# minutos e disciplina (cartões) por 90 minutos. Isso é documentado no
# README como uma limitação do dataset, não do método.
CLUSTER_FEATURES = [
    "gls_per90",
    "ast_per90",
    "g_a_per90",
    "crdy_per90",
    "crdr_per90",
]
