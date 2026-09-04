"""
analysis.py
============

Funções de análise estatística sobre os dados já limpos (o DataFrame
que sai de prepare_players_data, em data_cleaning.py).

Todas as funções verificam se as colunas necessárias existem antes de
usá-las, e avisam claramente quando uma análise não pode ser feita.
"""

from src.data_cleaning import check_required_columns, filter_by_minutes  # reaproveitado aqui também


def get_basic_statistics(df):
    """
    Retorna estatísticas descritivas básicas (média, desvio padrão, min,
    max, quartis) para todas as colunas numéricas do DataFrame.

    Equivale a um df.describe(), mas isolado em função própria para ser
    reaproveitado no dashboard e nos notebooks.
    """
    return df.describe()


def get_top_players_by_stat(df, stat, n=10):
    """
    Retorna os "n" jogadores com maior valor na coluna "stat".

    Se a coluna "stat" não existir, avisa e retorna um DataFrame vazio
    em vez de quebrar a aplicação.
    """
    verificacao = check_required_columns(df, [stat])
    if stat not in verificacao["disponiveis"]:
        return df.iloc[0:0]

    colunas_exibicao = [c for c in ["player", "squad", "pos", stat] if c in df.columns]
    return df.sort_values(by=stat, ascending=False).head(n)[colunas_exibicao]


def get_average_by_position(df, stat):
    """
    Retorna a média da coluna "stat", agrupada por posição ("pos").

    Requer que as colunas "pos" e "stat" existam.
    """
    verificacao = check_required_columns(df, ["pos", stat])
    if verificacao["faltando"]:
        return None

    return df.groupby("pos")[stat].mean().sort_values(ascending=False)


def get_average_by_team(df, stat):
    """
    Retorna a média da coluna "stat", agrupada por clube ("squad").

    Requer que as colunas "squad" e "stat" existam.
    """
    verificacao = check_required_columns(df, ["squad", stat])
    if verificacao["faltando"]:
        return None

    return df.groupby("squad")[stat].mean().sort_values(ascending=False)


def get_player_statistics(df, player_name):
    """
    Retorna todas as estatísticas disponíveis de um único jogador.

    Faz uma busca "case-insensitive" e parcial pelo nome, para facilitar
    o uso no dashboard (não precisa digitar o nome exato).

    Retorna None se o jogador não for encontrado.
    """
    if "player" not in df.columns:
        print("Aviso: coluna 'player' não encontrada -- não é possível buscar por jogador.")
        return None

    resultado = df[df["player"].str.contains(player_name, case=False, na=False)]

    if resultado.empty:
        print(f"Nenhum jogador encontrado com o nome '{player_name}'.")
        return None

    return resultado


def get_top_scorers(df, n=10):
    """Atalho para o ranking dos artilheiros (maior número de gols)."""
    return get_top_players_by_stat(df, "gls", n=n)


def get_top_assists(df, n=10):
    """Atalho para o ranking de assistências."""
    return get_top_players_by_stat(df, "ast", n=n)


def get_club_overview(df):
    """
    Retorna uma tabela-resumo por clube: quantidade de jogadores, idade
    média e totais/médias das principais estatísticas ofensivas.

    Só inclui colunas que realmente existem no DataFrame, seguindo a
    mesma filosofia defensiva do resto do projeto.
    """
    verificacao = check_required_columns(df, ["squad"])
    if verificacao["faltando"]:
        return None

    agregacoes = {}
    if "player" in df.columns:
        agregacoes["player"] = "nunique"
    if "age" in df.columns:
        agregacoes["age"] = "mean"
    if "gls" in df.columns:
        agregacoes["gls"] = "sum"
    if "ast" in df.columns:
        agregacoes["ast"] = "sum"
    if "gls_per90" in df.columns:
        agregacoes["gls_per90"] = "mean"
    if "ast_per90" in df.columns:
        agregacoes["ast_per90"] = "mean"
    if "crdy" in df.columns:
        agregacoes["crdy"] = "sum"
    if "crdr" in df.columns:
        agregacoes["crdr"] = "sum"

    if not agregacoes:
        return None

    resumo = df.groupby("squad").agg(agregacoes).round(2)

    renomear = {
        "player": "jogadores",
        "age": "idade_media",
        "gls": "gols_totais",
        "ast": "assistencias_totais",
        "gls_per90": "gols_por90_media",
        "ast_per90": "assist_por90_media",
        "crdy": "cartoes_amarelos",
        "crdr": "cartoes_vermelhos",
    }
    resumo = resumo.rename(columns=renomear)

    coluna_ordenacao = "gols_totais" if "gols_totais" in resumo.columns else resumo.columns[0]
    return resumo.sort_values(by=coluna_ordenacao, ascending=False).reset_index()


def compute_scouting_score(df, weights, minimum_minutes_applied=True):
    """
    Calcula um "score" simples e personalizável por jogador, combinando
    várias estatísticas por 90 minutos com pesos escolhidos pelo usuário.

    "weights" é um dicionário {nome_da_coluna: peso}. Cada coluna é
    normalizada para a escala 0-1 (min-max) antes de aplicar o peso, para
    que estatísticas em escalas diferentes (ex: gols/90 vs cartões/90)
    não dominem o score só por causa da escala.

    Pesos negativos são permitidos (ex: penalizar cartões). O resultado
    final também é reescalado para 0-100, só para ficar mais legível.

    Retorna um DataFrame ordenado do maior para o menor score, ou None se
    nenhuma das colunas pedidas em "weights" existir.
    """
    colunas_pedidas = list(weights.keys())
    verificacao = check_required_columns(df, colunas_pedidas)
    colunas_validas = verificacao["disponiveis"]

    if not colunas_validas:
        return None

    df_score = df.dropna(subset=colunas_validas).copy()
    if df_score.empty:
        return None

    score_bruto = 0
    for coluna in colunas_validas:
        peso = weights[coluna]
        minimo, maximo = df_score[coluna].min(), df_score[coluna].max()
        if maximo == minimo:
            normalizado = 0
        else:
            normalizado = (df_score[coluna] - minimo) / (maximo - minimo)
        score_bruto = score_bruto + normalizado * peso

    minimo_score, maximo_score = score_bruto.min(), score_bruto.max()
    if maximo_score == minimo_score:
        df_score["score"] = 50.0
    else:
        df_score["score"] = ((score_bruto - minimo_score) / (maximo_score - minimo_score) * 100).round(1)

    colunas_exibicao = [c for c in ["player", "squad", "pos"] if c in df_score.columns] + colunas_validas + ["score"]
    return df_score[colunas_exibicao].sort_values(by="score", ascending=False).reset_index(drop=True)


def prepare_radar_data(df, player_names, features):
    """
    Prepara os dados para um gráfico radar comparando 2 ou mais
    jogadores nas features escolhidas.

    Cada feature é normalizada para 0-100 (min-max, calculado sobre TODO
    o DataFrame recebido, não só os jogadores comparados) para que o
    radar fique legível mesmo comparando estatísticas em escalas
    diferentes.

    Retorna
    -------
    (df_radar, features_usadas) onde df_radar tem uma linha por jogador
    e uma coluna por feature (valores 0-100), ou (None, []) se não houver
    dados suficientes.
    """
    verificacao = check_required_columns(df, features)
    features_usadas = verificacao["disponiveis"]

    if "player" not in df.columns or len(features_usadas) < 3:
        return None, []

    df_normalizado = df.dropna(subset=features_usadas).copy()

    for feature in features_usadas:
        minimo, maximo = df_normalizado[feature].min(), df_normalizado[feature].max()
        if maximo == minimo:
            df_normalizado[feature] = 50.0
        else:
            df_normalizado[feature] = (df_normalizado[feature] - minimo) / (maximo - minimo) * 100

    linhas = df_normalizado[df_normalizado["player"].isin(player_names)]
    if linhas.empty:
        return None, []

    return linhas[["player"] + features_usadas].reset_index(drop=True), features_usadas
