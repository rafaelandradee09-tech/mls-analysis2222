"""
similarity.py
==============

Encontra jogadores com características estatísticas semelhantes a um
jogador escolhido, usando Nearest Neighbors.

Importante: "semelhante" aqui significa apenas "próximo no espaço das
features utilizadas", depois de normalizar os dados. Isso NÃO é uma
avaliação de qualidade -- não dizemos que um jogador é "melhor" ou
"pior" que outro, só que os números deles se parecem.
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data_cleaning import check_required_columns


def find_similar_players(df, player_name, features, n_neighbors=5):
    """
    Encontra os "n_neighbors" jogadores mais parecidos com "player_name",
    considerando as colunas em "features".

    Passo a passo:
    1. Verifica quais features existem de fato no DataFrame.
    2. Remove jogadores com dados ausentes nessas features.
    3. Localiza o jogador escolhido dentro dos dados filtrados.
    4. Normaliza os dados (StandardScaler) -- necessário para que
       nenhuma feature domine o cálculo de distância só por ter uma
       escala maior.
    5. Usa NearestNeighbors para calcular as distâncias entre todos os
       jogadores.
    6. Retorna os mais próximos do jogador escolhido, sem incluir ele
       mesmo na lista.

    Retorna
    -------
    pandas.DataFrame com os jogadores semelhantes e a coluna "distancia",
    ou None se o jogador não for encontrado ou não houver features
    suficientes.
    """
    if "player" not in df.columns:
        print("Aviso: coluna 'player' não encontrada -- não é possível buscar jogadores.")
        return None

    verificacao = check_required_columns(df, features)
    features_usadas = verificacao["disponiveis"]

    if len(features_usadas) < 2:
        print(
            "Aviso: são necessárias pelo menos 2 features válidas para calcular "
            "similaridade de forma minimamente confiável."
        )
        return None

    dados = df.dropna(subset=features_usadas).reset_index(drop=True)

    jogador_encontrado = dados[dados["player"].str.lower() == player_name.lower()]
    if jogador_encontrado.empty:
        # Tenta uma busca parcial, caso o nome exato não bata.
        jogador_encontrado = dados[dados["player"].str.contains(player_name, case=False, na=False)]

    if jogador_encontrado.empty:
        print(f"Jogador '{player_name}' não encontrado (ou não possui dados completos nas features usadas).")
        return None

    indice_jogador = jogador_encontrado.index[0]
    nome_real = dados.loc[indice_jogador, "player"]

    if len(dados) <= n_neighbors:
        print("Aviso: poucos jogadores disponíveis com dados completos para essa busca.")

    X = dados[features_usadas]
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)

    # +1 porque o próprio jogador será retornado como o vizinho mais
    # próximo dele mesmo (distância 0), e vamos removê-lo depois.
    k = min(n_neighbors + 1, len(dados))
    modelo = NearestNeighbors(n_neighbors=k)
    modelo.fit(X_normalizado)

    distancias, indices_vizinhos = modelo.kneighbors([X_normalizado[dados.index.get_loc(indice_jogador)]])

    colunas_exibicao = [c for c in ["player", "squad", "pos"] if c in dados.columns]
    resultado = dados.loc[indices_vizinhos[0], colunas_exibicao].copy()
    resultado["distancia"] = distancias[0]

    # Remove o próprio jogador da lista de semelhantes.
    resultado = resultado[resultado["player"] != nome_real]

    return resultado.reset_index(drop=True)
