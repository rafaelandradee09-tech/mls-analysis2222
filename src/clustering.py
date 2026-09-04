"""
clustering.py
==============

Funções para agrupar jogadores com características estatísticas
semelhantes, usando K-Means, e para visualizar esses grupos em duas
dimensões usando PCA.

Por que normalizar (StandardScaler) antes do K-Means?
-------------------------------------------------------
O K-Means agrupa pontos com base na DISTÂNCIA entre eles. Se uma feature
está em uma escala muito maior que outra (por exemplo, minutos jogados,
que vai de 0 a milhares, comparado a gols por 90, que vai de 0 a ~1), a
feature de escala maior domina o cálculo de distância e as outras quase
não importam. O StandardScaler resolve isso transformando cada feature
para ter média 0 e desvio padrão 1, colocando todas na mesma escala.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data_cleaning import check_required_columns


def prepare_clustering_data(df, features):
    """
    Prepara os dados para o clustering:

    1. Verifica quais features realmente existem no DataFrame.
    2. Remove linhas com valores ausentes nessas features (o K-Means não
       funciona com NaN).
    3. Retorna tanto o DataFrame filtrado (com todas as colunas
       originais, útil para depois interpretar os clusters) quanto a
       matriz X somente com as features numéricas.

    Retorna
    -------
    (df_filtrado, X, features_usadas)
    """
    verificacao = check_required_columns(df, features)
    features_usadas = verificacao["disponiveis"]

    if not features_usadas:
        print("Nenhuma das features solicitadas está disponível. Clustering não pode ser feito.")
        return df.iloc[0:0], pd.DataFrame(), []

    df_filtrado = df.dropna(subset=features_usadas).reset_index(drop=True)
    X = df_filtrado[features_usadas].copy()

    return df_filtrado, X, features_usadas


def scale_features(X):
    """
    Normaliza as features usando StandardScaler (média 0, desvio padrão 1).

    Retorna a matriz normalizada (numpy array) e o scaler já ajustado,
    caso seja necessário reaproveitá-lo depois.
    """
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)
    return X_normalizado, scaler


def find_optimal_clusters(X, max_clusters=10):
    """
    Calcula a inércia (soma das distâncias ao quadrado dentro de cada
    cluster) e o Silhouette Score para diferentes valores de K, de 2 até
    "max_clusters".

    Esses valores ajudam a escolher um número de clusters razoável
    (Elbow Method + Silhouette Score), mas NÃO substituem a interpretação
    humana do que os clusters significam.

    Retorna um DataFrame com colunas: k, inertia, silhouette_score
    """
    resultados = []
    max_k_possivel = min(max_clusters, len(X) - 1)

    for k in range(2, max_k_possivel + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        rotulos = modelo.fit_predict(X)
        resultados.append(
            {
                "k": k,
                "inertia": modelo.inertia_,
                "silhouette_score": silhouette_score(X, rotulos),
            }
        )

    return pd.DataFrame(resultados)


def perform_kmeans(X, n_clusters=4):
    """
    Executa o K-Means com "n_clusters" grupos.

    Usa random_state=42 para que o resultado seja sempre o mesmo ao
    rodar novamente com os mesmos dados (reprodutibilidade).

    Retorna os rótulos de cluster (um número de cluster para cada linha
    de X) e o modelo já treinado.
    """
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rotulos = modelo.fit_predict(X)
    return rotulos, modelo


def apply_pca(X):
    """
    Reduz as features para 2 dimensões (PC1 e PC2) usando PCA, apenas
    para fins de VISUALIZAÇÃO -- o PCA não é usado para o clustering em
    si, só para conseguirmos "ver" os grupos em um gráfico 2D.

    Retorna um DataFrame com colunas PC1 e PC2.
    """
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X)
    return pd.DataFrame(componentes, columns=["PC1", "PC2"])


def summarize_clusters(df_com_clusters, features, cluster_column="cluster"):
    """
    Resume as características de cada cluster para ajudar a interpretá-lo.

    Para cada cluster, mostra:
    - quantidade de jogadores;
    - média de cada feature usada no clustering;
    - as posições mais frequentes;
    - os clubes mais frequentes (se essas colunas existirem).

    Importante: esta função NÃO dá nomes automáticos aos clusters
    (tipo "Cluster 0 = atacantes"). Ela só organiza os dados para que a
    interpretação seja feita observando os números reais.
    """
    resumos = []

    for numero_cluster in sorted(df_com_clusters[cluster_column].unique()):
        grupo = df_com_clusters[df_com_clusters[cluster_column] == numero_cluster]

        resumo = {"cluster": numero_cluster, "quantidade_jogadores": len(grupo)}

        for feature in features:
            if feature in grupo.columns:
                resumo[f"media_{feature}"] = round(grupo[feature].mean(), 3)

        if "pos" in grupo.columns:
            resumo["posicoes_mais_frequentes"] = grupo["pos"].value_counts().head(3).to_dict()

        if "squad" in grupo.columns:
            resumo["clubes_mais_frequentes"] = grupo["squad"].value_counts().head(3).to_dict()

        resumos.append(resumo)

    return pd.DataFrame(resumos)
