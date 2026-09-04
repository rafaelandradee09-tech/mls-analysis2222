"""
test_clustering.py
====================

Testes básicos para as funções de src/clustering.py.

Usamos DataFrames pequenos e fictícios, criados só para teste.
"""

import numpy as np
import pandas as pd

from src.clustering import apply_pca, perform_kmeans, prepare_clustering_data, scale_features


def _dataframe_fake():
    """Cria um DataFrame fictício com dois grupos bem separados de jogadores."""
    return pd.DataFrame(
        {
            "player": [f"Jogador {i}" for i in range(10)],
            "pos": ["FW"] * 5 + ["DF"] * 5,
            # Grupo 1: muitos gols por 90. Grupo 2: poucos gols por 90.
            "gls_per90": [0.9, 0.8, 0.85, 0.95, 0.87, 0.02, 0.01, 0.03, 0.0, 0.02],
            "ast_per90": [0.3, 0.25, 0.28, 0.31, 0.29, 0.05, 0.04, 0.02, 0.03, 0.01],
        }
    )


def test_prepare_clustering_data_seleciona_apenas_features_existentes():
    df = _dataframe_fake()
    df_filtrado, X, features_usadas = prepare_clustering_data(df, ["gls_per90", "ast_per90", "xg_per90"])

    assert features_usadas == ["gls_per90", "ast_per90"]
    assert list(X.columns) == ["gls_per90", "ast_per90"]
    assert len(df_filtrado) == 10


def test_prepare_clustering_data_remove_linhas_com_nan():
    df = _dataframe_fake()
    df.loc[0, "gls_per90"] = np.nan

    df_filtrado, X, _ = prepare_clustering_data(df, ["gls_per90", "ast_per90"])

    assert len(df_filtrado) == 9
    assert X.isnull().sum().sum() == 0


def test_scale_features_gera_media_zero_e_desvio_um():
    df = _dataframe_fake()
    _, X, _ = prepare_clustering_data(df, ["gls_per90", "ast_per90"])
    X_normalizado, _ = scale_features(X)

    media = X_normalizado.mean(axis=0)
    desvio = X_normalizado.std(axis=0)

    assert np.allclose(media, 0, atol=1e-8)
    assert np.allclose(desvio, 1, atol=1e-8)


def test_perform_kmeans_cria_dois_clusters_bem_separados():
    df = _dataframe_fake()
    _, X, _ = prepare_clustering_data(df, ["gls_per90", "ast_per90"])
    X_normalizado, _ = scale_features(X)

    rotulos, modelo = perform_kmeans(X_normalizado, n_clusters=2)

    # Deve haver exatamente 2 clusters diferentes.
    assert len(set(rotulos)) == 2
    # Os 5 primeiros jogadores (muitos gols) devem cair no mesmo cluster,
    # assim como os 5 últimos (poucos gols).
    assert len(set(rotulos[:5])) == 1
    assert len(set(rotulos[5:])) == 1


def test_perform_kmeans_e_reproduzivel():
    df = _dataframe_fake()
    _, X, _ = prepare_clustering_data(df, ["gls_per90", "ast_per90"])
    X_normalizado, _ = scale_features(X)

    rotulos_1, _ = perform_kmeans(X_normalizado, n_clusters=2)
    rotulos_2, _ = perform_kmeans(X_normalizado, n_clusters=2)

    # random_state=42 garante que o resultado é sempre o mesmo.
    assert list(rotulos_1) == list(rotulos_2)


def test_apply_pca_retorna_duas_colunas():
    df = _dataframe_fake()
    _, X, _ = prepare_clustering_data(df, ["gls_per90", "ast_per90"])
    X_normalizado, _ = scale_features(X)

    componentes = apply_pca(X_normalizado)

    assert list(componentes.columns) == ["PC1", "PC2"]
    assert len(componentes) == len(df)
