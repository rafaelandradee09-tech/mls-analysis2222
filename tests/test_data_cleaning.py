"""
test_data_cleaning.py
=======================

Testes básicos para as funções de src/data_cleaning.py.

Usamos DataFrames pequenos e fictícios criados só para teste -- eles
NUNCA são usados como dados reais da aplicação.
"""

import numpy as np
import pandas as pd

from src.data_cleaning import (
    build_per_90_metrics,
    check_required_columns,
    clean_column_names,
    convert_numeric_columns,
    filter_by_minutes,
    handle_missing_values,
    remove_invalid_rows,
)


def test_clean_column_names_renomeia_colunas_conhecidas():
    df = pd.DataFrame({"Player": ["Jogador A"], "Performance_Gls": [5], "Matches": ["Matches"]})
    df_limpo = clean_column_names(df)

    assert "player" in df_limpo.columns
    assert "gls" in df_limpo.columns
    # "Matches" deve ser removida, pois não carrega informação estatística.
    assert "Matches" not in df_limpo.columns


def test_clean_column_names_nao_quebra_com_colunas_desconhecidas():
    df = pd.DataFrame({"ColunaQualquer": [1, 2, 3]})
    df_limpo = clean_column_names(df)
    # Coluna sem mapeamento deve continuar existindo, sem erro.
    assert "ColunaQualquer" in df_limpo.columns


def test_convert_numeric_columns_converte_idade_no_formato_fbref():
    df = pd.DataFrame({"age": ["22-357", "24-319"]})
    df_convertido = convert_numeric_columns(df)

    assert df_convertido["age"].iloc[0] == 22
    assert df_convertido["age"].iloc[1] == 24


def test_handle_missing_values_remove_linhas_sem_nome_de_jogador():
    df = pd.DataFrame({"player": ["Jogador A", None], "gls": [1, 2]})
    df_tratado = handle_missing_values(df)

    assert len(df_tratado) == 1
    assert df_tratado["player"].iloc[0] == "Jogador A"


def test_handle_missing_values_preenche_estatisticas_de_contagem_com_zero():
    df = pd.DataFrame({"player": ["Jogador A"], "gls": [np.nan]})
    df_tratado = handle_missing_values(df)

    assert df_tratado["gls"].iloc[0] == 0


def test_remove_invalid_rows_remove_duplicadas_e_sem_minutos():
    df = pd.DataFrame({"player": ["A", "A", "B"], "min": [500, 500, 0]})
    df_limpo = remove_invalid_rows(df)

    assert len(df_limpo) == 1
    assert df_limpo["player"].iloc[0] == "A"


def test_check_required_columns_identifica_colunas_faltando():
    df = pd.DataFrame({"gls": [1, 2]})
    resultado = check_required_columns(df, ["gls", "xg"])

    assert resultado["disponiveis"] == ["gls"]
    assert resultado["faltando"] == ["xg"]


def test_build_per_90_metrics_calcula_corretamente():
    df = pd.DataFrame({"min": [180], "crdy": [2]})
    df_com_metrica = build_per_90_metrics(df, base_columns=[("crdy", "crdy_per90")])

    # 2 cartões em 180 minutos = 1 cartão por 90 minutos.
    assert df_com_metrica["crdy_per90"].iloc[0] == 1.0


def test_build_per_90_metrics_evita_divisao_por_zero():
    df = pd.DataFrame({"min": [0], "crdy": [2]})
    df_com_metrica = build_per_90_metrics(df, base_columns=[("crdy", "crdy_per90")])

    assert pd.isna(df_com_metrica["crdy_per90"].iloc[0])


def test_build_per_90_metrics_nao_cria_coluna_se_base_nao_existir():
    df = pd.DataFrame({"min": [180]})
    df_resultado = build_per_90_metrics(df, base_columns=[("coluna_inexistente", "coluna_per90")])

    assert "coluna_per90" not in df_resultado.columns


def test_filter_by_minutes_filtra_corretamente():
    df = pd.DataFrame({"player": ["A", "B", "C"], "min": [1000, 200, 600]})
    df_filtrado = filter_by_minutes(df, 500)

    assert set(df_filtrado["player"]) == {"A", "C"}
