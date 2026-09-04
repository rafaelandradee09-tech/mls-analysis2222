"""
data_cleaning.py
=================

Funções para limpar e preparar os dados dos jogadores.

Regra geral seguida em todo este arquivo: NUNCA assumir que uma coluna
existe. Sempre verificar antes de usar. Isso é importante porque o CSV
do FBref pode mudar (colunas a mais, a menos, ou com nomes diferentes).
"""

import numpy as np
import pandas as pd

from src.config import COLUMN_RENAME_MAP, COLUMNS_TO_DROP, PER_90_BASE_COLUMNS


def clean_column_names(df):
    """
    Renomeia as colunas do CSV original para nomes internos mais simples,
    usando o mapeamento definido em src/config.py (COLUMN_RENAME_MAP).

    Só renomeia as colunas que realmente existem no DataFrame -- não
    quebra se alguma coluna do mapa não estiver presente.
    """
    df = df.copy()

    # Remove colunas que não têm valor estatístico (link "Matches" e o
    # id interno do FBref), apenas se elas existirem.
    colunas_para_remover = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=colunas_para_remover)

    # Renomeia apenas o que existir no DataFrame.
    mapa_existente = {
        original: novo for original, novo in COLUMN_RENAME_MAP.items() if original in df.columns
    }
    df = df.rename(columns=mapa_existente)

    return df


def convert_numeric_columns(df):
    """
    Converte colunas que deveriam ser numéricas, mas podem ter vindo como
    texto (ex: "age" no formato "22-357", onde 22 são anos e 357 dias).

    Colunas que já são numéricas são deixadas como estão.
    """
    df = df.copy()

    # A coluna "age" do FBref vem como "anos-dias" (ex: "22-357").
    # Extraímos só a parte de anos, que é o que interessa para a análise.
    if "age" in df.columns:
        idade_em_anos = df["age"].astype(str).str.split("-").str[0]
        df["age"] = pd.to_numeric(idade_em_anos, errors="coerce")

    # "born_year" já costuma ser numérico, mas garantimos a conversão.
    if "born_year" in df.columns:
        df["born_year"] = pd.to_numeric(df["born_year"], errors="coerce")

    # Demais colunas conhecidas que devem ser numéricas, se existirem.
    colunas_numericas_esperadas = [
        "mp", "starts", "min", "nineties",
        "gls", "ast", "g_a", "g_minus_pk", "pk", "pkatt", "crdy", "crdr",
        "gls_per90", "ast_per90", "g_a_per90", "g_minus_pk_per90", "g_a_minus_pk_per90",
    ]
    for coluna in colunas_numericas_esperadas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    return df


def handle_missing_values(df):
    """
    Trata valores ausentes de forma conservadora.

    NÃO substitui tudo por zero, pois isso distorceria estatísticas
    (ex: um jogador sem "age" registrada não tem idade zero, tem idade
    desconhecida). A estratégia é:

    - Colunas de identificação (player, squad, pos, nation): linhas sem
      "player" são removidas (não há como analisar um jogador sem nome).
    - Colunas numéricas de contagem (gls, ast, cartões, etc.): valores
      ausentes viram 0 SOMENTE se fizer sentido (uma estatística de
      contagem ausente normalmente significa "não aconteceu").
    - Colunas como "age" e "nation": valores ausentes são mantidos como
      NaN, porque não sabemos o valor real e não devemos inventar.
    """
    df = df.copy()

    if "player" in df.columns:
        antes = len(df)
        df = df[df["player"].notna()]
        removidos = antes - len(df)
        if removidos > 0:
            print(f"Removidas {removidos} linha(s) sem nome de jogador.")

    # Estatísticas de contagem: ausência plausivelmente significa "zero
    # eventos registrados", então preenchemos com 0 apenas essas colunas.
    colunas_contagem = ["gls", "ast", "g_a", "g_minus_pk", "pk", "pkatt", "crdy", "crdr"]
    for coluna in colunas_contagem:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna(0)

    # Colunas como "age", "born_year", "nation" NÃO são preenchidas --
    # ficam como NaN para não distorcer médias e comparações.

    return df


def remove_invalid_rows(df):
    """
    Remove linhas claramente inválidas para análise, como jogadores com
    0 minutos jogados (não há estatística possível de se analisar) ou
    linhas totalmente duplicadas.
    """
    df = df.copy()

    antes = len(df)
    df = df.drop_duplicates()
    duplicadas = antes - len(df)
    if duplicadas > 0:
        print(f"Removidas {duplicadas} linha(s) duplicada(s).")

    if "min" in df.columns:
        antes = len(df)
        df = df[df["min"] > 0]
        sem_minutos = antes - len(df)
        if sem_minutos > 0:
            print(f"Removidas {sem_minutos} linha(s) com 0 minutos jogados.")

    return df.reset_index(drop=True)


def check_required_columns(df, required_columns):
    """
    Verifica quais colunas de "required_columns" existem no DataFrame.

    Nunca lança um erro que interrompa a aplicação -- em vez disso,
    retorna quais colunas faltam, para que a análise que as pede possa
    avisar o usuário e continuar funcionando com o que está disponível.

    Retorna
    -------
    dict com duas chaves:
        "disponiveis": lista de colunas que existem
        "faltando": lista de colunas que não existem
    """
    disponiveis = [c for c in required_columns if c in df.columns]
    faltando = [c for c in required_columns if c not in df.columns]

    if faltando:
        print(
            "Aviso: as seguintes colunas não estão disponíveis neste "
            f"dataset e serão ignoradas: {faltando}"
        )

    return {"disponiveis": disponiveis, "faltando": faltando}


def build_per_90_metrics(df, base_columns=None):
    """
    Cria métricas "por 90 minutos" de forma genérica.

    Para cada par (coluna_bruta, nome_novo) em base_columns, a métrica só
    é criada se AMBAS "coluna_bruta" e "min" existirem no DataFrame, e o
    cálculo evita divisão por zero (jogadores com 0 minutos ficam com a
    métrica ausente, não com infinito).

    Fórmula: valor_por_90 = coluna_bruta / min * 90
    """
    if base_columns is None:
        base_columns = PER_90_BASE_COLUMNS

    df = df.copy()

    if "min" not in df.columns:
        print("Aviso: coluna 'min' não encontrada -- métricas por 90 minutos não podem ser criadas.")
        return df

    for coluna_bruta, nome_novo in base_columns:
        if coluna_bruta not in df.columns:
            continue
        # np.where evita divisão por zero: onde min == 0, o resultado fica NaN.
        df[nome_novo] = np.where(
            df["min"] > 0,
            df[coluna_bruta] / df["min"] * 90,
            np.nan,
        )
        df[nome_novo] = df[nome_novo].round(2)

    return df


def filter_by_minutes(df, minimum_minutes):
    """
    Filtra jogadores com pelo menos "minimum_minutes" minutos jogados.

    Isso evita comparar diretamente um jogador que jogou 20 minutos com
    um que jogou a temporada inteira -- estatísticas por 90 minutos de
    amostras muito pequenas são pouco confiáveis.
    """
    if "min" not in df.columns:
        print("Aviso: coluna 'min' não encontrada -- filtro de minutos não aplicado.")
        return df

    return df[df["min"] >= minimum_minutes].reset_index(drop=True)


def prepare_players_data(df):
    """
    Executa o processo principal de limpeza, na ordem correta:

    1. Renomear colunas para nomes internos.
    2. Converter colunas para tipos numéricos corretos.
    3. Tratar valores ausentes de forma conservadora.
    4. Remover linhas inválidas (duplicadas ou sem minutos jogados).
    5. Criar métricas adicionais por 90 minutos.

    Retorna um novo DataFrame -- o original passado como argumento nunca
    é modificado "in place".
    """
    df = clean_column_names(df)
    df = convert_numeric_columns(df)
    df = handle_missing_values(df)
    df = remove_invalid_rows(df)
    df = build_per_90_metrics(df)
    return df
