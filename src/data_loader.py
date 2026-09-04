"""
data_loader.py
===============

Responsável apenas por LER o CSV original e SALVAR o CSV processado.
Não faz limpeza nem análise -- isso fica em outros arquivos, cada um
com uma responsabilidade só (mais fácil de entender e testar).
"""

import os
import pandas as pd


def load_players_data(path):
    """
    Carrega o CSV de jogadores da MLS exportado do FBref.

    O arquivo do FBref tem um cabeçalho "duplo" (duas linhas de título),
    por isso usamos header=[0, 1] e depois "achatamos" essas duas linhas
    em um único nome de coluna por vez (ex: "Playing Time" + "Min" vira
    "Playing Time_Min"). Esse nome ainda vai passar por
    clean_column_names() em data_cleaning.py para virar algo mais curto.

    Parâmetros
    ----------
    path : str
        Caminho para o arquivo CSV (ex: "data/raw/mls_players.csv").

    Retorna
    -------
    pandas.DataFrame
        Os dados exatamente como estão no CSV, sem nenhuma limpeza.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Não foi possível encontrar o arquivo '{path}'. "
            "Verifique se o CSV do FBref foi colocado em data/raw/mls_players.csv."
        )

    try:
        # header=[0, 1]: o FBref exporta duas linhas de cabeçalho
        # (grupo da estatística, ex: "Performance", e o nome dela, ex: "Gls").
        df = pd.read_csv(path, header=[0, 1])
    except Exception as erro:
        raise ValueError(f"Erro ao ler o CSV em '{path}': {erro}")

    # Junta as duas linhas de cabeçalho em uma só string por coluna.
    # Se a primeira parte é um "Unnamed" (coluna sem grupo, ex: "Player"),
    # usamos só o nome da segunda linha.
    colunas_achatadas = []
    for nivel_1, nivel_2 in df.columns:
        if str(nivel_1).startswith("Unnamed"):
            colunas_achatadas.append(str(nivel_2))
        else:
            colunas_achatadas.append(f"{nivel_1}_{nivel_2}")
    df.columns = colunas_achatadas

    if df.empty:
        raise ValueError(f"O arquivo '{path}' foi lido, mas está vazio.")

    return df


def save_processed_data(df, path):
    """
    Salva o DataFrame processado em disco (data/processed/...).

    Nunca deve ser usada para sobrescrever o CSV original -- por isso o
    caminho padrão do projeto aponta sempre para data/processed/.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame já limpo/processado.
    path : str
        Caminho de destino (ex: "data/processed/mls_players_processed.csv").
    """
    pasta = os.path.dirname(path)
    if pasta and not os.path.exists(pasta):
        os.makedirs(pasta, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Dados processados salvos em: {path} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
