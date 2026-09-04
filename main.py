"""
main.py
========

Pipeline principal do projeto. Executa, em sequência:

1. Carregar o CSV original.
2. Fazer uma análise inicial (estrutura do arquivo).
3. Limpar e preparar os dados.
4. Salvar o resultado em data/processed/mls_players_processed.csv.

O clustering NÃO é executado aqui, porque no dashboard o usuário pode
escolher o número de clusters interativamente -- rodar clustering fixo
aqui não faria sentido.

Como executar:
    python main.py
"""

from src.data_loader import load_players_data, save_processed_data
from src.data_cleaning import prepare_players_data

CAMINHO_CSV_ORIGINAL = "data/raw/mls_players.csv"
CAMINHO_CSV_PROCESSADO = "data/processed/mls_players_processed.csv"


def analisar_estrutura_inicial(df):
    """Mostra informações básicas sobre o CSV, antes de qualquer limpeza."""
    print("=" * 60)
    print("ANÁLISE INICIAL DO CSV")
    print("=" * 60)
    print(f"Quantidade de linhas: {df.shape[0]}")
    print(f"Quantidade de colunas: {df.shape[1]}")
    print(f"\nColunas encontradas:\n{list(df.columns)}")
    print(f"\nValores ausentes por coluna:\n{df.isnull().sum()}")
    print()


def main():
    print("Carregando dados originais...")
    df_original = load_players_data(CAMINHO_CSV_ORIGINAL)

    analisar_estrutura_inicial(df_original)

    print("Limpando e preparando os dados...")
    df_processado = prepare_players_data(df_original)

    print(f"\nDados processados: {df_processado.shape[0]} linhas, {df_processado.shape[1]} colunas.")
    print(f"Colunas finais: {list(df_processado.columns)}")

    save_processed_data(df_processado, CAMINHO_CSV_PROCESSADO)

    print("\nPipeline concluído com sucesso!")
    print("Para explorar os dados interativamente, execute:")
    print("    streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
