"""
dashboard.py
=============

Dashboard interativo do projeto "MLS Football Data Analysis", feito com
Streamlit.

Como executar (a partir da raiz do projeto):
    streamlit run app/dashboard.py

O dashboard lê o CSV processado (data/processed/mls_players_processed.csv).
Se ele ainda não existir, rode antes:
    python main.py

Páginas disponíveis:
    - Overview: números gerais e distribuições da base filtrada.
    - Análise de jogador: estatísticas individuais de um jogador.
    - Rankings: top jogadores por estatística escolhida.
    - Comparar jogadores: radar comparando 2-3 jogadores lado a lado.
    - Clubes: resumo agregado por clube.
    - Clustering: agrupamento K-Means + visualização PCA.
    - Score personalizado: ranking com pesos definidos pelo usuário.
    - Jogadores semelhantes: busca por Nearest Neighbors.
    - Sobre: metodologia, fonte dos dados e limitações do projeto.
"""

import os
import sys

import pandas as pd
import streamlit as st

from PIL import Image

# Permite importar os módulos de "src" mesmo quando o Streamlit executa
# este arquivo de dentro da pasta "app".
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components import (
    download_csv_button,
    plot_bar_plotly,
    plot_cluster_scatter_plotly,
    plot_histogram_plotly,
    plot_radar_chart,
    show_cluster_summary,
    show_player_info,
    show_ranking_table,
    show_similar_players,
)
from src.analysis import (
    compute_scouting_score,
    get_average_by_position,
    get_club_overview,
    get_player_statistics,
    get_top_players_by_stat,
    prepare_radar_data,
)
from src.clustering import apply_pca, perform_kmeans, prepare_clustering_data, scale_features, summarize_clusters
from src.config import CLUSTER_FEATURES
from src.data_cleaning import filter_by_minutes
from src.similarity import find_similar_players

CAMINHO_DADOS_PROCESSADOS = "data/processed/mls_players_processed.csv"
URL_REPOSITORIO = "https://github.com/rafaelandradee09-tech/mls-analysis.git"
diretorio_app = os.path.dirname(os.path.abspath(__file__))
caminho_imagem = os.path.join(
    diretorio_app, "..", "data", "raw", "usa_mls_64x64"
)
caminho_imagem2 = os.path.join(
    diretorio_app, "..", "data", "raw", "usa_mls_128x128"
)


logo = Image.open(caminho_imagem)
logo2 = Image.open(caminho_imagem2)
col1, col2, col3 = st.columns(3)
with col2:
    st.image(logo, use_container_width=40)

@st.cache_data
def carregar_dados():
    """Carrega o CSV processado. Roda uma vez só, graças ao cache do Streamlit."""
    if not os.path.exists(CAMINHO_DADOS_PROCESSADOS):
        return None
    return pd.read_csv(CAMINHO_DADOS_PROCESSADOS)


def resetar_filtros():
    """
    Remove as chaves dos filtros do session_state. Na próxima renderização,
    os widgets voltam ao valor padrão (é chamada pelo botão "Resetar
    filtros" na sidebar).
    """
    for chave in ["filtro_posicao", "filtro_clube", "filtro_minutos", "filtro_clusters"]:
        if chave in st.session_state:
            del st.session_state[chave]
    # Um novo filtro invalida o clustering calculado anteriormente.
    st.session_state.pop("resultado_clustering", None)


def montar_sidebar(df):
    """Cria os filtros da barra lateral e retorna as escolhas do usuário."""
    st.sidebar.image(logo2, use_container_width=35)
    st.sidebar.markdown(
    "<h2 style='text-align: center; color: white; font-weight: bold;'>MLS Analysis</h2>",
    unsafe_allow_html=True,
    )
    st.sidebar.caption("Análise de jogadores da MLS com base em dados do FBref.")
    st.sidebar.header("Filtros")

    posicoes = ["Todas"] + sorted(df["pos"].dropna().unique().tolist()) if "pos" in df.columns else ["Todas"]
    posicao_escolhida = st.sidebar.selectbox(
        "Posição", posicoes, key="filtro_posicao", help="Filtra os jogadores pela posição registrada no FBref."
    )

    clubes = ["Todos"] + sorted(df["squad"].dropna().unique().tolist()) if "squad" in df.columns else ["Todos"]
    clube_escolhido = st.sidebar.selectbox(
        "Clube", clubes, key="filtro_clube", help="Filtra os jogadores por clube (squad)."
    )

    minutos_minimos = st.sidebar.select_slider(
        "Minutos mínimos",
        options=[0, 100, 300, 500, 900, 1500],
        value=500,
        key="filtro_minutos",
        help="Jogadores com poucos minutos têm estatísticas por 90 minutos menos confiáveis.",
    )

    numero_clusters = st.sidebar.slider(
        "Número de clusters (K-Means)",
        min_value=2,
        max_value=8,
        value=4,
        key="filtro_clusters",
        help="K-Means agrupa jogadores parecidos em 'K' grupos. Mais grupos = grupos mais específicos.",
    )

    st.sidebar.button("↺ Resetar filtros", on_click=resetar_filtros, use_container_width=True)

    return posicao_escolhida, clube_escolhido, minutos_minimos, numero_clusters


def aplicar_filtros(df, posicao, clube, minutos_minimos):
    """Aplica os filtros escolhidos na sidebar ao DataFrame."""
    df_filtrado = filter_by_minutes(df, minutos_minimos)

    if posicao != "Todas" and "pos" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["pos"] == posicao]

    if clube != "Todos" and "squad" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["squad"] == clube]

    return df_filtrado.reset_index(drop=True)


def pagina_overview(df):
    st.header("Visão geral")

    colunas = st.columns(5)
    if "player" in df.columns:
        colunas[0].metric("Jogadores", df["player"].nunique())
    if "squad" in df.columns:
        colunas[1].metric("Clubes", df["squad"].nunique())
    if "age" in df.columns:
        colunas[2].metric("Idade média", round(df["age"].mean(), 1))
    if "gls" in df.columns:
        colunas[3].metric("Total de gols", int(df["gls"].sum()))
    if "ast" in df.columns:
        colunas[4].metric("Total de assistências", int(df["ast"].sum()))

    col1, col2 = st.columns(2)

    with col1:
        if "pos" in df.columns:
            plot_bar_plotly(
                df["pos"].value_counts(), "Jogadores por posição", "Posição", "Quantidade de jogadores"
            )

    with col2:
        if "age" in df.columns:
            plot_histogram_plotly(df["age"], "Distribuição de idade", "Idade")

    if "squad" in df.columns:
        plot_bar_plotly(
            df["squad"].value_counts(),
            "Jogadores por clube",
            "Clube",
            "Quantidade de jogadores",
            rotacionar_x=True,
        )

    if "min" in df.columns:
        plot_histogram_plotly(df["min"], "Distribuição de minutos jogados", "Minutos", bins=25)


def pagina_player_analysis(df):
    st.header("Análise de jogador")

    if "player" not in df.columns:
        st.warning("Coluna 'player' não disponível.")
        return

    nomes = sorted(df["player"].dropna().unique().tolist())
    jogador_escolhido = st.selectbox("Selecione um jogador", nomes)

    dados_jogador = get_player_statistics(df, jogador_escolhido)
    if dados_jogador is None or dados_jogador.empty:
        st.warning("Jogador não encontrado com os filtros atuais.")
        return

    show_player_info(dados_jogador.iloc[0])

    if "pos" in df.columns and "gls_per90" in df.columns:
        st.markdown("**Comparação com a média da posição**")
        posicao_jogador = dados_jogador.iloc[0]["pos"]
        media_posicao = get_average_by_position(df, "gls_per90")
        if media_posicao is not None and posicao_jogador in media_posicao.index:
            col1, col2 = st.columns(2)
            col1.metric("Gols/90 do jogador", round(dados_jogador.iloc[0]["gls_per90"], 2))
            col2.metric(f"Média de gols/90 na posição ({posicao_jogador})", round(media_posicao[posicao_jogador], 2))


def pagina_rankings(df):
    st.header("Rankings")

    estatisticas_disponiveis = {
        "Gols": "gls",
        "Assistências": "ast",
        "Gols + Assistências": "g_a",
        "Gols por 90 minutos": "gls_per90",
        "Assistências por 90 minutos": "ast_per90",
    }
    estatisticas_disponiveis = {
        rotulo: coluna for rotulo, coluna in estatisticas_disponiveis.items() if coluna in df.columns
    }

    if not estatisticas_disponiveis:
        st.warning("Nenhuma estatística de ranking está disponível neste dataset.")
        return

    col1, col2 = st.columns(2)
    stat_escolhida = col1.selectbox("Estatística", list(estatisticas_disponiveis.keys()))
    quantidade = col2.selectbox("Quantidade de jogadores", [10, 20, 30])

    coluna_stat = estatisticas_disponiveis[stat_escolhida]
    ranking = get_top_players_by_stat(df, coluna_stat, n=quantidade)
    show_ranking_table(ranking, f"Top {quantidade} - {stat_escolhida}")
    download_csv_button(ranking, f"ranking_{coluna_stat}.csv", "⬇ Baixar ranking (CSV)")


def pagina_comparacao(df):
    st.header("Comparar jogadores")
    st.caption("Compare até 3 jogadores em um gráfico radar. Cada estatística é normalizada de 0 a 100 para ficar comparável na mesma escala.")

    if "player" not in df.columns:
        st.warning("Coluna 'player' não disponível.")
        return

    nomes = sorted(df["player"].dropna().unique().tolist())
    jogadores_escolhidos = st.multiselect(
        "Selecione de 2 a 3 jogadores", nomes, max_selections=3,
        help="O radar precisa de pelo menos 2 jogadores para fazer sentido como comparação.",
    )

    features_padrao = [f for f in CLUSTER_FEATURES if f in df.columns]
    features_escolhidas = st.multiselect(
        "Estatísticas a comparar", features_padrao, default=features_padrao,
        help="Escolha pelo menos 3 estatísticas para o radar ter um formato legível.",
    )

    if len(jogadores_escolhidos) < 2:
        st.info("Selecione pelo menos 2 jogadores para comparar.")
        return

    if len(features_escolhidas) < 3:
        st.info("Selecione pelo menos 3 estatísticas para montar o radar.")
        return

    df_radar, features_usadas = prepare_radar_data(df, jogadores_escolhidos, features_escolhidas)
    if df_radar is None or df_radar.empty:
        st.warning("Não há dados suficientes para comparar os jogadores selecionados.")
        return

    plot_radar_chart(df_radar, features_usadas)

    st.markdown("**Valores originais (não normalizados)**")
    colunas_tabela = [c for c in ["player", "squad", "pos"] if c in df.columns] + features_usadas
    tabela_comparacao = df[df["player"].isin(jogadores_escolhidos)][colunas_tabela].reset_index(drop=True)
    st.dataframe(tabela_comparacao, use_container_width=True, hide_index=True)


def pagina_clubes(df):
    st.header("Clubes")
    st.caption("Resumo agregado por clube, considerando os jogadores após os filtros aplicados.")

    resumo_clubes = get_club_overview(df)
    if resumo_clubes is None or resumo_clubes.empty:
        st.warning("Não há dados suficientes para montar o resumo por clube.")
        return

    if "gols_totais" in resumo_clubes.columns:
        top_clubes = resumo_clubes.set_index("squad")["gols_totais"].head(15)
        plot_bar_plotly(top_clubes, "Gols totais por clube (top 15)", "Clube", "Gols", rotacionar_x=True)

    st.markdown("**Tabela completa por clube**")
    st.dataframe(resumo_clubes, use_container_width=True, hide_index=True)
    download_csv_button(resumo_clubes, "resumo_clubes.csv", "⬇ Baixar resumo por clube (CSV)")


def pagina_clustering(df, numero_clusters):
    st.header("Clustering (K-Means)")

    df_prontos, X, features_usadas = prepare_clustering_data(df, CLUSTER_FEATURES)

    if X.empty or len(df_prontos) < numero_clusters:
        st.warning("Não há jogadores suficientes com dados completos para executar o clustering com os filtros atuais.")
        return

    st.caption(f"Features utilizadas no agrupamento: {', '.join(features_usadas)}")

    if st.button("Executar clustering", help="Roda o K-Means com o número de clusters escolhido na barra lateral."):
        with st.spinner("Agrupando jogadores..."):
            X_normalizado, _ = scale_features(X)
            rotulos, _ = perform_kmeans(X_normalizado, n_clusters=numero_clusters)

            df_com_clusters = df_prontos.copy()
            df_com_clusters["cluster"] = rotulos

            componentes_pca = apply_pca(X_normalizado)
            df_com_clusters["PC1"] = componentes_pca["PC1"]
            df_com_clusters["PC2"] = componentes_pca["PC2"]

            resumo = summarize_clusters(df_com_clusters, features_usadas)

        # Guarda o resultado no session_state para não perder ao trocar de
        # filtro/aba antes de clicar no botão de novo.
        st.session_state["resultado_clustering"] = {
            "df": df_com_clusters,
            "resumo": resumo,
            "k": numero_clusters,
            "features": features_usadas,
        }

    resultado = st.session_state.get("resultado_clustering")
    if resultado is None:
        st.info("Clique em 'Executar clustering' para ver os resultados.")
        return

    df_com_clusters = resultado["df"]
    resumo = resultado["resumo"]
    numero_clusters_usado = resultado["k"]
    features_do_resultado = resultado["features"]

    col1, col2 = st.columns(2)
    col1.metric("Número de clusters", numero_clusters_usado)
    col2.metric("Jogadores agrupados", len(df_com_clusters))

    st.markdown("**Visualização PCA (2 dimensões)**")
    plot_cluster_scatter_plotly(df_com_clusters, numero_clusters_usado)

    show_cluster_summary(resumo)

    st.markdown("**Tabela de jogadores por cluster**")
    colunas_exibicao = [c for c in ["player", "squad", "pos", "cluster"] + features_do_resultado if c in df_com_clusters.columns]
    st.dataframe(df_com_clusters[colunas_exibicao], use_container_width=True, hide_index=True)
    download_csv_button(df_com_clusters[colunas_exibicao], "clustering_mls.csv", "⬇ Baixar tabela de clusters (CSV)")


def pagina_score_personalizado(df):
    st.header("Score personalizado")
    st.caption(
        "Monte um ranking próprio ajustando o peso de cada estatística. "
        "Cada estatística é normalizada (0-1) antes de aplicar o peso, "
        "então clubes/jogadores com poucos minutos não distorcem a escala."
    )

    colunas_disponiveis = {
        "Gols por 90": "gls_per90",
        "Assistências por 90": "ast_per90",
        "Gols + Assistências por 90": "g_a_per90",
        "Cartões amarelos por 90": "crdy_per90",
        "Cartões vermelhos por 90": "crdr_per90",
    }
    colunas_disponiveis = {r: c for r, c in colunas_disponiveis.items() if c in df.columns}

    if not colunas_disponiveis:
        st.warning("Nenhuma estatística por 90 minutos está disponível neste dataset.")
        return

    st.markdown("**Pesos de cada estatística** (negativo penaliza, positivo valoriza)")
    pesos = {}
    colunas_sliders = st.columns(len(colunas_disponiveis))
    for coluna_ui, (rotulo, coluna) in zip(colunas_sliders, colunas_disponiveis.items()):
        valor_padrao = -1.0 if "cart" in rotulo.lower() else 1.0
        pesos[coluna] = coluna_ui.slider(rotulo, min_value=-3.0, max_value=3.0, value=valor_padrao, step=0.5)

    quantidade = st.selectbox("Quantidade de jogadores no ranking", [10, 20, 30], key="qtd_score")

    if st.button("Calcular score"):
        ranking_score = compute_scouting_score(df, pesos)
        if ranking_score is None or ranking_score.empty:
            st.warning("Não foi possível calcular o score com os dados disponíveis.")
            return
        st.session_state["resultado_score"] = ranking_score

    ranking_score = st.session_state.get("resultado_score")
    if ranking_score is None:
        st.info("Ajuste os pesos e clique em 'Calcular score' para ver o ranking.")
        return

    show_ranking_table(ranking_score.head(quantidade), "Ranking por score personalizado")
    download_csv_button(ranking_score.head(quantidade), "score_personalizado.csv", "⬇ Baixar ranking (CSV)")


def pagina_similar_players(df):
    st.header("Jogadores semelhantes")

    if "player" not in df.columns:
        st.warning("Coluna 'player' não disponível.")
        return

    nomes = sorted(df["player"].dropna().unique().tolist())
    jogador_escolhido = st.selectbox("Encontre jogadores semelhantes a:", nomes)

    if st.button("Buscar jogadores semelhantes"):
        resultado = find_similar_players(df, jogador_escolhido, CLUSTER_FEATURES, n_neighbors=5)
        st.session_state["resultado_similares"] = {"jogador": jogador_escolhido, "resultado": resultado}

    dados_similares = st.session_state.get("resultado_similares")
    if dados_similares is not None:
        show_similar_players(dados_similares["resultado"], dados_similares["jogador"])
        download_csv_button(dados_similares["resultado"], "jogadores_semelhantes.csv", "⬇ Baixar tabela (CSV)")


def pagina_sobre():
    st.header("Sobre o projeto")
    st.markdown(
        """
Projeto de análise de dados esportivos com foco na **MLS (Major League
Soccer)**, feito para portfólio. Usa um único CSV exportado do
[FBref](https://fbref.com/) e cobre o fluxo completo de um projeto de
Data Science clássico: exploração, limpeza, análise estatística,
visualização, clustering (K-Means + PCA) e este dashboard interativo.

**Tecnologias:** Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit.

**Fonte dos dados:** tabela *Standard Stats* do FBref para jogadores da
MLS. Contém estatísticas básicas (jogos, minutos, gols, assistências,
cartões). Não contém xG, xAG, passes progressivos, desarmes ou
interceptações -- por isso análises mais avançadas (mapas de calor,
score de criação de jogadas, etc.) não são possíveis com este dataset
específico.

**Limitações:**
- Estatísticas não representam completamente a qualidade de um jogador.
- Jogadores de posições diferentes têm funções diferentes em campo.
- K-Means encontra agrupamentos matemáticos, não "entende" futebol.
- A quantidade de minutos jogados influencia as estatísticas (por isso
  existe o filtro de minutos mínimos na sidebar).
- Este projeto é uma ferramenta de análise exploratória, **não** um
  sistema definitivo de scouting.
        """
    )
    st.markdown(f"[📂 Repositório no GitHub]({URL_REPOSITORIO})")


def main():
    st.title("MLS Football Data Analysis")
    st.caption("Análise de jogadores da MLS com base em dados do FBref.")

    df = carregar_dados()

    if df is None:
        st.error(
            "Dados processados não encontrados. Execute 'python main.py' antes de abrir o dashboard."
        )
        return

    posicao, clube, minutos_minimos, numero_clusters = montar_sidebar(df)
    df_filtrado = aplicar_filtros(df, posicao, clube, minutos_minimos)

    st.caption(f"{len(df_filtrado)} jogador(es) após os filtros aplicados.")

    abas = st.tabs(
        [
            "Overview",
            "Análise de jogador",
            "Rankings",
            "Comparar jogadores",
            "Clubes",
            "Clustering",
            "Score personalizado",
            "Jogadores semelhantes",
            "Sobre",
        ]
    )

    with abas[0]:
        pagina_overview(df_filtrado)
    with abas[1]:
        pagina_player_analysis(df_filtrado)
    with abas[2]:
        pagina_rankings(df_filtrado)
    with abas[3]:
        pagina_comparacao(df_filtrado)
    with abas[4]:
        pagina_clubes(df_filtrado)
    with abas[5]:
        pagina_clustering(df_filtrado, numero_clusters)
    with abas[6]:
        pagina_score_personalizado(df_filtrado)
    with abas[7]:
        pagina_similar_players(df_filtrado)
    with abas[8]:
        pagina_sobre()


if __name__ == "__main__":
    main()
