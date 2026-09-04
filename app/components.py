"""
components.py
==============

Pequenos componentes de interface reutilizados em várias páginas do
dashboard (app/dashboard.py). Cada função apenas desenha algo na tela
usando Streamlit -- nenhuma lógica de análise fica aqui, isso já foi
feito em src/analysis.py, src/clustering.py e src/similarity.py.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COR_PRINCIPAL = "#1f6feb"
PALETA_CLUSTERS = px.colors.qualitative.Set2


def show_metric_card(label, value):
    """Mostra um único cartão de métrica (número + rótulo)."""
    st.metric(label, value)


def show_player_info(dados_jogador):
    """
    Mostra as informações de um jogador em um cartão simples.

    "dados_jogador" é uma linha (Series) do DataFrame de jogadores.
    Só exibe os campos que realmente existem.
    """
    nome = dados_jogador.get("player", "Jogador")
    st.subheader(nome)

    colunas = st.columns(4)
    campos_para_mostrar = [
        ("squad", "Clube"),
        ("pos", "Posição"),
        ("age", "Idade"),
        ("min", "Minutos"),
    ]

    for coluna_ui, (campo, rotulo) in zip(colunas, campos_para_mostrar):
        if campo in dados_jogador.index:
            coluna_ui.metric(rotulo, dados_jogador[campo])

    st.markdown("**Estatísticas principais**")
    colunas_stats = st.columns(4)
    campos_stats = [
        ("gls", "Gols"),
        ("ast", "Assistências"),
        ("gls_per90", "Gols / 90"),
        ("ast_per90", "Assist. / 90"),
    ]
    for coluna_ui, (campo, rotulo) in zip(colunas_stats, campos_stats):
        if campo in dados_jogador.index:
            coluna_ui.metric(rotulo, dados_jogador[campo])


def show_ranking_table(df_ranking, titulo):
    """Mostra uma tabela de ranking com um título acima."""
    st.markdown(f"**{titulo}**")
    if df_ranking is None or df_ranking.empty:
        st.info("Não há dados suficientes para gerar este ranking.")
    else:
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)


def show_cluster_summary(df_resumo):
    """Mostra a tabela-resumo com as características de cada cluster."""
    st.markdown("**Resumo dos clusters**")
    st.dataframe(df_resumo, use_container_width=True, hide_index=True)


def show_similar_players(df_similares, nome_jogador):
    """Mostra a tabela de jogadores semelhantes a um jogador escolhido."""
    st.markdown(f"**Jogadores estatisticamente semelhantes a {nome_jogador}**")
    if df_similares is None or df_similares.empty:
        st.info("Não foi possível encontrar jogadores semelhantes com os dados disponíveis.")
    else:
        st.caption("Quanto menor a distância, mais parecidas são as estatísticas dos jogadores.")
        st.dataframe(df_similares, use_container_width=True, hide_index=True)


def download_csv_button(df, nome_arquivo, rotulo="Baixar CSV"):
    """
    Mostra um botão para baixar "df" como CSV.

    Não faz nada (não quebra) se o DataFrame for None ou vazio -- só
    exibe o botão quando há algo de fato para exportar.
    """
    if df is None or df.empty:
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=rotulo,
        data=csv_bytes,
        file_name=nome_arquivo,
        mime="text/csv",
    )


def plot_bar_plotly(serie, titulo, rotulo_x, rotulo_y, cor=COR_PRINCIPAL, rotacionar_x=False):
    """Desenha um gráfico de barras interativo (Plotly) a partir de um pandas.Series."""
    fig = px.bar(
        x=serie.index.astype(str),
        y=serie.values,
        labels={"x": rotulo_x, "y": rotulo_y},
        title=titulo,
        color_discrete_sequence=[cor],
    )
    if rotacionar_x:
        fig.update_layout(xaxis_tickangle=-90)
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig, use_container_width=True)


def plot_histogram_plotly(series, titulo, rotulo_x, cor=COR_PRINCIPAL, bins=20):
    """Desenha um histograma interativo (Plotly) a partir de uma coluna numérica."""
    fig = px.histogram(
        series.dropna(),
        nbins=bins,
        title=titulo,
        labels={"value": rotulo_x},
        color_discrete_sequence=[cor],
    )
    fig.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_scatter_plotly(df_com_clusters, numero_clusters):
    """
    Desenha o gráfico de dispersão PCA (PC1 x PC2) colorido por cluster,
    de forma interativa: passar o mouse mostra o nome do jogador.
    """
    colunas_hover = [c for c in ["player", "squad", "pos"] if c in df_com_clusters.columns]
    fig = px.scatter(
        df_com_clusters,
        x="PC1",
        y="PC2",
        color=df_com_clusters["cluster"].astype(str),
        hover_data=colunas_hover,
        color_discrete_sequence=PALETA_CLUSTERS,
        title="Jogadores agrupados por semelhança estatística (PCA)",
        labels={"color": "Cluster"},
    )
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_radar_chart(df_radar, features_usadas):
    """
    Desenha um gráfico radar comparando os jogadores em "df_radar"
    (uma linha por jogador, colunas = features já normalizadas 0-100).
    """
    fig = go.Figure()

    for _, linha in df_radar.iterrows():
        valores = [linha[f] for f in features_usadas]
        fig.add_trace(
            go.Scatterpolar(
                r=valores + [valores[0]],
                theta=features_usadas + [features_usadas[0]],
                fill="toself",
                name=linha["player"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(t=30, b=10, l=40, r=40),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)
