# MLS Football Data Analysis

Projeto de análise de dados esportivos com foco na **MLS (Major League
Soccer)**, feito para portfólio. Usa um único CSV exportado do
[FBref](https://fbref.com/) e cobre o fluxo completo de um projeto de Data
Science "clássico": exploração, limpeza, análise estatística, visualização,
clustering (K-Means + PCA) e um dashboard interativo em Streamlit.

O código foi escrito para ser **simples e didático**: funções pequenas, cada
uma com uma responsabilidade clara, comentadas, sem frameworks ou
arquiteturas desnecessárias.

---

## 1. Objetivo do projeto

Construir uma aplicação de análise de jogadores da MLS que permita:

1. Explorar os dados dos jogadores.
2. Limpar e preparar os dados.
3. Criar métricas estatísticas (como "por 90 minutos").
4. Fazer análise exploratória.
5. Criar rankings de jogadores.
6. Criar visualizações.
7. Comparar jogadores.
8. Agrupar jogadores semelhantes com K-Means.
9. Reduzir dimensões com PCA (para visualização).
10. Encontrar jogadores semelhantes com Nearest Neighbors.
11. Explorar tudo isso em um dashboard interativo (Streamlit).

## 2. Problema que o projeto tenta resolver

Analisar manualmente centenas de jogadores e dezenas de estatísticas é
lento e pouco visual. Este projeto organiza esses dados, cria métricas
comparáveis (por 90 minutos), agrupa jogadores parecidos automaticamente e
oferece uma interface simples para explorar tudo isso -- sem exigir que
quem for usar saiba programar.

## 3. Fonte dos dados

Os dados vêm de um único arquivo CSV exportado do FBref:
`data/raw/mls_players.csv`.

**Importante sobre este dataset específico:** o CSV utilizado é a tabela
*Standard Stats* do FBref para jogadores da MLS. Ele contém estatísticas
**básicas**: jogos, minutos, gols, assistências, cartões (e versões "por 90
minutos" de gols/assistências). Ele **não contém** estatísticas avançadas
como xG, xAG, chutes, passes progressivos, conduções progressivas,
desarmes ou interceptações. O projeto foi construído para se adaptar a
isso: nenhuma coluna é inventada, e sempre que uma análise pede uma
estatística ausente, o código avisa claramente em vez de quebrar ou
simular dados.

Se, no futuro, você exportar um CSV mais completo do FBref (com mais
tabelas de estatísticas), o projeto vai automaticamente aproveitar as
colunas novas -- basta ajustar os nomes em `src/config.py`.

## 4. Estrutura das pastas

```
football-data-analysis/
├── data/
│   ├── raw/mls_players.csv                 # CSV original (nunca é alterado)
│   └── processed/mls_players_processed.csv # gerado por main.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_player_analysis.ipynb
│   └── 03_clustering.ipynb
├── src/
│   ├── config.py          # mapeamento de colunas e grupos de features
│   ├── data_loader.py     # carregar/salvar CSV
│   ├── data_cleaning.py   # limpeza e métricas por 90 minutos
│   ├── analysis.py        # rankings e estatísticas
│   ├── clustering.py      # K-Means, PCA, escolha de K
│   └── similarity.py      # Nearest Neighbors
├── app/
│   ├── dashboard.py       # dashboard Streamlit (5 páginas)
│   └── components.py      # componentes de UI reutilizáveis
├── tests/
│   ├── test_data_cleaning.py
│   └── test_clustering.py
├── requirements.txt
├── README.md
└── main.py                # pipeline: carregar -> limpar -> salvar
```

## 5. Tecnologias utilizadas

- **Python 3**
- **Pandas / NumPy** -- manipulação de dados
- **Matplotlib / Seaborn** -- visualizações
- **scikit-learn** -- StandardScaler, KMeans, PCA, NearestNeighbors, silhouette_score
- **Streamlit** -- dashboard interativo
- **pytest** -- testes automatizados

Nenhum banco de dados, API paga, Docker, React, FastAPI, deep learning ou
redes neurais foram utilizados, propositalmente -- o objetivo é manter o
projeto simples e fácil de rodar localmente.

## 6. Como instalar Python

Baixe o Python 3.10+ em [python.org/downloads](https://www.python.org/downloads/)
e instale normalmente. Para conferir se já está instalado, rode no
terminal:

```
python --version
```

## 7. Como criar o ambiente virtual

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```
python3 -m venv .venv
source .venv/bin/activate
```

## 8. Como instalar as dependências

Com o ambiente virtual ativado:

```
pip install -r requirements.txt
```

## 9. Onde colocar o CSV

O arquivo original do FBref já deve estar em:

```
data/raw/mls_players.csv
```

Esse arquivo **nunca** é modificado pelo código -- todas as transformações
geram um novo arquivo em `data/processed/`.

## 10. Como executar o projeto (pipeline principal)

```
python main.py
```

Isso vai:
1. Carregar o CSV original.
2. Mostrar uma análise inicial da estrutura do arquivo (linhas, colunas,
   valores ausentes).
3. Limpar os dados e criar as métricas por 90 minutos.
4. Salvar o resultado em `data/processed/mls_players_processed.csv`.

## 11. Como executar o dashboard

Depois de rodar `python main.py` pelo menos uma vez:

```
streamlit run app/dashboard.py
```

O navegador vai abrir automaticamente com o dashboard.

## 12. Como executar os testes

```
pytest
```

Os testes usam pequenos DataFrames fictícios (nunca os dados reais) para
verificar se as funções de limpeza e clustering funcionam como esperado.

## 13. Como funciona a análise exploratória

O notebook `01_data_exploration.ipynb` carrega o CSV original (sem
limpeza) e mostra sua estrutura real: quantidade de linhas/colunas, nomes
das colunas, tipos de dados, valores ausentes e estatísticas descritivas
(`df.head()`, `df.shape`, `df.columns`, `df.info()`, `df.describe()`,
`df.isnull().sum()`). Só depois disso os dados são limpos com
`prepare_players_data`.

## 14. Como funcionam as métricas por 90 minutos

Estatísticas de contagem (como gols) não são diretamente comparáveis entre
um jogador que jogou 200 minutos e outro que jogou 2000. A métrica "por 90
minutos" resolve isso:

```
metrica_por_90 = metrica_bruta / minutos_jogados * 90
```

A função `build_per_90_metrics` (em `src/data_cleaning.py`) faz isso de
forma genérica: para cada par (coluna bruta, nome da métrica nova), ela só
cria a métrica se a coluna bruta **e** a coluna de minutos existirem, e
evita divisão por zero (jogadores com 0 minutos recebem `NaN`, nunca
infinito). O CSV usado já traz gols/assistências por 90 prontos do FBref;
o projeto também calcula por 90 minutos os cartões e pênaltis, que não
vêm prontos.

## 15. O que é StandardScaler

É uma técnica que transforma cada coluna numérica para ter média 0 e
desvio padrão 1. Isso é necessário antes do K-Means e do Nearest
Neighbors porque esses algoritmos usam **distância** entre pontos: sem
normalizar, uma feature em escala maior (ex: minutos, que vai a milhares)
dominaria o cálculo, e features em escala menor (ex: gols por 90, entre 0
e ~1) praticamente não influenciariam o resultado.

## 16. O que é K-Means

É um algoritmo de agrupamento (clustering) que separa os dados em `K`
grupos, tentando minimizar a distância entre os pontos de um mesmo grupo e
o "centro" desse grupo. Ele não sabe nada sobre futebol -- só enxerga
números. Por isso, depois de rodar o K-Means, o projeto sempre analisa as
médias de cada grupo (função `summarize_clusters`) para permitir uma
interpretação humana do que cada cluster representa.

## 17. Como escolher o número de clusters

O projeto oferece duas ferramentas (função `find_optimal_clusters`):

- **Elbow Method (inércia):** mede o quão "compactos" estão os clusters.
  Procura-se o ponto onde a curva para de cair rapidamente (o "cotovelo").
- **Silhouette Score:** mede o quão bem separados estão os clusters
  (valores mais próximos de 1 são melhores).

Essas métricas ajudam a escolher `K`, mas a decisão final também deve
considerar se os grupos resultantes fazem sentido ao serem interpretados.
No dashboard, o número de clusters pode ser ajustado livremente pelo
usuário.

## 18. O que é PCA

PCA (Principal Component Analysis) é uma técnica para reduzir a
quantidade de dimensões (colunas) dos dados, preservando o máximo possível
da variação original. Aqui ele é usado **apenas para visualização**: como
não conseguimos "ver" um espaço com 5 ou mais dimensões, o PCA resume tudo
em 2 componentes (PC1 e PC2), permitindo plotar os jogadores em um gráfico
de dispersão colorido por cluster.

## 19. Como funciona a similaridade entre jogadores

A função `find_similar_players` (em `src/similarity.py`) usa
**Nearest Neighbors**: normaliza as features escolhidas, calcula a
distância entre o jogador selecionado e todos os outros, e retorna os
mais próximos. "Semelhante" aqui significa apenas "próximo no espaço das
estatísticas usadas" -- o projeto nunca afirma que um jogador é "melhor"
ou "pior" que outro, apenas que os números se parecem.

## 20. Limitações do projeto

- Estatísticas não representam completamente a qualidade de um jogador.
- Jogadores de posições diferentes têm funções diferentes em campo, o que
  torna certas comparações diretas menos justas.
- K-Means encontra agrupamentos matemáticos -- ele não "entende" futebol.
- PCA é usado principalmente para visualização, não para a decisão dos
  clusters em si.
- A similaridade entre jogadores depende inteiramente das features
  escolhidas.
- A quantidade de minutos jogados influencia diretamente as estatísticas
  (por isso existe o filtro de minutos mínimos).
- Dados do FBref podem conter valores ausentes.
- **Este dataset específico** só tem estatísticas básicas (gols,
  assistências, cartões, minutos) -- não há xG, xAG, passes, progressão ou
  estatísticas defensivas, o que limita a riqueza do clustering e das
  comparações.
- Este projeto é uma ferramenta de análise exploratória, **não** um
  sistema definitivo de scouting.

## 21. Possíveis melhorias futuras

(apenas documentadas, não implementadas nesta versão)

- Coleta automatizada de dados.
- Atualização automática por temporada.
- Análise no nível de clubes (não só de jogadores).
- Comparação entre diferentes ligas.
- Sistema de scouting mais completo.
- Score personalizado por posição.
- Tela dedicada de comparação lado a lado entre jogadores.
- Mapas de calor (requer dados de posicionamento, que este CSV não tem).
- Análise temporal (evolução ao longo de temporadas).
- Previsão de desempenho.
- Deploy do dashboard Streamlit (ex: Streamlit Community Cloud).

---

## Fluxo completo do projeto

```
CSV original (data/raw/mls_players.csv)
        ↓
carregamento (src/data_loader.py -> load_players_data)
        ↓
inspeção inicial (main.py / notebook 01)
        ↓
limpeza (src/data_cleaning.py -> prepare_players_data)
        ↓
métricas por 90 minutos (build_per_90_metrics, dentro da limpeza)
        ↓
CSV processado (data/processed/mls_players_processed.csv)
        ↓
análise (src/analysis.py) e visualizações (notebook 02)
        ↓
clustering: StandardScaler -> K-Means -> PCA (src/clustering.py, notebook 03)
        ↓
similaridade entre jogadores (src/similarity.py)
        ↓
dashboard interativo (app/dashboard.py + app/components.py)
```
