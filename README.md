# NBA Rookie 5-Year Classification

Aplicação de IA que estima se um calouro da NBA permanecerá **5 anos ou mais** na liga, a partir das
estatísticas por jogo da sua temporada de estreia.

- **Disciplina:** Tópicos Avançados em Inteligência Artificial
- **Link da aplicação publicada:** https://nba-rookie-5yr-79gg9bbvpax7xidzwrgjar.streamlit.app
- **Repositório:** `COLE_AQUI_O_LINK_DO_GITHUB`
- **Integrantes:** Mateus Muler Bellini e Matheus Pasiani da Silva

## Estrutura
| Arquivo | Função |
|---|---|
| `common.py` | Pré-processamento e validação compartilhados (treino e app) |
| `train.py` | EDA, limpeza, split, comparação de modelos, avaliação e salvamento |
| `app.py` | Aplicação Streamlit (somente inferência) |
| `models/model.joblib` | Pipeline treinado (imputer + scaler + modelo), limiar e metadados |
| `data/nba_rookies.csv` | Dataset (ver "Dataset" abaixo) |
| `reports/` | Métricas, logs e figuras geradas pelo treino |
| `requirements.txt` | Dependências |

## Dataset
Kaggle: **5-Year Career Longevity for NBA Rookies** (versão utilizada: https://www.kaggle.com/datasets/mexwell/5-year-nba-rookie-classification?resource=download).
Baixe o CSV e salve como `data/nba_rookies.csv`. ~1.340 jogadores, estatísticas por jogo da temporada de
calouro e o alvo `TARGET_5Yrs` (1 = ficou ≥ 5 anos).

## Como executar
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --data data/nba_rookies.csv          # gera models/model.joblib e reports/
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Suba a pasta para um repositório público no GitHub (inclua `models/model.joblib`).
2. Em https://share.streamlit.io → *New app* → escolha o repositório, branch e `app.py`.
3. Teste o link em outro navegador e cole-o neste README e no relatório.

## Uso de ferramentas de IA
Descreva aqui como o grupo usou IA (ex.: geração inicial do código, que foi revisado, testado e adaptado) e o que mudou.

## Contribuições
| Integrante | Tarefas |
|---|---|
| Nome 1 | ... |
| Nome 2 | ... |
| Nome 3 | ... |
