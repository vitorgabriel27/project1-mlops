# 🏠 project1-mlops

Projeto de MLOps com análise de preços de imóveis no Airbnb do Rio de Janeiro utilizando Streamlit, machine learning e bibliotecas modernas de ciência de dados.


## 📦 Requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker e Docker Compose (opcional, para rodar em container)

## 🚀 Como rodar o projeto

Você pode rodar o projeto de duas formas:

---

### ✅ 1. Rodar localmente com Poetry

> Recomendado para desenvolvimento local.

#### 🔧 Instale as dependências

```bash
poetry install

#ative o ambiente virtual
poetry shell


#rode a aplicação
streamlit run streamlit/app.py
```
### ✅ 2. Rodar com Docker + Docker Compose

> Útil para evitar instalação de dependências localmente.


```bash
docker-compose build

docker-compose up
```

### 📊 Funcionalidades

- Dashboard interativo de análise de preços

- Filtros por bairro, tipo de imóvel, número de quartos etc.

- Modelos de machine learning (regressão, árvores, ensemble)

- Visualizações com seaborn, matplotlib, plotly e shap

- Pipeline de experimentos com PyCaret e LazyPredict
