import streamlit as st
from ui.components.airbnb_price_dashboard import render_price_analysis


title = "Projeto 1 - MLOps - Inside AirBnb (Rio de Janeiro)"

st.set_page_config(
    page_title=title,
    layout="wide",
)

st.title("{} 🌤️".format(title))

tabs = st.tabs(
    [
        "📊 EDA Dashboard",
        "📈 Immplementação do Modelo",
        "ℹ️ Meu Modelo",
        "📊 Comparativo entre Modelos",
    ]
)

with tabs[0]:
    render_price_analysis()