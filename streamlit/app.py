import streamlit as st
from ui.components.airbnb_price_dashboard import render_price_analysis
from ui.tabs.my_model_interactive import ModelTrainingApp


title = "Projeto 1 - MLOps - Inside AirBnb (Rio de Janeiro)"

st.set_page_config(
    page_title=title,
    layout="wide",
    page_icon="🏠",
)

st.title("{}".format(title))

tabs = st.tabs(
    [
        "📊 EDA Dashboard",
        "📈 Implementação & Treinamento do Modelo",
        "📊 Comparativo entre Modelos",
    ]
)

with tabs[0]:
    render_price_analysis()
with tabs[1]:
    model_tab = ModelTrainingApp()
    model_tab.run()