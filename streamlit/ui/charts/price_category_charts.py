import streamlit as st
import plotly.express as px
import pandas as pd
from services.dataframes import AirbnbRioDataFrame 
from domain.plotter import Plotter

class PriceCategoryPlotter(Plotter):
    """
    Plotter para análise de preços por categoria
    """
    dataframe = AirbnbRioDataFrame

    def plot(self):
        st.header("🏷️ Análise de Preços por Categoria")
        
        # Verificar se a coluna price_category existe
        if 'price_category' not in self.df.columns:
            # Criar categorias de preço dinamicamente
            self.df['price_category'] = pd.cut(
                self.df['price'], 
                bins=[0, 100, 200, 300, 500, 1000, float('inf')],
                labels=['<100', '100-200', '200-300', '300-500', '500-1000', '>1000']
            )
        
        tab1, tab2 = st.tabs(["Distribuição por Categoria", "Comparativo entre Categorias"])
        
        with tab1:
            self._plot_category_distribution()
        
        with tab2:
            self._plot_category_comparison()

    def _plot_category_distribution(self):
        """Plot da distribuição por categoria de preço"""
        category_counts = self.df['price_category'].value_counts().sort_index()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Distribuição de Listings por Categoria de Preço",
            labels={"x": "Categoria de Preço (BRL)", "y": "Número de Listings"},
            color=category_counts.values,
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Categoria de Preço",
            yaxis_title="Número de Listings",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas por categoria
        st.subheader("📊 Estatísticas por Categoria")
        category_stats = self.df.groupby('price_category').agg({
            'price': ['count', 'mean', 'median', 'min', 'max'],
            'bedrooms': 'mean',
            'accommodates': 'mean'
        }).round(2)
        
        # Renomear colunas para melhor visualização
        category_stats.columns = ['Count', 'Avg Price', 'Median Price', 'Min Price', 'Max Price', 'Avg Bedrooms', 'Avg Accommodates']
        st.dataframe(category_stats, use_container_width=True)

    def _plot_category_comparison(self):
        """Plot comparativo entre categorias"""
        fig = px.box(
            self.df,
            x="price_category",
            y="price",
            title="Distribuição de Preços por Categoria",
            labels={"price_category": "Categoria de Preço", "price": "Preço (BRL)"},
            color="price_category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Categoria de Preço",
            yaxis_title="Preço (BRL)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)