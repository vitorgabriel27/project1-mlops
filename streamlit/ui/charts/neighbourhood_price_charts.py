import streamlit as st
import plotly.express as px
from services.dataframes import AirbnbRioDataFrame 
from domain.plotter import Plotter

class NeighbourhoodPricePlotter(Plotter):
    """
    Plotter para análise de preços por bairro
    """
    dataframe = AirbnbRioDataFrame

    def plot(self):
        st.header("🏘️ Análise de Preços por Bairro")
        
        # Selecionar top N bairros para análise
        neighbourhood_counts = self.df['neighbourhood_cleansed'].value_counts()
        top_neighbourhoods = neighbourhood_counts.head(15).index.tolist()
        
        # Filtro interativo para selecionar bairros
        selected_neighbourhoods = st.multiselect(
            "Selecione os bairros para análise:",
            options=self.df['neighbourhood_cleansed'].unique(),
            default=top_neighbourhoods[:5]
        )
        
        if not selected_neighbourhoods:
            st.warning("Selecione pelo menos um bairro para visualizar os dados.")
            return
        
        # Filtrar dados para bairros selecionados
        filtered_df = self.df[self.df['neighbourhood_cleansed'].isin(selected_neighbourhoods)]
        
        tab1, tab2, tab3 = st.tabs(["Preço Médio por Bairro", "Distribuição Detalhada", "Comparativo de Estatísticas"])
        
        with tab1:
            self._plot_neighbourhood_avg_prices(filtered_df, selected_neighbourhoods)
        
        with tab2:
            self._plot_neighbourhood_distribution(filtered_df)
        
        with tab3:
            self._plot_neighbourhood_stats_comparison(filtered_df)

    def _plot_neighbourhood_avg_prices(self, df, neighbourhoods):
        """Plot do preço médio por bairro"""
        avg_prices = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=avg_prices.values,
            y=avg_prices.index,
            orientation='h',
            title="Preço Médio por Bairro",
            labels={"x": "Preço Médio (BRL)", "y": "Bairro"},
            color=avg_prices.values,
            color_continuous_scale="thermal"
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Preço Médio (BRL)",
            yaxis_title="Bairro",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_neighbourhood_distribution(self, df):
        """Plot da distribuição de preços por bairro"""
        fig = px.box(
            df,
            x="neighbourhood_cleansed",
            y="price",
            title="Distribuição de Preços por Bairro",
            labels={"neighbourhood_cleansed": "Bairro", "price": "Preço (BRL)"},
            color="neighbourhood_cleansed"
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Bairro",
            yaxis_title="Preço (BRL)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_neighbourhood_stats_comparison(self, df):
        """Plot comparativo de estatísticas por bairro"""
        stats_by_neighbourhood = df.groupby('neighbourhood_cleansed').agg({
            'price': ['mean', 'median', 'std', 'count'],
            'bedrooms': 'mean',
            'accommodates': 'mean'
        }).round(2)
        
        # Simplificar nomes das colunas
        stats_by_neighbourhood.columns = ['Preço Médio', 'Preço Mediano', 'Desvio Padrão', 'Total Listings', 'Média Quartos', 'Média Hóspedes']
        
        st.subheader("📈 Estatísticas Comparativas por Bairro")
        st.dataframe(stats_by_neighbourhood, use_container_width=True)
    

