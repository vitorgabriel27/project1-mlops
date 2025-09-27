import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from services.dataframes import AirbnbRioDataFrame 
from domain.plotter import Plotter

class PriceDistributionPlotter(Plotter):
    """
    Plotter para análise de distribuição de preços
    """
    dataframe = AirbnbRioDataFrame

    def plot(self):
        st.header("📊 Análise de Distribuição de Preços")
        
        # Criar abas para os diferentes tipos de visualização
        tab1, tab2, tab3 = st.tabs(["Distribuição Absoluta", "Distribuição Logarítmica", "Boxplot e Estatísticas"])
        
        with tab1:
            self._plot_absolute_distribution()
        
        with tab2:
            self._plot_log_distribution()
        
        with tab3:
            self._plot_boxplot_and_stats()

    def _plot_absolute_distribution(self):
        """Plot da distribuição absoluta de preços"""
        fig = px.histogram(
            self.df,
            x="price",
            nbins=60,
            title="Distribuição de Preços (Valores Absolutos)",
            labels={"price": "Preço (BRL)", "count": "Frequência"},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Adicionar linhas de média e mediana
        mean_price = self.df["price"].mean()
        median_price = self.df["price"].median()
        
        fig.add_vline(x=mean_price, line_dash="dash", line_color="red", 
                     annotation_text=f"Média: R$ {mean_price:.2f}")
        fig.add_vline(x=median_price, line_dash="dash", line_color="green", 
                     annotation_text=f"Mediana: R$ {median_price:.2f}")
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Preço (BRL)",
            yaxis_title="Número de Listings"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas rápidas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Média", f"R$ {mean_price:.2f}")
        with col2:
            st.metric("Mediana", f"R$ {median_price:.2f}")
        with col3:
            st.metric("Desvio Padrão", f"R$ {self.df['price'].std():.2f}")
        with col4:
            st.metric("Total Listings", len(self.df))

    def _plot_log_distribution(self):
        """Plot da distribuição logarítmica de preços"""
        # Adicionar coluna de log price se não existir
        df_plot = self.df.copy()
        df_plot['log_price'] = np.log1p(df_plot['price'])
        
        fig = px.histogram(
            df_plot,
            x="log_price",
            nbins=60,
            title="Distribuição de Preços (Escala Logarítmica)",
            labels={"log_price": "log(Preço + 1)", "count": "Frequência"},
            color_discrete_sequence=['#ff7f0e']
        )
        
        # Adicionar linhas de média e mediana do log
        mean_log = df_plot["log_price"].mean()
        median_log = df_plot["log_price"].median()
        
        fig.add_vline(x=mean_log, line_dash="dash", line_color="red",
                     annotation_text=f"Média: {mean_log:.2f}")
        fig.add_vline(x=median_log, line_dash="dash", line_color="green",
                     annotation_text=f"Mediana: {median_log:.2f}")
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="log(Preço + 1)",
            yaxis_title="Número de Listings"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explicação sobre a transformação logarítmica
        with st.expander("ℹ️ Sobre a escala logarítmica"):
            st.write("""
            A transformação logarítmica ajuda a visualizar distribuições assimétricas:
            - **Reduz o efeito de outliers** extremos
            - **Torna a distribuição mais simétrica**
            - **Facilita a identificação** de padrões em dados com grande variação
            - **log(Preço + 1)** é usado para evitar problemas com preços zero
            """)

    def _plot_boxplot_and_stats(self):
        """Plot do boxplot e estatísticas detalhadas"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Boxplot interativo
            fig = px.box(
                self.df,
                y="price",
                title="Boxplot de Preços",
                labels={"price": "Preço (BRL)"},
                color_discrete_sequence=['#2ca02c']
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis_title="Preço (BRL)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estatísticas detalhadas
            st.subheader("📈 Estatísticas Detalhadas")
            
            Q1 = self.df['price'].quantile(0.25)
            Q3 = self.df['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            stats_data = {
                "Mínimo": f"R$ {self.df['price'].min():.2f}",
                "25º Percentil (Q1)": f"R$ {Q1:.2f}",
                "Mediana (Q2)": f"R$ {self.df['price'].median():.2f}",
                "75º Percentil (Q3)": f"R$ {Q3:.2f}",
                "Máximo": f"R$ {self.df['price'].max():.2f}",
                "IQR": f"R$ {IQR:.2f}",
                "Limite Inferior (Outliers)": f"R$ {lower_bound:.2f}",
                "Limite Superior (Outliers)": f"R$ {upper_bound:.2f}"
            }
            
            for stat, value in stats_data.items():
                st.metric(stat, value)
            
            # Calcular número de outliers
            outliers_count = len(self.df[self.df['price'] > upper_bound])
            outlier_percentage = (outliers_count / len(self.df)) * 100
            
            st.info(f"**Outliers detectados:** {outliers_count} listings ({outlier_percentage:.1f}%)")