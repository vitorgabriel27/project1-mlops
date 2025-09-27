import streamlit as st
from services.dataframes import AirbnbRioDataFrame
from ui.charts.price_distribution_charts import PriceDistributionPlotter
from ui.charts.price_category_charts import PriceCategoryPlotter
from ui.charts.neighbourhood_price_charts import NeighbourhoodPricePlotter

class AirbnbPriceDashboard:
    """
    Dashboard completo para análise de preços do Airbnb
    """
    
    def __init__(self):
        self.filters = {}
    
    def set_filters(self, filters: dict):
        """Define os filtros para todos os plotters"""
        self.filters = filters
    
    def render_dashboard(self):
        """Renderiza o dashboard completo"""
        st.title("🏠 Airbnb Rio de Janeiro - Análise de Preços")
        
        # Sidebar com filtros
        st.sidebar.header("Filtros")
        self._render_filters_sidebar()
        
        # Renderizar todos os plotters
        try:
            PriceDistributionPlotter(self.filters).render()
            st.markdown("---")
            PriceCategoryPlotter(self.filters).render()
            st.markdown("---")
            NeighbourhoodPricePlotter(self.filters).render()
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
    
    def _render_filters_sidebar(self):
        """Renderiza a sidebar com filtros interativos"""
        # Carregar dados para obter ranges
        df = AirbnbRioDataFrame.mount()
        
        # Filtro de preço
        price_range = st.sidebar.slider(
            "Faixa de Preço (R$):",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(0, 500)
        )
        self.filters['price_min'] = price_range[0]
        self.filters['price_max'] = price_range[1]
        
        # Filtro de bairros
        neighbourhoods = df['neighbourhood_cleansed'].unique()
        selected_neighbourhoods = st.sidebar.multiselect(
            "Bairros:",
            options=neighbourhoods,
            default=[]
        )
        if selected_neighbourhoods:
            self.filters['neighbourhood'] = selected_neighbourhoods
        
        # Filtro de tipo de quarto
        room_types = df['room_type'].unique()
        selected_room_types = st.sidebar.multiselect(
            "Tipo de Quarto:",
            options=room_types,
            default=[]
        )
        if selected_room_types:
            self.filters['room_type'] = selected_room_types
        
        # Filtro de número de quartos
        bedrooms_range = st.sidebar.slider(
            "Número de Quartos:",
            min_value=int(df['bedrooms'].min()),
            max_value=int(df['bedrooms'].max()),
            value=(int(df['bedrooms'].min()), int(df['bedrooms'].max()))
        )
        self.filters['bedrooms_min'] = bedrooms_range[0]
        self.filters['bedrooms_max'] = bedrooms_range[1]


# Função de conveniência para uso rápido
def render_price_analysis(filters: dict = None):
    """
    Função conveniente para renderizar a análise de preços
    """
    if filters is None:
        filters = {}
    
    dashboard = AirbnbPriceDashboard()
    dashboard.set_filters(filters)
    dashboard.render_dashboard()