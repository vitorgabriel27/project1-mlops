import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ml.models.lazy_predict_model import LazyPredictModel
from ml.models.pycaret_model import PyCaretModel

class MLToolsComparisonTab:
    def __init__(self, df):
        self.df = df
        self.lazy_results = None
        self.pycaret_results = None
        
    def render(self):
        st.header("🔬 Comparação de Ferramentas de AutoML")
        
        if len(self.df) == 0:
            st.error("⚠️ Nenhum dado carregado. Por favor, carregue os dados na aba '📊 Carregar Dados' primeiro.")
            return
        
        st.info("""
        **Objetivo:** Comparar o desempenho de duas ferramentas de AutoML (LazyPredict e PyCaret) 
        no dataset do Airbnb Rio. Ambas as ferramentas automatizam o processo de seleção e 
        comparação de modelos de machine learning.
        """)
        
        # Métricas do dataset
        self.show_dataset_info()
        
        # Configurações
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Tamanho do teste", 0.1, 0.3, 0.2, 0.05)
        with col2:
            n_models = st.slider("Número de modelos a comparar", 5, 20, 10)
        
        # Execução das ferramentas
        # tab1, tab2, tab3 = st.tabs(["🚀 Executar Ferramentas", "📊 Comparação de Resultados", "🔮 Previsões Comparativas"])
        
        st.title("🚀 Executar Ferramentas de AutoML")
        self.run_tools_comparison(test_size, n_models)
        
        # with tab2:
        #     st.header("📊 Resultados da Comparação")
            
        # with tab3:
        #     st.header("🔮 Previsões Comparativas")
        #     st.text("Em breve: Interface para inserir dados e comparar previsões entre os modelos.")

    def show_dataset_info(self):
        """Mostrar informações sobre o dataset"""
        st.subheader("📁 Informações do Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Listings", len(self.df))
        with col2:
            if 'price' in self.df.columns:
                st.metric("Preço Médio", f"R$ {self.df['price'].mean():.2f}")
        with col3:
            if 'price' in self.df.columns:
                st.metric("Preço Mínimo", f"R$ {self.df['price'].min():.2f}")
        with col4:
            if 'price' in self.df.columns:
                st.metric("Preço Máximo", f"R$ {self.df['price'].max():.2f}")
        
        # Distribuição de preços
        if 'price' in self.df.columns:
            fig = px.histogram(self.df, x='price', title='Distribuição de Preços dos Imóveis',
                              labels={'price': 'Preço (R$)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    def run_tools_comparison(self, test_size, n_models):
        """Executa a comparação entre as ferramentas de AutoML"""
        input_data = {
            'neighbourhood_cleansed': st.selectbox('Bairro', self.df['neighbourhood_cleansed'].unique()),
            'room_type': st.selectbox('Tipo do quarto', self.df['room_type'].unique()),
            'accommodates': st.number_input('Número de hóspedes', min_value=1, max_value=10, value=2),
            'bathrooms': st.number_input('Banheiros', min_value=0, max_value=5, value=1),
            'bedrooms': st.number_input('Quartos', min_value=0, max_value=5, value=1),
            'beds': st.number_input('Camas', min_value=0, max_value=5, value=1),
        }

        if st.button('Prever preço'):
            col1, col2 = st.columns(2)

            # ----------- PYCARET -----------
            with col1:
                st.markdown("### 🧠 PyCaret")
                try:
                    pred_df = PyCaretModel.predict(input_data)
                    print(pred_df)
                    predicted_value = float(pred_df['prediction_label'].iloc[0])
                    r2, rmse, mae = PyCaretModel.evaluate_predictions(pred_df)

                    st.metric("Preço Previsto", f"R$ {predicted_value:.2f}")
                    if r2 is not None:
                        st.write(f"**R²**: {r2:.3f}")
                        st.write(f"**RMSE**: {rmse:.2f}")
                        st.write(f"**MAE**: {mae:.2f}")
                    else:
                        st.write("⚠️ Métricas indisponíveis (sem valores reais).")

                    scatter, hist, line = PyCaretModel.plot_metrics(pred_df)
                    st.altair_chart(scatter, use_container_width=True)
                    st.altair_chart(hist, use_container_width=True)
                    st.altair_chart(line, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro PyCaret: {e}")

            with col2:
                st.markdown("### 🐢 LazyPredict")
                try:
                    # Previsão via modelo ONNX + pré-processador
                    lazy_pred, lazy_r2, lazy_rmse, lazy_mae = LazyPredictModel.predict_onnx(
                        input_data,
                        model_path="lazy_model.onnx",
                        preprocessor_path="preprocessor.pkl"
                    )

                    st.metric("Preço Previsto", f"R$ {lazy_pred:.2f}")
                    st.write(f"**R²**: {lazy_r2:.3f}")
                    st.write(f"**RMSE**: {lazy_rmse:.2f}")
                    st.write(f"**MAE**: {lazy_mae:.2f}")
                except Exception as e:
                    st.error(f"Erro LazyPredict (ONNX): {e}")