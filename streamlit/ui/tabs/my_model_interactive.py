import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
from ml.models.my_model import MyModel

sys.path.append('.')

# Configuração da página
st.set_page_config(
    page_title="Airbnb Price Predictor - Model Training",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .training-progress {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class ModelTrainingApp:

    def __init__(self):
        path = Path(__file__).resolve().parent.parent.parent / "data" / "airbnb_rio_cleaned.csv" 
        airbnb_data = pd.read_csv(path) #if path.exists() else pd.DataFrame()
        self.model = None
        self.df = airbnb_data.copy()    
        self.training_history = None
        self.metrics = None
        
        # Inicializar estados da sessão
        if 'config' not in st.session_state:
            st.session_state.config = self.get_default_config()
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = True 
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'section' not in st.session_state:
            st.session_state.section = "📊 Carregar Dados"
    
    def get_default_config(self):
        """Retorna configuração padrão"""
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_epochs': 50,
            'patience': 5,
            'hidden_layers': [32, 16],
            'dropout_rate': 0.1,
            'test_size': 0.2
        }
        
        
    def load_data(self):
        """Interface para carregar dados"""
        st.header("📊 Carregamento de Dados")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mostrar prévia dos dados
            if st.session_state.data_loaded:
                with st.expander("📋 Visualizar dados carregados"):
                    st.dataframe(self.df)
                
                # Estatísticas básicas
                if 'price' in self.df.columns:
                    st.subheader("📈 Estatísticas dos Dados")
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Total de Listings", len(self.df))
                    with stats_col2:
                        st.metric("Preço Médio", f"R$ {self.df['price'].mean():.2f}")
                    with stats_col3:
                        st.metric("Bairros Únicos", self.df['neighbourhood_cleansed'].nunique() if 'neighbourhood_cleansed' in self.df.columns else "N/A")
                    with stats_col4:
                        st.metric("Tipos de Quarto", self.df['room_type'].nunique() if 'room_type' in self.df.columns else "N/A")
                    
            return st.session_state.data_loaded
                    
        with col2:
            st.info("""
            **📋 Formato Esperado:**
            - accommodates (numérico)
            - bedrooms (numérico) 
            - bathrooms (numérico)
            - neighbourhood_cleansed (texto)
            - room_type (texto)
            - price (numérico - target)
            - latitude, longitude (numéricos)
            - minimum_nights, maximum_nights (numéricos)
            - number_of_reviews (numérico)
            """)
            
        return False
    
    def model_configuration(self):
        """Interface para configuração do modelo"""
        st.header("⚙️ Configuração do Modelo")
        
        # Usar valores da session_state ou padrão
        config = st.session_state.config
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hiperparâmetros Básicos")
            learning_rate = st.slider(
                "Taxa de Aprendizado (Learning Rate)",
                min_value=0.0001,
                max_value=0.01,
                value=config['learning_rate'],
                step=0.0001,
                format="%.4f",
                key="lr_slider"
            )
            
            batch_size = st.selectbox(
                "Tamanho do Batch",
                options=[16, 32, 64, 128],
                index=[16, 32, 64, 128].index(config['batch_size']),
                key="batch_select"
            )
            
            n_epochs = st.slider(
                "Número de Épocas",
                min_value=10,
                max_value=200,
                value=config['n_epochs'],
                key="epochs_slider"
            )

            normalization = st.selectbox(
                "Tipo de Normalização",
                options=[ "zscore", "minmax", "none"],
                index=["zscore", "minmax", "none"].index(config.get('normalization', 'zscore')),
                key="norm_select"
            )
            
            optimizer = st.selectbox(
                "Otimizador",
                options=["adam", "sgd", "rmsprop"],
                index=["adam", "sgd", "rmsprop"].index(config.get('optimizer', 'adam')),
                key="opt_select"
            )
            
        with col2:
            st.subheader("Arquitetura da Rede")
            
            # Converter lista de hidden_layers para string
            hidden_layers_str = st.text_input(
                "Camadas Ocultas (separadas por vírgula)",
                value=",".join(map(str, config['hidden_layers'])),
                help="Exemplo: 32,16 cria 2 camadas ocultas com 32 e 16 neurônios",
                key="hidden_layers_input"
            )
            
            try:
                hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
                if hidden_layers:
                    st.success(f"✅ Arquitetura: Input → {' → '.join(map(str, hidden_layers))} → Output")
                else:
                    hidden_layers = [32, 16]
                    st.warning("⚠️ Usando arquitetura padrão: 32, 16")
            except:
                hidden_layers = [32, 16]
                st.warning("⚠️ Usando arquitetura padrão: 32, 16")
            
            dropout_rate = st.slider(
                "Taxa de Dropout",
                min_value=0.0,
                max_value=0.5,
                value=config['dropout_rate'],
                step=0.05,
                key="dropout_slider"
            )
            
            test_size = st.slider(
                "Tamanho do Conjunto de Teste",
                min_value=0.1,
                max_value=0.4,
                value=config['test_size'],
                step=0.05,
                key="test_size_slider"
            )

            enable_early_stopping = st.checkbox("Habilitar Early Stopping", value=config.get('enable_early_stopping', True), key="early_stopping_checkbox")

            if enable_early_stopping:
                patience = st.slider(
                    "Paciência do Early Stopping",
                    min_value=1,
                    max_value=20,
                    value=config.get('patience', 5),
                    step=1,
                    key="patience_slider"
                )
            
        # Botão para salvar configuração
        if st.button("💾 Salvar Configuração", key="save_config_btn"):
            st.session_state.config = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'patience': patience if enable_early_stopping else 5,
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'test_size': test_size,
                'normalization': normalization, 
                'optimizer': optimizer,
                'enable_early_stopping': enable_early_stopping         
            }
            st.success("✅ Configuração salva!")
            
        # Mostrar configuração atual
        with st.expander("📋 Configuração Atual Salva"):
            st.json(st.session_state.config)
        
        return st.session_state.config
    
    def train_model(self):
        """Interface para treinamento do modelo"""
        st.header("🚀 Treinamento do Modelo")
        
        if not st.session_state.data_loaded:
            st.error("❌ Por favor, carregue os dados primeiro na seção 'Carregar Dados'.")
            return False
        
        if 'config' not in st.session_state:
            st.error("❌ Por favor, configure o modelo primeiro na seção 'Configurar Modelo'.")
            return False
        
        config = st.session_state.config

        
        if st.button("🎯 Iniciar Treinamento", type="primary", use_container_width=True, key="train_btn"):
            try:
                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Criar modelo
                status_text.text("📥 Inicializando modelo...")
                self.model = MyModel(
                    learning_rate=config['learning_rate'],
                    batch_size=config['batch_size'],
                    random_state=42,
                    normalization=config.get('normalization', 'standard'),
                    optimizer_name=config.get('optimizer', 'adam'),
                    enable_early_stopping=config.get('enable_early_stopping', False),
                    patience=config.get('patience', 5)
                )
                progress_bar.progress(10)
                
                # 2. Preparar dados
                status_text.text("📊 Preparando dados...")
                train_loader, val_loader, test_loader = self.model.prepare_data(
                    self.df, 
                    test_size=config['test_size']
                )
                progress_bar.progress(30)
                
                # Verificar se os DataLoaders são válidos
                if train_loader is None:
                    st.error("❌ Falha na preparação dos dados. Verifique o formato do DataFrame.")
                    return False
                
                # 3. Construir modelo
                status_text.text("🏗️ Construindo modelo...")
                # Obter número de features (input_dim) corretamente:
                X_batch, _ = next(iter(train_loader))
                input_dim = X_batch.shape[1]
                self.model.build_model(
                    input_dim=input_dim,
                    hidden_layers=config['hidden_layers'],
                    dropout_rate=config['dropout_rate']
                )
                progress_bar.progress(50)
                
                # 4. Treinar modelo
                status_text.text("🎯 Treinando modelo...")
                
                # Simular progresso
                for i in range(50, 90, 10):
                    progress_bar.progress(i)
                    time.sleep(0.3)
                
                # Treinamento real
                self.training_history = self.model.train_model(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=config['n_epochs'],
                )
                progress_bar.progress(90)
                
                # 5. Avaliar modelo
                status_text.text("📈 Avaliando modelo...")
                self.metrics = self.model.evaluate(test_loader)
                progress_bar.progress(100)
                
                status_text.text("✅ Treinamento concluído com sucesso!")
                st.session_state.model_trained = True
                
                # Mostrar resultados
                self.show_training_results()
                return True
                
            except Exception as e:
                st.error(f"❌ Erro durante o treinamento: {str(e)}")
                st.info("""
                **💡 Possíveis soluções:**
                1. Verifique se todas as colunas necessárias estão presentes
                2. Certifique-se de que não há valores missing críticos
                3. Tente usar uma arquitetura mais simples
                4. Reduza o número de épocas
                """)
                return False
        return False
    
    def show_training_results(self):
        """Mostrar resultados do treinamento"""
        if self.training_history is None or self.metrics is None:
            return
        
        st.header("📊 Resultados do Treinamento")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"R$ {self.metrics.get('rmse', 0):.2f}")
        with col2:
            st.metric("MAE", f"R$ {self.metrics.get('mae', 0):.2f}")
        with col3:
            st.metric("R² Score", f"{self.metrics.get('r2', 0):.4f}")
        with col4:
            best_epoch = self.training_history.get('best_epoch', 0) + 1
            st.metric("Melhor Época", best_epoch)
        
        # Gráficos
        if 'train_loss' in self.training_history and 'val_loss' in self.training_history:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de histórico de treinamento
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs = range(1, len(self.training_history['train_loss']) + 1)
                
                ax.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
                ax.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
                
                if 'best_epoch' in self.training_history:
                    ax.axvline(self.training_history['best_epoch'] + 1, color='gray', linestyle='--', 
                              label=f'Melhor época')
                
                ax.set_yscale('log')
                ax.set_xlabel('Época')
                ax.set_ylabel('Loss (log scale)')
                ax.set_title('Histórico de Treinamento')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        # Comparação de preços reais vs previstos
        st.subheader("📈 Comparação: Preços Reais vs Previstos")
        try:
            # Garantir que temos os dados de treino salvos
            if hasattr(self, "train_df"):
                df_to_plot = self.train_df.copy()
            else:
                st.warning("⚠️ Dados de treino não encontrados. Usando dataset completo.")
                df_to_plot = self.df.copy()
            
            # Fazer previsões no conjunto de treino
            predictions_df = self.model.predict_dataframe(df_to_plot)
            
            fig = px.scatter(
                predictions_df,
                x='price',
                y='predicted_price',
                title='Preços Reais vs Previstos (Treinamento)',
                labels={'price': 'Preço Real (R$)', 'predicted_price': 'Preço Previsto (R$)'}
            )
            
            # Linha de referência perfeita (y=x)
            min_price = predictions_df[['price', 'predicted_price']].min().min()
            max_price = predictions_df[['price', 'predicted_price']].max().max()
            
            fig.add_trace(go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                name='Linha Perfeita',
                line=dict(dash='dash', color='red')
            ))
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Não foi possível gerar gráfico de comparação: {str(e)}")

    
    def prediction_interface(self):
        """Interface para fazer previsões"""
        st.header("🔮 Fazer Previsões")
        
        if not st.session_state.model_trained or self.model is None:
            st.warning("⚠️ Treine um modelo primeiro para fazer previsões.")
            if st.button("🚀 Ir para Treinamento"):
                st.session_state.section = "🚀 Treinar Modelo"
                st.rerun()
            return
        
        tab1, tab2 = st.tabs(["📝 Formulário Individual", "📊 Lote de Dados"])
        
        with tab1:
            self.single_prediction_interface()
        
        with tab2:
            self.batch_prediction_interface()
    
    def single_prediction_interface(self):
        """Interface para previsão individual"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Características do Imóvel")
            accommodates = st.slider("Número de Hóspedes", 1, 16, 2, key="acc")
            bedrooms = st.slider("Quartos", 0, 10, 1, key="bedr")
            bathrooms = st.slider("Banheiros", 0.0, 10.0, 1.0, 0.5, key="bath")
            beds = st.slider("Camas", 0, 10, 1, key="beds")
            
        with col2:
            st.subheader("Localização e Tipo")
            neighbourhoods = self.df['neighbourhood_cleansed'].unique() if 'neighbourhood_cleansed' in self.df.columns else ['Copacabana', 'Ipanema', 'Leblon']
            neighbourhood = st.selectbox("Bairro", neighbourhoods, key="neigh")
            
            room_types = self.df['room_type'].unique() if 'room_type' in self.df.columns else ['Entire home/apt', 'Private room']
            room_type = st.selectbox("Tipo de Quarto", room_types, key="room")
            
            minimum_nights = st.slider("Mínimo de Noites", 1, 365, 1, key="min_nights")
            number_of_reviews = st.slider("Número de Reviews", 0, 500, 10, key="reviews")
        
        # Coordenadas padrão
        default_lat = -22.9068  # Centro do Rio
        default_lon = -43.1729
        
        if 'latitude' in self.df.columns and 'neighbourhood_cleansed' in self.df.columns:
            try:
                default_lat = self.df[self.df['neighbourhood_cleansed'] == neighbourhood]['latitude'].mean()
                default_lon = self.df[self.df['neighbourhood_cleansed'] == neighbourhood]['longitude'].mean()
            except:
                pass
        
        lat = st.number_input("Latitude", value=float(default_lat), key="lat")
        lon = st.number_input("Longitude", value=float(default_lon), key="lon")
        
        if st.button("🎯 Prever Preço", type="primary", key="predict_btn"):
            try:
                # Criar dados de entrada
                input_data = {
                    'accommodates': accommodates,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'beds': beds,
                    'neighbourhood_cleansed': neighbourhood,
                    'room_type': room_type,
                    'minimum_nights': minimum_nights,
                    'maximum_nights': 30,  # Valor padrão
                    'number_of_reviews': number_of_reviews,
                    'latitude': lat,
                    'longitude': lon
                }
                
                input_df = pd.DataFrame([input_data])
                result = self.model.predict_dataframe(input_df)
                
                predicted_price = result['predicted_price'].iloc[0]
                
                # Mostrar resultado
                st.success(f"## 💰 Preço Previsto: R$ {predicted_price:.2f}")
                
                # Comparar com média do bairro
                if 'price' in self.df.columns and 'neighbourhood_cleansed' in self.df.columns:
                    try:
                        avg_price = self.df[self.df['neighbourhood_cleansed'] == neighbourhood]['price'].mean()
                        diff = predicted_price - avg_price
                        diff_percent = (diff / avg_price) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Preço Previsto", f"R$ {predicted_price:.2f}")
                        with col2:
                            st.metric("Vs Média do Bairro", 
                                    f"R$ {diff:+.2f} ({diff_percent:+.1f}%)",
                                    delta=f"{diff_percent:+.1f}%")
                    except:
                        st.metric("Preço Previsto", f"R$ {predicted_price:.2f}")
                
            except Exception as e:
                st.error(f"❌ Erro na previsão: {str(e)}")
    
    def batch_prediction_interface(self):
        """Interface para previsão em lote"""
        st.subheader("📊 Previsão em Lote")
        
        st.info("💡 Para previsão em lote, prepare um CSV com as mesmas colunas dos dados de treino (sem a coluna 'price')")
        
        uploaded_file = st.file_uploader(
            "Carregue um arquivo CSV com dados para previsão",
            type=["csv"],
            key="batch_prediction"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(batch_df)} registros carregados para previsão")
                
                if st.button("🎯 Executar Previsões em Lote", key="batch_predict_btn"):
                    with st.spinner("Processando previsões..."):
                        results_df = self.model.predict_dataframe(batch_df)
                    
                    st.success("✅ Previsões concluídas!")
                    
                    # Mostrar resultados
                    st.dataframe(results_df.head(10))
                    
                    # Estatísticas das previsões
                    if 'predicted_price' in results_df.columns:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Preço Médio Previsto", f"R$ {results_df['predicted_price'].mean():.2f}")
                        with col2:
                            st.metric("Preço Mínimo", f"R$ {results_df['predicted_price'].min():.2f}")
                        with col3:
                            st.metric("Preço Máximo", f"R$ {results_df['predicted_price'].max():.2f}")
                    
                    # Download dos resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download dos Resultados",
                        data=csv,
                        file_name=f"previsoes_airbnb_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Erro no processamento em lote: {str(e)}")
    
    def run(self):
        
        # Sidebar
        st.sidebar.title("Navegação")
        
        # Usar selectbox em vez de radio para melhor controle
        section = st.sidebar.selectbox(
            "Seções:",
            ["📊 Carregar Dados", "⚙️ Configurar Modelo", "🚀 Treinar Modelo", "🔮 Fazer Previsões"],
            index=["📊 Carregar Dados", "⚙️ Configurar Modelo", "🚀 Treinar Modelo", "🔮 Fazer Previsões"].index(st.session_state.section)
        )
        
        # Atualizar seção atual
        st.session_state.section = section
        
        # Navegação entre seções
        if section == "📊 Carregar Dados":
            self.load_data()
                
        elif section == "⚙️ Configurar Modelo":
            if st.session_state.data_loaded:
                self.model_configuration()
            else:
                st.warning("⚠️ Carregue os dados primeiro na seção 'Carregar Dados'")
                
        elif section == "🚀 Treinar Modelo":
            if st.session_state.data_loaded:
                self.train_model()
            else:
                st.warning("⚠️ Carregue os dados primeiro na seção 'Carregar Dados'")
                
        elif section == "🔮 Fazer Previsões":
            self.prediction_interface()
        
        # Footer
        st.sidebar.markdown("---")
        
        # Debug: mostrar estados atuais (opcional)
        with st.sidebar.expander("🔧 Estados da Sessão"):
            st.write(f"**Dados carregados:** {st.session_state.data_loaded}")
            st.write(f"**Modelo treinado:** {st.session_state.model_trained}")
            st.write(f"**Seção atual:** {st.session_state.section}")