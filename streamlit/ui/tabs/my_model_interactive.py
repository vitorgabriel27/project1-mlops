import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import streamlit as st
import json
from pathlib import Path
import pandas as pd
from ml.models.my_model import MyModel
from ui.tabs.ml_tools_comparison import MLToolsComparisonTab

class ModelTrainingApp:
    def __init__(self):
        self.df = self.load_dataset()
        self.init_session_state()

    # ------------------------------
    # Inicialização de dados e session_state
    # ------------------------------
    def load_dataset(self):
        path = Path(__file__).resolve().parent.parent.parent / "data" / "airbnb_rio_cleaned.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def init_session_state(self):
        # Configurações padrão
        if 'config' not in st.session_state:
            st.session_state.config = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'n_epochs': 50,
                'patience': 5,
                'hidden_layers': [32, 16],
                'dropout_rate': 0.1,
                'test_size': 0.2,
                'normalization': 'zscore',
                'optimizer': 'adam',
                'enable_early_stopping': True
            }

        st.session_state.setdefault('data_loaded', bool(len(self.df) > 0))
        st.session_state.setdefault('model_trained', False)
        st.session_state.setdefault('section', "📊 Carregar Dados")
        st.session_state.setdefault('current_model', None)
        st.session_state.setdefault('current_metrics', None)
        st.session_state.setdefault('current_training_history', None)
        st.session_state.setdefault('last_model', None)
        st.session_state.setdefault('last_metrics', None)
        st.session_state.setdefault('last_training_history', None)

    # ------------------------------
    # Sidebar e navegação
    # ------------------------------
    def render_sidebar(self):
        st.sidebar.title("Navegação")
        section = st.sidebar.selectbox(
            "Seções:",
            ["📊 Carregar Dados", "⚙️ Configurar Modelo", "🚀 Treinar Modelo", "🔮 Fazer Previsões", "🔬 Comparativo entre Modelos"],
            index=["📊 Carregar Dados", "⚙️ Configurar Modelo", "🚀 Treinar Modelo", "🔮 Fazer Previsões", "🔬 Comparativo entre Modelos"].index(st.session_state.section)
        )
        st.session_state.section = section

        with st.sidebar.expander("🔧 Estados da Sessão"):
            st.json({k: str(v) for k, v in st.session_state.items()})

    def ml_tools_comparison_tab(self):
        self.ml_tools_comparison = MLToolsComparisonTab(self.df)
        self.ml_tools_comparison.render()

    # ------------------------------
    # Renderização de seções
    # ------------------------------
    def run(self):
        self.render_sidebar()
        section = st.session_state.section

        if section == "📊 Carregar Dados":
            self.load_data_section()
        elif section == "⚙️ Configurar Modelo":
            self.model_config_section()
        elif section == "🚀 Treinar Modelo":
            self.train_model_section()
        elif section == "🔮 Fazer Previsões":
            self.prediction_section()
        elif section == "🔬 Comparativo entre Modelos":
            self.ml_tools_comparison_tab()
            

    # ------------------------------
    # Seção de Carregamento de Dados
    # ------------------------------
    def load_data_section(self):
        st.header("📊 Carregamento de Dados")
        if st.session_state.data_loaded:
            st.dataframe(self.df.head())
            st.metric("Total de Listings", len(self.df))
            if 'price' in self.df.columns:
                st.metric("Preço Médio", f"R$ {self.df['price'].mean():.2f}")
        else:
            st.warning("Nenhum dado carregado.")

    # ------------------------------
    # Seção de Configuração de Modelo
    # ------------------------------
    def model_config_section(self):
        st.header("⚙️ Configuração do Modelo")
        config = st.session_state.config

        col1, col2 = st.columns(2)

        with col1:
            st.slider(
                "Taxa de Aprendizado", 
                0.0001, 
                0.01, 
                step=0.0001,
                value=config['learning_rate'],
                format="%f",  # Mostra todas as casas decimais disponíveis
                key="learning_rate"
            )
            st.selectbox("Tamanho do Batch", [16,32,64,128], key="batch_size", index=[16,32,64,128].index(config['batch_size']))
            st.slider("Número de Épocas", 10, 200, key="n_epochs", value=config['n_epochs'])
            st.selectbox("Normalização", ["zscore","minmax","none"], key="normalization", index=["zscore","minmax","none"].index(config.get('normalization','zscore')))
            st.selectbox("Otimizador", ["adam","sgd","rmsprop"], key="optimizer", index=["adam","sgd","rmsprop"].index(config.get('optimizer','adam')))
        
        with col2:
            hidden_layers_str = st.text_input("Camadas Ocultas", value=",".join(map(str, config['hidden_layers'])), key="hidden_layers_input")
            try:
                st.session_state.config['hidden_layers'] = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
            except:
                st.warning("Formato inválido. Usando padrão [32,16].")
                st.session_state.config['hidden_layers'] = [32,16]

            st.slider("Dropout", 0.0, 0.5, step=0.05, key="dropout_rate", value=config['dropout_rate'])
            st.slider("Tamanho do Teste", 0.1, 0.4, step=0.05, key="test_size", value=config['test_size'])
            enable_early = st.checkbox("Habilitar Early Stopping", value=config.get('enable_early_stopping', True), key="enable_early_stopping")
            if enable_early:
                st.slider("Paciência", 1, 20, key="patience", value=config.get('patience',5))

        # Atualiza session_state config automaticamente
        for k in ['learning_rate','batch_size','n_epochs','dropout_rate','test_size','normalization','optimizer','enable_early_stopping','patience']:
            if k in st.session_state:
                st.session_state.config[k] = st.session_state[k]

        st.json(st.session_state.config)

    # ------------------------------
    # Seção de Treinamento
    # ------------------------------
    def train_model_section(self):
        st.header("🚀 Treinamento do Modelo")
        if not st.session_state.data_loaded:
            st.error("Carregue os dados primeiro!")
            return
        config = st.session_state.config
        if st.button("🎯 Treinar Modelo"):
            self.train_model(config)

        # Mostrar resultados se já treinado
        if st.session_state.model_trained:
            self.show_training_results()

    # ------------------------------
    # Função de treino real
    # ------------------------------
    def train_model(self, config):
        try:
            # Elementos de UI
            training_status = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Etapa 1: Preparação (0% - 30%)
            training_status.subheader("🎯 Iniciando Treinamento")
            status_text.text("Preparando dados e modelo...")
            progress_bar.progress(30)
            
            self.model = MyModel(
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                normalization=config.get('normalization','standard'),
                optimizer_name=config.get('optimizer','adam'),
                enable_early_stopping=config.get('enable_early_stopping',False),
                patience=config.get('patience',5)
            )
            
            train_loader, val_loader, test_loader = self.model.prepare_data(self.df, test_size=config['test_size'])
            input_dim = next(iter(train_loader))[0].shape[1]
            self.model.build_model(input_dim=input_dim, hidden_layers=config['hidden_layers'], dropout_rate=config['dropout_rate'])

            # Etapa 2: Treinamento (30% - 80%)
            status_text.text("Treinando modelo... (Isto pode levar alguns minutos)")
            progress_bar.progress(60)
            
            # Mostrar estimativa de tempo
            with st.expander("📊 Detalhes do Treinamento"):
                st.write(f"**Configuração:**")
                st.write(f"- Épocas: {config['n_epochs']}")
                st.write(f"- Batch Size: {config['batch_size']}")
                st.write(f"- Paciência Early Stopping: {config.get('patience', 5)}")
            
            # Executar treinamento (sem progresso intermediário)
            history = self.model.train_model(train_loader, val_loader, n_epochs=config['n_epochs'])
            
            # Etapa 3: Avaliação (80% - 95%)
            status_text.text("Avaliando modelo...")
            progress_bar.progress(80)
            
            metrics = self.model.evaluate(test_loader)
            
            # Etapa 4: Finalização (95% - 100%)
            progress_bar.progress(100)
            status_text.text("✅ Treinamento concluído!")
            
            import time
            time.sleep(1)  # Pequena pausa para mostrar 100%
            
            # Atualizar session state
            st.session_state.model_trained = True
            st.session_state.current_model = self.model
            st.session_state.current_training_history = history
            st.session_state.current_metrics = metrics
            
            # Limpar elementos de UI
            progress_bar.empty()
            status_text.empty()
            training_status.empty()
            
            st.success("🎉 Modelo treinado com sucesso!")
            self.show_training_results()
            
        except Exception as e:
            st.error(f"❌ Erro no treinamento: {str(e)}")
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()


    # ------------------------------
    # Mostrar resultados de treino
    # ------------------------------
    def show_training_results(self):
        metrics = st.session_state.current_metrics
        history = st.session_state.current_training_history
        if metrics and history:
            st.subheader("📊 Resultados do Modelo")
            
            # Métricas principais em colunas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
            with col3:
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
            
            # Informações adicionais
            if 'best_epoch' in metrics and metrics['best_epoch'] is not None:
                st.info(f"🏆 Melhor época: {metrics['best_epoch'] + 1}")
            
            # Plotar histórico de treinamento
            if 'train_loss' in history and 'val_loss' in history:
                st.subheader("📈 Histórico de Treinamento")
                
                # Criar DataFrame para o histórico
                import pandas as pd
                loss_df = pd.DataFrame({
                    'Época': range(1, len(history['train_loss']) + 1),
                    'Loss Treino': history['train_loss'],
                    'Loss Validação': history['val_loss']
                })
                
                # Plotar usando Altair ou matplotlib
                try:
                    import altair as alt
                    
                    # Preparar dados para Altair (formato longo)
                    loss_df_long = loss_df.melt('Época', 
                                                var_name='Tipo', 
                                                value_name='Loss')
                    
                    # Definir escala de cores manualmente
                    color_scale = alt.Scale(
                        domain=['Loss Treino', 'Loss Validação'],
                        range=['blue', 'red']
                    )
                    
                    chart = alt.Chart(loss_df_long).mark_line().encode(
                        x=alt.X('Época:Q', title='Época'),
                        y=alt.Y('Loss:Q', title='Loss', scale=alt.Scale(zero=False)),
                        color=alt.Color('Tipo:N', 
                                    scale=color_scale,
                                    legend=alt.Legend(title="Dataset")),
                        strokeDash=alt.condition(
                            alt.datum.Tipo == 'Loss Validação',
                            alt.value([0]),
                            alt.value([0]) 
                        )
                    ).properties(
                        width=700,
                        height=400,
                        title='Evolução da Loss durante o Treinamento'
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                except ImportError:
                    # Fallback para matplotlib com cores específicas
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(loss_df['Época'], loss_df['Loss Treino'], 
                        label='Loss Treino', linewidth=2, color='blue')
                    ax.plot(loss_df['Época'], loss_df['Loss Validação'], 
                        label='Loss Validação', linewidth=2, color='red', linestyle='--')
                    
                    # Destacar a melhor época se disponível
                    if 'best_epoch' in metrics and metrics['best_epoch'] is not None:
                        best_epoch = metrics['best_epoch']
                        if best_epoch < len(loss_df):
                            ax.axvline(x=best_epoch + 1, color='green', linestyle=':', 
                                    alpha=0.7, label=f'Melhor Época ({best_epoch + 1})')
                    
                    ax.set_xlabel('Época')
                    ax.set_ylabel('Loss')
                    ax.set_title('Evolução da Loss durante o Treinamento')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Mostrar tabela com dados numéricos
                with st.expander("📋 Ver dados numéricos do histórico"):
                    st.dataframe(loss_df)
            
            # Informações de configuração usadas
            st.subheader("⚙️ Configuração do Treinamento")
            config = st.session_state.config
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write("**Hiperparâmetros:**")
                st.write(f"- Learning Rate: {config.get('learning_rate', 'N/A')}")
                st.write(f"- Batch Size: {config.get('batch_size', 'N/A')}")
                st.write(f"- Épocas: {config.get('n_epochs', 'N/A')}")
                st.write(f"- Otimizador: {config.get('optimizer', 'N/A').upper()}")
            
            with config_col2:
                st.write("**Arquitetura:**")
                st.write(f"- Camadas Ocultas: {config.get('hidden_layers', 'N/A')}")
                st.write(f"- Dropout: {config.get('dropout_rate', 'N/A')}")
                st.write(f"- Normalização: {config.get('normalization', 'N/A')}")
                st.write(f"- Early Stopping: {'Sim' if config.get('enable_early_stopping', False) else 'Não'}")
        
        else:
            st.warning("Nenhum resultado de treinamento disponível.")
    
    # ------------------------------
    # Seção de Previsão
    # ------------------------------
    def prediction_section(self):
        st.header("🔮 Fazer Previsões Individuais")
        model = st.session_state.get('current_model', None)
        # Garantir que o modelo já foi treinado
        if model is None:
            st.warning("⚠️ Treine um modelo primeiro para fazer previsões.")
            if st.button("🚀 Ir para Treinamento"):
                st.session_state.section = "🚀 Treinar Modelo"
                st.experimental_rerun()
            return

        st.subheader("Características do Imóvel")

        col1, col2 = st.columns(2)

        with col1:
            accommodates = st.slider("Número de Hóspedes", 1, 16, 2, key="acc")
            bedrooms = st.slider("Quartos", 0, 10, 1, key="bedr")
            bathrooms = st.slider("Banheiros", 0.0, 10.0, 1.0, 0.5, key="bath")
            beds = st.slider("Camas", 0, 10, 1, key="beds")

        with col2:
            neighbourhoods = self.df['neighbourhood_cleansed'].unique() if 'neighbourhood_cleansed' in self.df.columns else ['Copacabana','Ipanema','Leblon']
            neighbourhood = st.selectbox("Bairro", neighbourhoods, key="neigh")

            room_types = self.df['room_type'].unique() if 'room_type' in self.df.columns else ['Entire home/apt','Private room']
            room_type = st.selectbox("Tipo de Quarto", room_types, key="room")

        if st.button("🎯 Prever Preço", key="predict_btn"):
            input_data = {
                'accommodates': accommodates,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'beds': beds,
                'neighbourhood_cleansed': neighbourhood,
                'room_type': room_type
            }
            input_df = pd.DataFrame([input_data])
            try:
                result = st.session_state.current_model.predict_dataframe(input_df)
                predicted_price = result['predicted_price'].iloc[0]

                st.success(f"💰 Preço Previsto: R$ {predicted_price:.2f}")

                # Comparar com média do bairro
                if 'price' in self.df.columns:
                    avg_price = self.df[self.df['neighbourhood_cleansed'] == neighbourhood]['price'].mean()
                    diff = predicted_price - avg_price
                    diff_percent = (diff / avg_price) * 100
                    st.metric("Vs Média do Bairro", f"R$ {diff:+.2f} ({diff_percent:+.1f}%)")

            except Exception as e:
                st.error(f"❌ Erro na previsão: {e}")



    def debug_session_state(self):
        def safe_default(obj):
            """Fallback for non-serializable objects"""
            try:
                return str(obj)
            except Exception:
                return "UNSERIALIZABLE"

        st.subheader("🛠 Debug: Session State")
        st.json(json.loads(json.dumps(dict(st.session_state), default=safe_default)))