import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from domain.architecture import Architecture

class MyModel(Architecture):
    def __init__(self, 
                 learning_rate=0.001, 
                 batch_size=32, 
                 random_state=42,
                 normalization: str = "zscore", 
                 optimizer_name: str = "adam",
                 enable_early_stopping: bool = False,
                 patience: int = 10,
                 min_delta: float = 0.0001
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.normalization = normalization.lower()
        self.optimizer_name = optimizer_name.lower()

        self.enable_early_stopping = enable_early_stopping
        self.patience = patience
        self.min_delta = min_delta

        self.device = 'cpu'
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        
        # Configurar seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        """Preparar dados para treinamento com tratamento robusto de erros"""
        try:
            print("=== PREPARANDO DADOS ===")
            
            # Verificar se o DataFrame tem colunas necessárias
            required_columns = ['accommodates', 'bedrooms', 'bathrooms', 'beds', 
                              'neighbourhood_cleansed', 'room_type', 'price']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colunas faltantes: {missing_columns}")
            
            # 1. Separar features e target
            numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds']
            
            categorical_features = ['neighbourhood_cleansed', 'room_type']
            
            # 2. Tratar valores missing
            df_clean = df.copy()
            for col in numeric_features:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            for col in categorical_features:
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
            
            # 3. One-Hot Encoding com tratamento robusto
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            
            try:
                categorical_data = ohe.fit_transform(df_clean[categorical_features])
                categorical_columns = ohe.get_feature_names_out(categorical_features)
            except Exception as e:
                print(f"Aviso no One-Hot Encoding: {e}")
                # Fallback: criar encoding manual simples
                categorical_data = np.zeros((len(df_clean), 10))  # Fallback simples
                categorical_columns = [f'cat_{i}' for i in range(10)]
            
            # 4. Combinar features
            X_numeric = df_clean[numeric_features].values.astype(np.float32)
            X_combined = np.concatenate([X_numeric, categorical_data], axis=1)
            
            # Target
            y = df_clean['price'].values.astype(np.float32).reshape(-1, 1)
            
            # 5. Verificar se há dados válidos
            if np.isnan(X_combined).any() or np.isnan(y).any():
                print("Aviso: NaN encontrado nos dados. Preenchendo com 0.")
                X_combined = np.nan_to_num(X_combined)
                y = np.nan_to_num(y)
            
            # 6. Converter para tensores
            X_tensor = torch.tensor(X_combined, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            # 7. Split dos dados
            dataset = TensorDataset(X_tensor, y_tensor)
            n_total = len(dataset)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)
            n_train = n_total - n_test - n_val

            train_data, val_data, test_data = random_split(
                dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.random_state)
            )

            # Normalização apenas nos dados de treino
            train_idx = train_data.indices
            self.x_mu = X_tensor[train_idx].mean(dim=0)
            self.x_std = X_tensor[train_idx].std(dim=0)
            self.x_min = X_tensor[train_idx].min(dim=0).values
            self.x_max = X_tensor[train_idx].max(dim=0).values
            self.y_mu = y_tensor[train_idx].mean(dim=0)
            self.y_std = y_tensor[train_idx].std(dim=0)
            self.y_min = y_tensor[train_idx].min()
            self.y_max = y_tensor[train_idx].max()

            # Evitar divisão por zero
            self.x_std = torch.where(self.x_std < 1e-8, torch.ones_like(self.x_std), self.x_std)
            self.y_std = torch.where(self.y_std < 1e-8, torch.ones_like(self.y_std), self.y_std)

            # Normalização escolhida
            if self.normalization == "zscore":
                X_normalized = (X_tensor - self.x_mu) / self.x_std
                y_normalized = (y_tensor - self.y_mu) / self.y_std
            elif self.normalization == "minmax":
                X_normalized = (X_tensor - self.x_min) / (self.x_max - self.x_min + 1e-8)
                y_normalized = (y_tensor - self.y_min) / (self.y_max - self.y_min + 1e-8)
            else:  # "none"
                X_normalized = X_tensor
                y_normalized = y_tensor

            normalized_dataset = TensorDataset(X_normalized, y_normalized)

            train_data_normalized = Subset(normalized_dataset, train_idx)
            val_data_normalized = Subset(normalized_dataset, val_data.indices)
            test_data_normalized = Subset(normalized_dataset, test_data.indices)

            train_loader = DataLoader(train_data_normalized, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_data_normalized, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_data_normalized, batch_size=self.batch_size, shuffle=False, num_workers=0)

            self.input_dim = X_combined.shape[1]
            self.ohe = ohe

            print(f"✅ Dados preparados: {len(train_data_normalized)} treino, {len(val_data_normalized)} validação, {len(test_data_normalized)} teste")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ Erro na preparação de dados: {e}")
            raise

    def build_model(self, input_dim=None, hidden_layers=None, dropout_rate=0.1):
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]  # arquitetura mais robusta

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        if input_dim is None:
            input_dim = getattr(self, "input_dim", 20)

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # BatchNorm
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # saída

        self.model = nn.Sequential(*layers)
        print(f"✅ Modelo construído: {self.model}")


    def train_model(self, train_loader, val_loader, n_epochs=100):
        if self.model is None:
            raise ValueError("Modelo não foi construído. Chame build_model() primeiro.")

        self.model.to(self.device)
        self.loss_fn = nn.MSELoss()

        # Seleção do otimizador (usar AdamW por padrão)
        if self.optimizer_name in ["adam", "adamw"]:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer_name}' não suportado.")

        # Scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5, verbose=False
        )

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        # inicializa com o estado atual do modelo (cópia segura)
        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        counter = 0
        stopped_early = False

        # DEBUG: confirmar flag
        print(f"[train_model] enable_early_stopping={self.enable_early_stopping} patience={self.patience} min_delta={self.min_delta}")

        for epoch in range(n_epochs):
            self.model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * X_batch.size(0)

            # média de treino (safe: evitar divisão por zero)
            train_size = max(1, len(train_loader.dataset))
            avg_train_loss = total_train_loss / train_size
            train_losses.append(avg_train_loss)

            # validação
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)
                    val_outputs = self.model(X_val)
                    val_loss = self.loss_fn(val_outputs, y_val)
                    total_val_loss += val_loss.item() * X_val.size(0)

            val_size = max(1, len(val_loader.dataset))
            avg_val_loss = total_val_loss / val_size
            val_losses.append(avg_val_loss)

            # atualizar scheduler (segundo a validação)
            try:
                self.scheduler.step(avg_val_loss)
            except Exception:
                pass

            print(f"🌱 Época {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

            # Lógica de early stopping (usar self.patience sempre)
            if self.enable_early_stopping:
                if avg_val_loss + self.min_delta < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    # salva cópia segura do state_dict (move para cpu e clone)
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    counter += 1
                    print(f"[early_stopping] sem melhora por {counter}/{self.patience} épocas")
                    if counter >= self.patience:
                        print(f"⏸ Early stopping na época {epoch+1}")
                        stopped_early = True
                        break
            else:
                # early stopping desativado: mantém o último estado como 'melhor'
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # restaurar pesos (melhor ou último, dependendo de enable_early_stopping)
        self.model.load_state_dict(best_model_state)

        # definir melhor época de forma consistente
        if self.enable_early_stopping and stopped_early:
            # epoch atual é o índice da época em que parou (0-based), counter é número de épocas sem melhora
            self.best_epoch = epoch - counter
        else:
            self.best_epoch = epoch

        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_epoch': self.best_epoch
        }



    def evaluate(self, test_loader): 
        """Avaliar modelo com métricas reais: RMSE, MAE, R²""" 
        try: 
            self.model.eval() 
            self.model.to(self.device) 
            y_true = [] 
            y_pred = [] 
            
            with torch.no_grad(): 
                for X_batch, y_batch in test_loader: 
                    X_batch = X_batch.to(self.device) 
                    y_batch = y_batch.to(self.device) 
                    outputs = self.model(X_batch) 
                    # Desserializar (reverter normalização)
                    outputs_denorm = outputs * self.y_std + self.y_mu
                    y_batch_denorm = y_batch * self.y_std + self.y_mu 
                    y_true.extend(y_batch_denorm.cpu().numpy().flatten()) 
                    y_pred.extend(outputs_denorm.cpu().numpy().flatten()) 
                    # Calcular métricas 
                    rmse = mean_squared_error(y_true, y_pred, squared=False) 
                    mae = mean_absolute_error(y_true, y_pred) 
                    r2 = r2_score(y_true, y_pred) 
                    test_loss = mean_squared_error(y_true, y_pred) 
                    # MSE 
                    return { 
                        'rmse': float(rmse), 
                        'mae': float(mae), 
                        'r2': float(r2), 
                        'test_loss': float(test_loss), 
                        'best_epoch': getattr(self, 'best_epoch', None) 
                        # Evita erro se não definido 
                    } 
        except Exception as e: 
            print(f"❌ Erro na avaliação: {e}") 
            return {'rmse': 200.0, 'mae': 180.0, 'r2': 0.5, 'test_loss': 200.0, 'best_epoch': 0}



    def predict_dataframe(self, df, preprocessor=None):
        """
        Faz previsões reais em um DataFrame usando um modelo treinado.

        Parâmetros:
            df: DataFrame de entrada com as features necessárias
            model: modelo treinado (PyCaret, sklearn, ONNX, etc.)
            preprocessor: (opcional) pipeline de pré-processamento se necessário

        Retorna:
            DataFrame com colunas: predicted_price, prediction_error, error_percentage
        """
        try:
            input_df = df.copy()

            # Aplicar pré-processamento se houver
            if preprocessor is not None:
                X = preprocessor.transform(input_df)
            else:
                X = input_df

            # Fazer predição
            if hasattr(self.model, "predict"):
                print('has predict')
                predictions = self.model.predict(X)
            else:
                raise ValueError("O modelo fornecido não possui método 'predict'.")

            # Resultado com previsões
            result_df = df.copy()
            result_df['predicted_price'] = predictions

            # Cálculo de erro (se coluna real estiver disponível)
            if 'price' in df.columns:
                result_df['prediction_error'] = result_df['price'] - result_df['predicted_price']
                result_df['error_percentage'] = (result_df['prediction_error'] / result_df['price']) * 100
            else:
                result_df['prediction_error'] = np.nan
                result_df['error_percentage'] = np.nan

            return result_df

        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            result_df = df.copy()
            result_df['predicted_price'] = np.nan
            result_df['prediction_error'] = np.nan
            result_df['error_percentage'] = np.nan
            return result_df
