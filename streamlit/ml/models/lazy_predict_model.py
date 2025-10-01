import os
import pickle
import pandas as pd
import numpy as np
from joblib import dump, load
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
import onnxruntime as ort


class LazyPredictModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.pipeline = None

        # Colunas
        self.target_col = 'price'
        self.categorical_cols = ['neighbourhood', 'room_type']
        self.numerical_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds']

    def preprocess_and_train(self):
        # Separar features e target
        X = self.df[self.categorical_cols + self.numerical_cols]
        y = self.df[self.target_col]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # LazyRegressor (apenas para comparação)
        reg = LazyRegressor(verbose=0, ignore_warnings=True, random_state=42)
        models, _ = reg.fit(X_train, X_test, y_train, y_test)
        print(models.head())

        # Preprocessadores
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numerical_cols),
            ('cat', categorical_transformer, self.categorical_cols)
        ])

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])

        # Treinar pipeline
        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline  # Para salvar posteriormente

        # Avaliar
        y_pred = self.pipeline.predict(X_test)
        return {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

    def save(self, model_path='lazy_model.onnx', preprocessor_path='preprocessor.pkl'):
        if self.pipeline:
            # Salva o pré-processador separadamente
            preprocessor = self.pipeline.named_steps['preprocessor']
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            # Extrai os dados de exemplo
            X_example = self.df[self.categorical_cols + self.numerical_cols].iloc[:1]
            initial_types = [
                (col, StringTensorType([None, 1])) if X_example[col].dtype == object else
                (col, FloatTensorType([None, 1]))
                for col in X_example.columns
            ]

            # Converte para ONNX
            onnx_model = convert_sklearn(self.pipeline, initial_types=initial_types)
            with open(model_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

    @staticmethod
    def predict(model, input_data, base_price=100):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        pred = model.predict(input_df)

        # Simular métricas se não houver valor real
        y_true = np.full(len(pred), base_price)
        y_pred = pred

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return float(pred[0]), r2, rmse, mae

    @staticmethod
    def predict_onnx(input_data: dict,
                     model_path="lazy_model.onnx",
                     preprocessor_path="preprocessor.pkl",
                     base_price=100):

        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, model_path)
        preprocessor_path = os.path.join(base_path, preprocessor_path)
        # Carrega o pré-processador
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        # Transforma input
        input_df = pd.DataFrame([input_data])
        X_processed = preprocessor.transform(input_df).astype(np.float32)

        # ONNX prediction
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: X_processed})

        pred = prediction[0][0]

        # Simular métricas
        r2 = r2_score([base_price], [pred])
        rmse = np.sqrt(mean_squared_error([base_price], [pred]))
        mae = mean_absolute_error([base_price], [pred])

        return float(pred), r2, rmse, mae
