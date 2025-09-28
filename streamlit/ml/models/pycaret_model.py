import os
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class PyCaretModel:
    @staticmethod
    def load(filename='melhor_modelo_pycaret'):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, filename)
        return load_model(model_path.replace('.pkl', ''))

    @staticmethod
    def predict(new_data, filename='melhor_modelo_pycaret.pkl'):
        model = PyCaretModel.load(filename)

        if isinstance(new_data, dict):
            new_df = pd.DataFrame([new_data])
        else:
            new_df = new_data.copy()

        predictions = predict_model(model, data=new_df, verbose=False)

        return predictions

    @staticmethod
    def evaluate_predictions(predictions, base_price=391.36):
        """
        Calcula métricas de avaliação R², RMSE e MAE.
        Se 'price' não estiver nas colunas, usa um valor base fictício para calcular.

        Parameters:
        - predictions: DataFrame retornado pelo predict_model
        - base_price: valor assumido como verdadeiro para cálculo de métricas (default: 100)

        Returns:
        - r2, rmse, mae
        """

        if 'price' in predictions.columns:
            y_true = predictions['price']
        else:
            # Assumir valor base como real (repetido para cada linha)
            y_true = np.full(len(predictions), base_price)

        y_pred = predictions['prediction_label']

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return r2, rmse, mae

