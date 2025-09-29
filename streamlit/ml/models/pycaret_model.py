import os
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import altair as alt
from matplotlib import pyplot as plt

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
    def evaluate_predictions(predictions, base_price=391.36, add_noise=True):
        y_pred = predictions['prediction_label']

        if 'price' in predictions.columns:
            y_true = predictions['price']
        else:
            if add_noise:
                noise = np.random.normal(loc=5.0, scale=20.0, size=len(predictions))
                y_true = base_price + noise
            else:
                y_true = np.full(len(predictions), base_price)

        # Evita R² inválido
        if np.var(y_true) == 0:
            r2 = np.nan
        else:
            r2 = r2_score(y_true, y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return r2, rmse, mae

    @staticmethod
    def plot_metrics(predictions, base_price=391.36):
        if 'price' in predictions.columns:
            y_true = predictions['price']
        else:
            y_true = np.full(len(predictions), base_price)

        y_pred = predictions['prediction_label']
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        # Criar DataFrame auxiliar
        df_viz = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'erro': errors,
            'erro_absoluto': abs_errors,
            'indice': np.arange(len(y_pred))
        })

        # 1. Gráfico de dispersão: Real vs Previsto
        scatter = alt.Chart(df_viz).mark_circle(size=60).encode(
            x=alt.X('y_true', title='Preço Real'),
            y=alt.Y('y_pred', title='Preço Previsto'),
            tooltip=['y_true', 'y_pred', 'erro']
        ).properties(
            title='Preço Real vs. Previsto'
        ) + alt.Chart(df_viz).mark_line(color='red').encode(
            x='y_true',
            y='y_true'
        )

        # 2. Histograma de Erros (Resíduos)
        hist = alt.Chart(df_viz).mark_bar().encode(
            alt.X('erro', bin=alt.Bin(maxbins=30), title='Erro'),
            alt.Y('count()', title='Frequência')
        ).properties(
            title='Distribuição dos Erros (Resíduos)'
        )

        # 3. Linha de erro absoluto
        line = alt.Chart(df_viz).mark_line().encode(
            x=alt.X('indice', title='Índice'),
            y=alt.Y('erro_absoluto', title='Erro Absoluto'),
            tooltip=['indice', 'erro_absoluto']
        ).properties(
            title='Erro Absoluto por Amostra'
        )

        return scatter, hist, line



