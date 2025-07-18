import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prever_nivel_rio(dados_entrada):
    model_carregado = joblib.load('modelo_regressao_linear.pkl')
    entrada = pd.DataFrame([dados_entrada], columns=['NívelItuporanga', 'ChuvaItuporanaga'])  # Ajuste as colunas conforme features
    scaler = MinMaxScaler()
    entrada_scaled = scaler.fit_transform(entrada)
    previsao = model_carregado.predict(entrada_scaled)
    return previsao[0]
