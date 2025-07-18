# app.py
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Para sistemas Windows
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo e as features
model = joblib.load('modelo_regressao_linear.pkl')
FEATURES = joblib.load('features.pkl')  # Carregar as features salvas

@app.route('/')
def home():
    return render_template('index.html')  # Renderizar o HTML da pasta templates

@app.route('/features')
def get_features():
    return jsonify({'features': FEATURES}), 200, {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/prever', methods=['POST'])
def prever():
    data = request.get_json()
    dados_entrada = {feature: data.get(feature, 0.0) for feature in FEATURES}
    entrada = pd.DataFrame([dados_entrada], columns=FEATURES)
    # Normalização manual
    entrada_scaled = (entrada - entrada.min()) / (entrada.max() - entrada.min())
    entrada_scaled = entrada_scaled.fillna(0)
    previsao = model.predict(entrada_scaled)[0]
    return jsonify({'previsao': round(previsao, 2)})

if __name__ == '__main__':
    app.run(debug=True)