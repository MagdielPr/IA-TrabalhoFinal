import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Carregar o modelo, o scaler e as features
model = joblib.load('modelo_regressao_linear.pkl')
scaler = joblib.load('scaler.pkl')
FEATURES = joblib.load('features.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/features')
def get_features():
    print("Features retornadas:", FEATURES)
    return jsonify({'features': FEATURES}), 200, {'Content-Type': 'application/json; charset=utf-8'}

@app.route('/prever', methods=['POST'])
def prever():
    data = request.get_json()
    print("Dados recebidos:", data)
    dados_entrada = {feature: float(data.get(feature, 0.0)) for feature in FEATURES}
    print("Dados convertidos:", dados_entrada)
    entrada = pd.DataFrame([dados_entrada], columns=FEATURES)
    entrada_scaled = scaler.transform(entrada)
    print("Dados normalizados:", entrada_scaled.flatten())
    previsao = model.predict(entrada_scaled)[0]
    print("Previsão:", previsao)
    return jsonify({'previsao': round(previsao, 2)})

if __name__ == '__main__':
    app.run(debug=True)
