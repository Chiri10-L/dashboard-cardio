# app.py
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import joblib
import os

# ============================================
# 1. CARGAR DATOS Y MODELOS
# ============================================

# Cargar dataset
df = pd.read_csv('data/cardio_train.csv', sep=';')

# Limpieza básica (copia la que hiciste en el notebook)
df['age'] = (df['age'] / 365.25).astype(int)
df = df[(df['ap_hi'] > df['ap_lo']) & (df['ap_hi'] < 250) & (df['ap_lo'] < 150)]

# Cargar modelo (ajusta el nombre de tu archivo)
modelo = joblib.load('models/clasificador_cardio.pkl')

# ============================================
# 2. CREAR DASHBOARD
# ============================================

app = dash.Dash(__name__)
server = app.server  # Necesario para Render

app.layout = html.Div([

    html.H1("📊 Dashboard de Enfermedades Cardiovasculares", style={'textAlign': 'center'}),

    # ----- GRÁFICA 1: Distribución de Edad -----
    dcc.Graph(
        id='grafica-edad',
        figure=px.histogram(df, x='age', color='cardio',
                            title='Distribución de Edad por Estado Cardiovascular',
                            labels={'age': 'Edad (años)', 'count': 'Cantidad'})
    ),

    # ----- GRÁFICA 2: Presiones Arteriales -----
    dcc.Graph(
        id='grafica-presion',
        figure=px.scatter(df, x='ap_hi', y='ap_lo', color='cardio',
                          title='Presión Sistólica vs Diastólica',
                          labels={'ap_hi': 'Presión Sistólica', 'ap_lo': 'Presión Diastólica'})
    ),

    html.Hr(),

    # ----- CONTROLADOR: PREDICCIÓN -----
    html.H3("🔮 Predecir Riesgo Cardiovascular"),

    html.Label("Edad (años):"),
    dcc.Input(id='input-edad', type='number', value=50, style={'margin': '10px'}),

    html.Label("Presión Sistólica:"),
    dcc.Input(id='input-ap_hi', type='number', value=120, style={'margin': '10px'}),

    html.Label("Presión Diastólica:"),
    dcc.Input(id='input-ap_lo', type='number', value=80, style={'margin': '10px'}),

    html.Button('Predecir', id='btn-predecir', n_clicks=0,
                style={'margin': '10px', 'padding': '10px 20px', 'background': '#007bff', 'color': 'white'}),

    html.Div(id='resultado-prediccion', style={'fontSize': 20, 'fontWeight': 'bold', 'margin': '20px'})

])


# ============================================
# 3. CALLBACK PARA PREDICCIÓN
# ============================================

@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    Input('input-edad', 'value'),
    Input('input-ap_hi', 'value'),
    Input('input-ap_lo', 'value')
)
def predecir(n_clicks, edad, ap_hi, ap_lo):
    if n_clicks == 0:
        return "Ingresa los datos y presiona 'Predecir'"

    # Crear array con los datos (usa las mismas columnas que usaste en el entrenamiento)
    # Ajusta según tus variables predictoras
    datos = [[edad, ap_hi, ap_lo]]  # ¡AGREGA TODAS LAS VARIABLES QUE USASTE!

    prediccion = modelo.predict(datos)[0]

    if prediccion == 0:
        return "✅ El paciente tiene BAJO riesgo cardiovascular"
    else:
        return "⚠️ El paciente tiene ALTO riesgo cardiovascular"


# ============================================
# 4. EJECUTAR
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
