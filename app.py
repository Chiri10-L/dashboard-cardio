import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import joblib
import os

# ============================================
# 1. DATOS EMBEBIDOS (CREADOS EN EL CÓDIGO)
# ============================================

print("📊 Creando datos de ejemplo...")

# Crear datos de ejemplo (los mismos que usaste en el notebook)
data = {
    'id': list(range(100)),
    'age': [50, 55, 51, 48, 47, 52, 60, 45, 58, 53, 
            49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
            50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
            49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
            50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
            49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
            50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
            49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
            50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
            49, 56, 44, 61, 42, 57, 46, 59, 54, 43],
    'gender': [2, 1, 1, 2, 1, 2, 1, 2, 1, 2] * 10,
    'height': [168, 156, 165, 169, 156, 160, 170, 155, 175, 165] * 10,
    'weight': [62, 85, 64, 82, 56, 70, 80, 55, 90, 65] * 10,
    'ap_hi': [110, 140, 130, 150, 100, 120, 130, 110, 140, 120] * 10,
    'ap_lo': [80, 90, 70, 100, 60, 80, 85, 70, 90, 80] * 10,
    'cholesterol': [1, 3, 3, 1, 1, 2, 3, 1, 2, 1] * 10,
    'gluc': [1, 1, 1, 1, 1, 2, 1, 1, 2, 1] * 10,
    'smoke': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0] * 10,
    'alco': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] * 10,
    'active': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 10,
    'cardio': [0, 1, 1, 1, 0, 1, 0, 0, 1, 0] * 10
}

df = pd.DataFrame(data)

print("✅ Datos de ejemplo creados correctamente")

# ============================================
# 2. CARGAR MODELOS (si existen)
# ============================================

try:
    modelo_clasificacion = joblib.load('models/clasificador_cardio.pkl')
    print("✅ Modelo de clasificación cargado")
except:
    modelo_clasificacion = None
    print("⚠️ Modelo de clasificación NO encontrado")

# ============================================
# 3. CREAR GRÁFICAS
# ============================================

# FIGURA 1: Matriz de Correlación
corr_matrix = df.select_dtypes(include=['number']).corr()
fig_correlacion = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale='RdBu',
    showscale=True,
    annotation_text=corr_matrix.round(2).values
)
fig_correlacion.update_layout(
    title='📊 Matriz de Correlación - Variables Cardiovasculares',
    height=650,
    font_size=10
)

# FIGURA 2: Distribución de Edad
fig_edad = px.histogram(
    df, 
    x='age', 
    color='cardio',
    barmode='overlay',
    title='📈 Distribución de Edad por Estado Cardiovascular',
    labels={'age': 'Edad (años)', 'count': 'Cantidad de Pacientes'},
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    nbins=30
)
fig_edad.update_layout(bargap=0.1)

# FIGURA 3: Distribución de Peso
fig_peso = px.histogram(
    df, 
    x='weight', 
    color='cardio',
    barmode='overlay',
    title='📈 Distribución de Peso por Estado Cardiovascular',
    labels={'weight': 'Peso (kg)', 'count': 'Cantidad de Pacientes'},
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    nbins=30
)
fig_peso.update_layout(bargap=0.1)

# FIGURA 4: Presión Sistólica vs Diastólica
fig_presion = px.scatter(
    df, 
    x='ap_hi', 
    y='ap_lo', 
    color='cardio',
    title='🫀 Relación entre Presión Sistólica y Diastólica',
    labels={
        'ap_hi': 'Presión Sistólica (mmHg)',
        'ap_lo': 'Presión Diastólica (mmHg)'
    },
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    opacity=0.5,
    hover_data=['age', 'weight']
)

# ============================================
# 4. DASHBOARD
# ============================================

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    
    # TÍTULO
    html.H1(
        "📊 Análisis de Enfermedades Cardiovasculares",
        style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px', 'marginBottom': '30px'}
    ),
    
    # FILA 1: Matriz de Correlación
    html.Div([
        dcc.Graph(figure=fig_correlacion)
    ], style={'padding': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # FILA 2: Histogramas (2 columnas)
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_edad)
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(figure=fig_peso)
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    # FILA 3: Gráfico de Dispersión
    html.Div([
        dcc.Graph(figure=fig_presion)
    ], style={'padding': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # SEPARADOR
    html.Hr(style={'margin': '40px 0'}),
    
    # CONTROLADOR DE PREDICCIÓN
    html.H2(
        "🔮 Predicción de Riesgo Cardiovascular",
        style={'textAlign': 'center', 'color': '#2c3e50'}
    ),
    
    html.Div([
        html.P("Ingrese los datos del paciente para predecir su riesgo cardiovascular:", 
               style={'textAlign': 'center', 'fontSize': '16px'})
    ]),
    
    # Inputs en 2 columnas (más compacto)
    html.Div([
        html.Div([
            html.Div([
                html.Label("Edad (años):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-edad', type='number', value=50, min=18, max=100, 
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Género (1=Mujer, 2=Hombre):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-genero', type='number', value=1, min=1, max=2,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Altura (cm):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-altura', type='number', value=165, min=100, max=200,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Peso (kg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-peso', type='number', value=70, min=40, max=200,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'})
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Div([
                html.Label("Presión Sistólica (mmHg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-ap_hi', type='number', value=120, min=60, max=250,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Presión Diastólica (mmHg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-ap_lo', type='number', value=80, min=40, max=150,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Colesterol (1=Normal, 2=Alto, 3=Muy Alto):", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='input-colesterol',
                    options=[
                        {'label': '1 - Normal', 'value': 1},
                        {'label': '2 - Alto', 'value': 2},
                        {'label': '3 - Muy Alto', 'value': 3}
                    ],
                    value=1
                )
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Glucosa (1=Normal, 2=Alto, 3=Muy Alto):", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='input-glucosa',
                    options=[
                        {'label': '1 - Normal', 'value': 1},
                        {'label': '2 - Alto', 'value': 2},
                        {'label': '3 - Muy Alto', 'value': 3}
                    ],
                    value=1
                )
            ], style={'margin': '10px'})
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'textAlign': 'center'}),
    
    # Fila adicional para fuma, alcohol, activo
    html.Div([
        html.Div([
            html.Label("Fuma (0=No, 1=Sí):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='input-fuma',
                options=[
                    {'label': '0 - No fuma', 'value': 0},
                    {'label': '1 - Fuma', 'value': 1}
                ],
                value=0,
                style={'width': '150px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'margin': '10px'}),
        
        html.Div([
            html.Label("Alcohol (0=No, 1=Sí):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='input-alcohol',
                options=[
                    {'label': '0 - No consume', 'value': 0},
                    {'label': '1 - Consume', 'value': 1}
                ],
                value=0,
                style={'width': '150px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'margin': '10px'}),
        
        html.Div([
            html.Label("Actividad Física (0=No, 1=Sí):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='input-activo',
                options=[
                    {'label': '0 - No activo', 'value': 0},
                    {'label': '1 - Activo', 'value': 1}
                ],
                value=1,
                style={'width': '150px', 'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'margin': '10px'})
    ], style={'textAlign': 'center'}),
    
    # Botón
    html.Div([
        html.Button(
            '🔍 Predecir Riesgo Cardiovascular',
            id='btn-predecir',
            n_clicks=0,
            style={
                'padding': '15px 40px',
                'fontSize': '18px',
                'background': '#3498db',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'margin': '20px',
                'fontWeight': 'bold'
            }
        )
    ], style={'textAlign': 'center'}),
    
    # Resultado
    html.Div(
        id='resultado-prediccion',
        style={
            'textAlign': 'center',
            'fontSize': '20px',
            'color': '#7f8c8d',
            'padding': '20px'
        },
        children="Ingresa los datos y presiona 'Predecir Riesgo Cardiovascular'"
    ),
    
    # Footer
    html.Hr(),
    html.P(
        "📊 Dashboard creado con Dash - Análisis de Enfermedades Cardiovasculares",
        style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'}
    )
    
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})

# ============================================
# 5. CALLBACK - PREDICCIÓN
# ============================================

@app.callback(
    Output('resultado-prediccion', 'children'),
    Output('resultado-prediccion', 'style'),
    Input('btn-predecir', 'n_clicks'),
    State('input-edad', 'value'),
    State('input-genero', 'value'),
    State('input-altura', 'value'),
    State('input-peso', 'value'),
    State('input-ap_hi', 'value'),
    State('input-ap_lo', 'value'),
    State('input-colesterol', 'value'),
    State('input-glucosa', 'value'),
    State('input-fuma', 'value'),
    State('input-alcohol', 'value'),
    State('input-activo', 'value')
)
def predecir(n_clicks, edad, genero, altura, peso, ap_hi, ap_lo, 
             colesterol, glucosa, fuma, alcohol, activo):
    
    if n_clicks == 0:
        return "Ingresa los datos y presiona 'Predecir Riesgo Cardiovascular'", {
            'textAlign': 'center',
            'fontSize': '20px',
            'color': '#7f8c8d',
            'padding': '20px'
        }
    
    # Verificar que el modelo existe
    if modelo_clasificacion is None:
        return "⚠️ Modelo no disponible. Entrena y guarda el modelo primero.", {
            'textAlign': 'center',
            'fontSize': '20px',
            'color': '#e67e22',
            'background': '#fef9e7',
            'padding': '20px',
            'borderRadius': '10px'
        }
    
    try:
        # Crear array con los datos en el MISMO orden del entrenamiento
        # ORDEN: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        datos = [[edad, genero, altura, peso, ap_hi, ap_lo, colesterol, glucosa, fuma, alcohol, activo]]
        
        prediccion = modelo_clasificacion.predict(datos)[0]
        
        if prediccion == 0:
            return "✅ Paciente con BAJO riesgo cardiovascular", {
                'textAlign': 'center',
                'fontSize': '24px',
                'fontWeight': 'bold',
                'color': '#27ae60',
                'background': '#d5f5e3',
                'padding': '20px',
                'borderRadius': '10px'
            }
        else:
            return "⚠️ Paciente con ALTO riesgo cardiovascular", {
                'textAlign': 'center',
                'fontSize': '24px',
                'fontWeight': 'bold',
                'color': '#c0392b',
                'background': '#fadbd8',
                'padding': '20px',
                'borderRadius': '10px'
            }
    except Exception as e:
        return f"❌ Error en la predicción: {str(e)}", {
            'textAlign': 'center',
            'fontSize': '20px',
            'color': '#c0392b',
            'background': '#fadbd8',
            'padding': '20px',
            'borderRadius': '10px'
        }

# ============================================
# 6. EJECUTAR
# ============================================

if __name__ == '__main__':
    app.run(debug=True)

