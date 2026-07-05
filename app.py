# ============================================
# app.py - Dashboard Cardiovascular
# Universidad Tecnológica de Panamá - 2026
# ============================================

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 Iniciando Dashboard Cardiovascular...")

# ============================================
# 1. CONFIGURACIÓN DE LA APP
# ============================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ============================================
# 2. CARGAR DATOS (CON FALLBACK)
# ============================================

print("📊 Cargando datos...")

# Intentar cargar el archivo CSV
try:
    # Buscar en diferentes ubicaciones
    rutas = [
        '../cardio_train_clean.csv',
        'cardio_train_clean.csv',
        '/opt/render/project/src/cardio_train_clean.csv'
    ]
    
    df = None
    for ruta in rutas:
        try:
            df = pd.read_csv(ruta)
            print(f"✅ Datos cargados desde: {ruta}")
            break
        except:
            continue
    
    # Si no se encontró el archivo, crear datos de ejemplo
    if df is None:
        print("⚠️ No se encontró el archivo CSV. Creando datos de ejemplo...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'gender': np.random.choice([1, 2], n),
            'height(cm)': np.random.randint(150, 190, n),
            'weight(kg)': np.random.uniform(50, 100, n),
            'ap_hi(mmHg)': np.random.randint(100, 180, n),
            'ap_lo(mmHg)': np.random.randint(60, 120, n),
            'cholesterol': np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]),
            'gluc': np.random.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]),
            'smoke': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'alco': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'active': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'cardio': np.random.choice([0, 1], n, p=[0.5, 0.5])
        })
        print("✅ Datos de ejemplo creados (1000 registros)")
        
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
    # Crear datos de emergencia
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'cardio': np.random.choice([0, 1], 100)
    })
    print("✅ Datos de emergencia creados")

print(f"📊 Dataset: {len(df)} registros, {len(df.columns)} columnas")

# ============================================
# 3. CREAR GRÁFICAS
# ============================================

print("📈 Creando gráficas...")

# Gráfica 1: Distribución de Edad
fig1 = px.histogram(
    df, x='age', color='cardio',
    title='Distribución de Edad por Condición Cardiovascular',
    labels={'age': 'Edad (años)', 'count': 'Cantidad'},
    barmode='group',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
    nbins=20
)

# Gráfica 2: Presión Sistólica
fig2 = px.box(
    df, x='cardio', y='ap_hi(mmHg)',
    title='Presión Sistólica por Condición',
    labels={'cardio': 'Condición', 'ap_hi(mmHg)': 'Presión Sistólica (mmHg)'},
    color='cardio',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
)

# Gráfica 3: Matriz de Correlación
cols_corr = ['age', 'ap_hi(mmHg)', 'ap_lo(mmHg)', 'cholesterol', 'gluc', 'cardio']
corr_matrix = df[cols_corr].corr()
fig3 = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    aspect='auto',
    title='Matriz de Correlación'
)

# Gráfica 4: Factores de Riesgo
df_melted = df.melt(
    id_vars=['cardio'],
    value_vars=['cholesterol', 'gluc'],
    var_name='Factor',
    value_name='Nivel'
)
nivel_map = {1: 'Normal', 2: 'Alto', 3: 'Muy Alto'}
df_melted['Nivel_Label'] = df_melted['Nivel'].map(nivel_map)

fig4 = px.histogram(
    df_melted,
    x='Nivel_Label',
    color='cardio',
    facet_col='Factor',
    title='Colesterol y Glucosa por Condición',
    labels={'count': 'Cantidad'},
    barmode='group',
    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
)

print("✅ Gráficas creadas")

# ============================================
# 4. LAYOUT DEL DASHBOARD
# ============================================

app.layout = html.Div([
    # Título
    html.Div([
        html.H1("🫀 Predicción de Enfermedades Cardiovasculares",
                style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px 0'}),
        html.P("Dashboard interactivo para análisis de riesgo cardiovascular",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Gráficas
    html.Div([
        html.H2("📊 Análisis de Datos", style={'margin': '30px 0 20px 0'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig1)], className='six columns', style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig2)], className='six columns', style={'width': '48%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=fig3)], className='six columns', style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig4)], className='six columns', style={'width': '48%', 'display': 'inline-block'})
        ])
    ]),
    
    # Controladores
    html.Div([
        html.H2("🎛️ Controladores", style={'margin': '30px 0 20px 0'}),
        html.Div([
            html.Label("Filtrar por Edad:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='slider-edad',
                min=int(df['age'].min()),
                max=int(df['age'].max()),
                step=1,
                value=[int(df['age'].min()), int(df['age'].max())],
                marks={i: str(i) for i in range(20, 81, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(id='output-edad', style={'marginTop': '15px', 'padding': '15px', 
                                               'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'})
    ]),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P(
            "Proyecto Final - Análisis de Datos | Universidad Tecnológica de Panamá | 2026",
            style={'textAlign': 'center', 'color': '#7f8c8d'}
        )
    ])
], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})

# ============================================
# 5. CALLBACKS
# ============================================

@app.callback(
    Output('output-edad', 'children'),
    Input('slider-edad', 'value')
)
def update_edad(rango):
    if rango:
        min_e, max_e = rango
        df_filtrado = df[(df['age'] >= min_e) & (df['age'] <= max_e)]
        total = len(df_filtrado)
        enfermos = len(df_filtrado[df_filtrado['cardio'] == 1])
        sanos = len(df_filtrado[df_filtrado['cardio'] == 0])
        return html.Div([
            html.Span(f"👥 Total: {total} pacientes | ", style={'fontWeight': 'bold'}),
            html.Span(f"🟢 Sanos: {sanos} | ", style={'color': '#2ecc71'}),
            html.Span(f"🔴 Enfermos: {enfermos}", style={'color': '#e74c3c'})
        ])

# ============================================
# 6. EJECUTAR
# ============================================

print("✅ Dashboard listo para ejecutar!")
print("🌐 Servidor iniciado en http://0.0.0.0:8050")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

