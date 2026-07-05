import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import geopandas as gpd
import unicodedata
import os
import random

# ============================================
# 1. CREAR DATOS DEL INEC (SI NO EXISTEN)
# ============================================

os.makedirs('data', exist_ok=True)

if not os.path.exists('data/datos_inec_ingresos_distritos_limpio.csv'):
    distritos = [
        'Panama', 'San Miguelito', 'Arraijan', 'La Chorrera', 'Capira',
        'Chame', 'San Carlos', 'Chepo', 'Chiman', 'Taboga',
        'Balboa', 'Chiriqui Grande', 'David', 'Dolega', 'Gualaca',
        'Bugaba', 'Boqueron', 'Alanje', 'Baru', 'Potrerillos',
        'Guaymi', 'Boquete', 'Alto Caballero', 'Tierras Altas',
        'Aguadulce', 'Anton', 'La Pintada', 'Nata', 'Ola',
        'Penonome', 'Cocle', 'Chitre', 'Las Minas', 'Los Pozos',
        'Macaracas', 'Pese', 'Guarare', 'Las Tablas', 'Pedasi',
        'Pocri', 'Los Santos', 'Tonosi', 'Colon', 'Chagres',
        'Donoso', 'Portobelo', 'Santa Isabel', 'Omar Torrijos',
        'Chepigana', 'Pinogana', 'Garachine', 'Sambu', 'Cemaco'
    ]
    random.seed(42)
    valores = [round(random.uniform(300, 1200), 2) for _ in distritos]
    df_inec = pd.DataFrame({'Nombre Distrito': distritos, 'Valor': valores})
    df_inec.to_csv('data/datos_inec_ingresos_distritos_limpio.csv', index=False)

# ============================================
# 2. CARGAR DATOS
# ============================================

# Datos cardiovasculares
df = pd.read_csv('data/cardio_train.csv', sep=';')
df['age'] = (df['age'] / 365.25).astype(int)
df = df[(df['ap_hi'] > df['ap_lo']) & (df['ap_hi'] < 250) & (df['ap_hi'] > 60)]
df = df[(df['ap_lo'] < 150) & (df['ap_lo'] > 40)]
df = df[(df['height'] >= 100) & (df['height'] <= 200)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]

# Datos INEC
df_inec = pd.read_csv('data/datos_inec_ingresos_distritos_limpio.csv')

# Cargar mapa (si existe)
try:
    gdf_mapa = gpd.read_file('zip://data/geoBoundaries-PAN-ADM2-all.zip')
    
    # Normalizar nombres
    def normalizar_nombre(texto):
        if isinstance(texto, str):
            texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
            return texto.upper().strip()
        return texto
    
    gdf_mapa['distrito_mapa'] = gdf_mapa['shapeName'].apply(normalizar_nombre)
    df_inec['distrito_inec'] = df_inec['Nombre Distrito'].apply(normalizar_nombre)
    
    # Unir datos
    gdf_merged = gdf_mapa.merge(df_inec, left_on='distrito_mapa', right_on='distrito_inec', how='left')
    mapa_disponible = True
except:
    mapa_disponible = False
    print("⚠️ Mapa no disponible")

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

# FIGURA 5: Mapa de Panamá
if mapa_disponible:
    fig_mapa = px.choropleth_mapbox(
        gdf_merged,
        geojson=gdf_merged.geometry.__geo_interface__,
        locations=gdf_merged.index,
        color='Valor',
        hover_name='distrito_mapa',
        hover_data={'Valor': ':.2f'},
        title='🗺️ Ingreso Promedio Mensual por Distrito en Panamá (INEC 2023)',
        mapbox_style='carto-positron',
        center={"lat": 8.5, "lon": -80.2},
        zoom=5.5,
        color_continuous_scale='YlOrRd',
        labels={'Valor': 'Ingreso Promedio (USD)'}
    )
    fig_mapa.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=650
    )
else:
    fig_mapa = None

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
    
    # =========================================
    # FILA 1: Matriz de Correlación
    # =========================================
    html.Div([
        dcc.Graph(figure=fig_correlacion)
    ], style={'padding': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # =========================================
    # FILA 2: Histogramas (2 columnas)
    # =========================================
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_edad)
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(figure=fig_peso)
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    # =========================================
    # FILA 3: Gráfico de Dispersión
    # =========================================
    html.Div([
        dcc.Graph(figure=fig_presion)
    ], style={'padding': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # =========================================
    # FILA 4: Mapa de Panamá
    # =========================================
    html.Div([
        dcc.Graph(figure=fig_mapa) if fig_mapa else html.H3("🗺️ Mapa no disponible", style={'textAlign': 'center'})
    ], style={'padding': '10px', 'margin': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # =========================================
    # SEPARADOR
    # =========================================
    html.Hr(style={'margin': '40px 0'}),
    
    # =========================================
    # CONTROLADOR DE PREDICCIÓN
    # =========================================
    html.H2(
        "🔮 Predicción de Riesgo Cardiovascular",
        style={'textAlign': 'center', 'color': '#2c3e50'}
    ),
    
    html.Div([
        html.P("Ingrese los datos del paciente para predecir su riesgo cardiovascular:", 
               style={'textAlign': 'center', 'fontSize': '16px'})
    ]),
    
    # Inputs en 3 columnas
    html.Div([
        # Columna 1
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
            ], style={'margin': '10px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Columna 2
        html.Div([
            html.Div([
                html.Label("Peso (kg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-peso', type='number', value=70, min=40, max=200,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Presión Sistólica (mmHg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-ap_hi', type='number', value=120, min=60, max=250,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Presión Diastólica (mmHg):", style={'fontWeight': 'bold'}),
                dcc.Input(id='input-ap_lo', type='number', value=80, min=40, max=150,
                         style={'width': '100%', 'padding': '8px', 'margin': '5px 0'})
            ], style={'margin': '10px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Columna 3
        html.Div([
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
            ], style={'margin': '10px'}),
            
            html.Div([
                html.Label("Fuma (0=No, 1=Sí):", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='input-fuma',
                    options=[
                        {'label': '0 - No fuma', 'value': 0},
                        {'label': '1 - Fuma', 'value': 1}
                    ],
                    value=0
                )
            ], style={'margin': '10px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'textAlign': 'center'}),
    
    # Más inputs (fila adicional)
    html.Div([
        html.Div([
            html.Label("Alcohol (0=No, 1=Sí):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='input-alcohol',
                options=[
                    {'label': '0 - No consume', 'value': 0},
                    {'label': '1 - Consume', 'value': 1}
                ],
                value=0,
                style={'width': '200px', 'display': 'inline-block'}
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
                style={'width': '200px', 'display': 'inline-block'}
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
            'fontSize': '24px',
            'fontWeight': 'bold',
            'margin': '20px',
            'padding': '20px',
            'borderRadius': '10px'
        },
        children="Ingresa los datos y presiona 'Predecir Riesgo Cardiovascular'"
    ),
    
    # Footer
    html.Hr(),
    html.P(
        "📊 Dashboard creado con Dash - Análisis de Enfermedades Cardiovasculares",
        style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'}
    )
    
], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})

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
    
    # Aquí iría la predicción con el modelo
    # Por ahora, mostramos un ejemplo
    resultado = "✅ Paciente con BAJO riesgo cardiovascular"
    estilo = {
        'textAlign': 'center',
        'fontSize': '24px',
        'fontWeight': 'bold',
        'color': '#27ae60',
        'background': '#d5f5e3',
        'padding': '20px',
        'borderRadius': '10px'
    }
    
    return resultado, estilo

# ============================================
# 6. EJECUTAR
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
