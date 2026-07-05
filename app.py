import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import geopandas as gpd
import joblib
import unicodedata
import os

# ============================================
# 1. CARGAR DATOS
# ============================================

# Cargar dataset cardiovascular
df = pd.read_csv('data/cardio_train.csv', sep=';')

# Limpieza (copia de tu notebook)
df['age'] = (df['age'] / 365.25).astype(int)
df = df[(df['ap_hi'] > df['ap_lo']) & (df['ap_hi'] < 250) & (df['ap_hi'] > 60)]
df = df[(df['ap_lo'] < 150) & (df['ap_lo'] > 40)]
df = df[(df['height'] >= 100) & (df['height'] <= 200)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]

# Cargar datos geográficos (MAPA)
ruta_mapa = 'data/geoBoundaries-PAN-ADM2-all.zip'
ruta_inec = 'data/datos_inec_ingresos_distritos_limpio.csv'

gdf_mapa = gpd.read_file(f'zip://{ruta_mapa}')

def normalizar_nombre(texto):
    if isinstance(texto, str):
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
        return texto.upper().strip()
    return texto

df_inec = pd.read_csv(ruta_inec)
gdf_mapa['distrito_mapa'] = gdf_mapa['shapeName'].apply(normalizar_nombre)
df_inec['distrito_inec'] = df_inec['Nombre Distrito'].apply(normalizar_nombre)
gdf_merged = gdf_mapa.merge(df_inec, left_on='distrito_mapa', right_on='distrito_inec', how='left')

# ============================================
# 2. CREAR FIGURAS
# ============================================

# FIGURA 1: Matriz de Correlación
corr_matrix = df.select_dtypes(include=['number']).corr()
fig_correlacion = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale='RdBu',
    showscale=True
)
fig_correlacion.update_layout(title='Matriz de Correlación - Variables Cardiovasculares', height=600)

# FIGURA 2: Histograma de Edad
fig_edad = px.histogram(
    df, x='age', color='cardio', barmode='overlay',
    title='Distribución de Edad por Estado Cardiovascular',
    labels={'age': 'Edad (años)', 'count': 'Cantidad'},
    color_discrete_map={0: '#1f77b4', 1: '#d62728'}
)

# FIGURA 3: Histograma de Peso
fig_peso = px.histogram(
    df, x='weight', color='cardio', barmode='overlay',
    title='Distribución de Peso por Estado Cardiovascular',
    labels={'weight': 'Peso (kg)', 'count': 'Cantidad'},
    color_discrete_map={0: '#1f77b4', 1: '#d62728'}
)

# FIGURA 4: Presión Sistólica vs Diastólica
fig_presion = px.scatter(
    df, x='ap_hi', y='ap_lo', color='cardio',
    title='Relación entre Presión Sistólica y Diastólica',
    labels={'ap_hi': 'Presión Sistólica (mmHg)', 'ap_lo': 'Presión Diastólica (mmHg)'},
    color_discrete_map={0: '#1f77b4', 1: '#d62728'},
    opacity=0.5
)

# FIGURA 5: Mapa de Panamá
fig_mapa = px.choropleth_mapbox(
    gdf_merged,
    geojson=gdf_merged.geometry.__geo_interface__,
    locations=gdf_merged.index,
    color='Valor',
    hover_name='distrito_mapa',
    hover_data={'Valor': ':.2f'},
    title='Ingreso Promedio Mensual por Distrito en Panamá (INEC 2023)',
    mapbox_style='carto-positron',
    center={"lat": 8.5, "lon": -80.2},
    zoom=5.5,
    color_continuous_scale='YlOrRd',
    labels={'Valor': 'Ingreso (USD)'}
)
fig_mapa.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=600)

# ============================================
# 3. CREAR DASHBOARD
# ============================================

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("📊 Análisis de Enfermedades Cardiovasculares", style={'textAlign': 'center', 'marginBottom': 30}),

    html.Div([
        dcc.Graph(figure=fig_correlacion),
        dcc.Graph(figure=fig_edad),
        dcc.Graph(figure=fig_peso),
        dcc.Graph(figure=fig_presion),
        dcc.Graph(figure=fig_mapa),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)
