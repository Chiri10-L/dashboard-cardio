import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html

# ============================================
# COLORES / ESTILO
# ============================================
COLORS = {'muted': '#6c757d', 'card': '#ffffff', 'primary': '#3b5bdb'}
CARD_STYLE = {
    'backgroundColor': COLORS['card'], 'borderRadius': '16px',
    'padding': '20px', 'margin': '14px',
    'boxShadow': '0 4px 16px rgba(0,0,0,0.06)',
}

# ============================================
# 1. CARGA DE DATOS DEL MAPA
# ============================================
try:
    with open('data/panama_geo.json', encoding='utf-8') as f:
        panama_geojson = json.load(f)
    df_panama = pd.read_csv('data/panama_ingresos.csv')
    print(f"✅ Mapa de Panamá cargado ({len(df_panama)} distritos)")
except Exception as e:
    panama_geojson = None
    df_panama = None
    print(f"⚠️ Datos del mapa de Panamá NO encontrados: {e}")
    print(f"📂 Directorio actual (cwd): {os.getcwd()}")
    print(f"📂 Archivos/carpetas en la raíz: {os.listdir('.')}")
    if os.path.isdir('data'):
        print(f"📂 Contenido de la carpeta 'data/': {os.listdir('data')}")
    else:
        print("📂 La carpeta 'data/' NO existe en este directorio")


# ============================================
# 2. FIGURA DEL MAPA
# ============================================
def crear_mapa_panama():
    if panama_geojson is None or df_panama is None:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{
                'text': '⚠️ No se encontraron data/panama_geo.json y data/panama_ingresos.csv',
                'showarrow': False, 'font': {'size': 14, 'color': COLORS['muted']}
            }],
            height=550, xaxis={'visible': False}, yaxis={'visible': False},
            font_family='Inter, sans-serif'
        )
        return fig

    fig = px.choropleth_mapbox(
        df_panama,
        geojson=panama_geojson,
        locations='shapeName',
        featureidkey='properties.shapeName',
        color='ingreso',
        color_continuous_scale='YlGnBu',
        range_color=(df_panama['ingreso'].min(), df_panama['ingreso'].max()),
        mapbox_style='carto-positron',   # no requiere token de Mapbox
        zoom=6.1,
        center={'lat': 8.6, 'lon': -80.3},
        opacity=0.78,
        hover_name='shapeName',
        hover_data={'provincia': True, 'ingreso': ':.0f', 'shapeName': False},
        labels={'ingreso': 'Ingreso promedio (USD/mes)', 'provincia': 'Provincia'},
    )
    fig.update_layout(
        title='Ingreso Promedio Mensual por Persona, por Distrito (INEC 2023)',
        title_font_size=16, title_x=0.02,
        margin=dict(l=0, r=0, t=50, b=0),
        height=580,
        font_family='Inter, sans-serif',
        coloraxis_colorbar=dict(title='USD/mes'),
    )
    return fig


fig_mapa_panama = crear_mapa_panama()

# ============================================
# 3. APP DASH (mínima, solo el mapa)
# ============================================
app = dash.Dash(__name__)
app.title = "Mapa Panamá - Ingresos por Distrito"
server = app.server  # <- esto es lo que gunicorn busca como 'app:server'

app.layout = html.Div([
    html.Div([
        html.H1(
            "🗺️ Mapa de Panamá — Ingreso Promedio por Distrito",
            style={'textAlign': 'center', 'color': 'white', 'margin': 0, 'fontSize': '26px'}
        ),
    ], style={
        'background': f"linear-gradient(120deg, {COLORS['primary']}, #2c46b0)",
        'padding': '28px 20px', 'borderRadius': '0 0 20px 20px', 'marginBottom': '10px',
    }),

    html.Div([
        html.P(
            "Mapa interactivo a nivel de distritos de Panamá. Pasa el mouse sobre cada "
            "distrito para ver su provincia e ingreso promedio mensual por persona (INEC 2023).",
            style={'color': COLORS['muted'], 'fontSize': '14px', 'textAlign': 'center'}
        ),
        dcc.Graph(figure=fig_mapa_panama, config={'scrollZoom': True}),
    ], style={**CARD_STYLE, 'maxWidth': '1100px', 'margin': '0 auto'}),

], style={'backgroundColor': '#f4f6fa', 'minHeight': '100vh', 'paddingBottom': '20px',
          'fontFamily': 'Inter, sans-serif'})

if __name__ == '__main__':
    app.run(debug=True)
