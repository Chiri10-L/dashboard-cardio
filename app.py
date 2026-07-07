import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

# ============================================
# COLORES / ESTILO
# ============================================
COLORS = {
    'muted': '#6c757d', 'card': '#ffffff', 'primary': '#3b5bdb',
    'primary_dark': '#2c46b0', 'accent': '#0ca678', 'text': '#212529', 'bg': '#f4f6fa',
}
CARD_STYLE = {
    'backgroundColor': COLORS['card'], 'borderRadius': '16px',
    'padding': '20px', 'margin': '14px',
    'boxShadow': '0 4px 16px rgba(0,0,0,0.06)',
}
KPI_STYLE = {
    'backgroundColor': COLORS['card'], 'borderRadius': '14px',
    'padding': '18px 20px', 'margin': '8px', 'flex': '1', 'minWidth': '190px',
    'boxShadow': '0 4px 16px rgba(0,0,0,0.06)', 'textAlign': 'center',
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

PROVINCIAS = sorted(df_panama['provincia'].unique()) if df_panama is not None else []


# ============================================
# 2. FIGURA DEL MAPA
# ============================================
def crear_mapa_panama(provincia_sel=None):
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

    df = df_panama.copy()
    zoom, center = 6.1, {'lat': 8.6, 'lon': -80.3}
    opacity_col = [1.0 if (not provincia_sel or provincia_sel == 'Todas' or p == provincia_sel) else 0.15
                   for p in df['provincia']]

    fig = px.choropleth_mapbox(
        df,
        geojson=panama_geojson,
        locations='shapeName',
        featureidkey='properties.shapeName',
        color='ingreso',
        color_continuous_scale='YlGnBu',
        range_color=(df_panama['ingreso'].min(), df_panama['ingreso'].max()),
        mapbox_style='carto-positron',
        zoom=zoom,
        center=center,
        hover_name='shapeName',
        hover_data={'provincia': True, 'ingreso': ':.0f', 'shapeName': False},
        labels={'ingreso': 'Ingreso promedio (USD/mes)', 'provincia': 'Provincia'},
    )
    fig.update_traces(marker_opacity=opacity_col, marker_line_width=0.5, marker_line_color='white')
    fig.update_layout(
        title='Ingreso Promedio Mensual por Persona, por Distrito (INEC 2023)',
        title_font_size=16, title_x=0.02,
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
        font_family='Inter, sans-serif',
        coloraxis_colorbar=dict(title='USD/mes'),
    )
    return fig


def crear_barras_provincias(provincia_sel=None):
    if df_panama is None:
        return go.Figure()

    resumen = df_panama.groupby('provincia', as_index=False)['ingreso'].mean()
    resumen = resumen.sort_values('ingreso', ascending=True)
    colores = [COLORS['primary'] if (not provincia_sel or provincia_sel == 'Todas' or p == provincia_sel)
               else '#d7dbe8' for p in resumen['provincia']]

    fig = go.Figure(go.Bar(
        x=resumen['ingreso'], y=resumen['provincia'], orientation='h',
        marker_color=colores,
        text=resumen['ingreso'].round(0).astype(int).astype(str) + ' USD',
        textposition='outside',
    ))
    fig.update_layout(
        title='Ingreso Promedio por Provincia', title_font_size=15, title_x=0.02,
        margin=dict(l=10, r=40, t=50, b=10), height=430,
        font_family='Inter, sans-serif',
        xaxis_title='USD/mes', yaxis_title=None,
        plot_bgcolor='white',
    )
    return fig


def calcular_kpis(provincia_sel=None):
    if df_panama is None:
        return None
    df = df_panama if (not provincia_sel or provincia_sel == 'Todas') else df_panama[df_panama['provincia'] == provincia_sel]
    fila_max = df.loc[df['ingreso'].idxmax()]
    fila_min = df.loc[df['ingreso'].idxmin()]
    return {
        'n_distritos': len(df),
        'ingreso_prom': df['ingreso'].mean(),
        'distrito_max': fila_max['shapeName'], 'valor_max': fila_max['ingreso'],
        'distrito_min': fila_min['shapeName'], 'valor_min': fila_min['ingreso'],
    }


def tarjeta_kpi(titulo, valor, color=None):
    return html.Div([
        html.P(titulo, style={'margin': 0, 'color': COLORS['muted'], 'fontSize': '13px', 'fontWeight': 600}),
        html.P(valor, style={'margin': '4px 0 0', 'fontSize': '22px', 'fontWeight': 800,
                              'color': color or COLORS['text']}),
    ], style=KPI_STYLE)


def render_kpis(provincia_sel=None):
    k = calcular_kpis(provincia_sel)
    if k is None:
        return html.Div()
    return html.Div([
        tarjeta_kpi("Distritos", f"{k['n_distritos']}"),
        tarjeta_kpi("Ingreso Promedio", f"${k['ingreso_prom']:.0f}", COLORS['primary']),
        tarjeta_kpi("Mayor Ingreso", f"{k['distrito_max']} — ${k['valor_max']:.0f}", COLORS['accent']),
        tarjeta_kpi("Menor Ingreso", f"{k['distrito_min']} — ${k['valor_min']:.0f}", '#e03131'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'maxWidth': '1100px', 'margin': '0 auto'})


# ============================================
# 3. APP DASH
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
        'background': f"linear-gradient(120deg, {COLORS['primary']}, {COLORS['primary_dark']})",
        'padding': '28px 20px', 'borderRadius': '0 0 20px 20px', 'marginBottom': '10px',
    }),

    html.Div([
        html.Label("Filtrar por provincia:", style={'fontWeight': 600, 'marginRight': '10px'}),
        dcc.Dropdown(
            id='filtro-provincia',
            options=[{'label': 'Todas las provincias', 'value': 'Todas'}] +
                    [{'label': p.title(), 'value': p} for p in PROVINCIAS],
            value='Todas', clearable=False,
            style={'maxWidth': '360px', 'display': 'inline-block', 'width': '100%'}
        ),
    ], style={'maxWidth': '1100px', 'margin': '0 auto 10px', 'padding': '0 14px'}),

    html.Div(id='kpis-container'),

    html.Div([
        html.Div([
            dcc.Loading(dcc.Graph(id='grafico-mapa', config={'scrollZoom': True})),
        ], style={**CARD_STYLE, 'flex': '1.4', 'minWidth': '400px'}),
        html.Div([
            dcc.Loading(dcc.Graph(id='grafico-barras', config={'displayModeBar': False})),
        ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '320px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'maxWidth': '1100px', 'margin': '0 auto'}),

], style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'paddingBottom': '20px',
          'fontFamily': 'Inter, sans-serif'})


@app.callback(
    Output('grafico-mapa', 'figure'),
    Output('grafico-barras', 'figure'),
    Output('kpis-container', 'children'),
    Input('filtro-provincia', 'value'),
)
def actualizar_dashboard(provincia_sel):
    return (
        crear_mapa_panama(provincia_sel),
        crear_barras_provincias(provincia_sel),
        render_kpis(provincia_sel),
    )


if __name__ == '__main__':
    app.run(debug=True)
