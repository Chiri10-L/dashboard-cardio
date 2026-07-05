
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import joblib

app = dash.Dash(__name__)
server = app.server

# Cargar datos
df = pd.read_csv('../cardio_train_clean.csv')

# Gráfica
fig = px.histogram(df, x='age', color='cardio', 
                   title='Distribución de Edad por Condición Cardiovascular',
                   barmode='group')

# Layout
app.layout = html.Div([
    html.H1("🫀 Dashboard Cardiovascular", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig),
    html.Hr(),
    html.Label("Filtrar por rango de edad:"),
    dcc.RangeSlider(
        id='slider-edad',
        min=18, max=100, step=1,
        value=[30, 70],
        marks={i: str(i) for i in range(20, 101, 20)}
    ),
    html.Div(id='output-edad', style={'marginTop': '20px', 'fontSize': '18px'}),
    html.Hr(),
    html.P("Proyecto Final - Universidad Tecnológica de Panamá - 2026")
])

@app.callback(Output('output-edad', 'children'), Input('slider-edad', 'value'))
def update_edad(rango):
    min_e, max_e = rango
    df_filtrado = df[(df['age'] >= min_e) & (df['age'] <= max_e)]
    total = len(df_filtrado)
    enfermos = len(df_filtrado[df_filtrado['cardio'] == 1])
    sanos = len(df_filtrado[df_filtrado['cardio'] == 0])
    return f"👥 Total: {total} | 🟢 Sanos: {sanos} | 🔴 Enfermos: {enfermos}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
