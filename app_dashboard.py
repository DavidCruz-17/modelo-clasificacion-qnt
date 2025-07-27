import pandas as pd
import joblib
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

# Cargar modelo y dataframes procesado

df_general = pd.read_parquet("df_general.parquet")
df_con_score = pd.read_parquet("df_con_score.parquet")
scorecard_sin = pd.read_parquet("scorecard_sin.parquet")

# Umbrales para deciles (opcional: puedes recalcular o guardar)
umbrales = df_con_score['probabilidad'].quantile([i/10 for i in range(11)]).values

# Coordenadas geográficas (simplificadas)
df_coordenadas = pd.DataFrame([
    {'CIUDAD_FINAL': 'BOGOTA D.C.', 'lat': 4.7110, 'lon': -74.0721},
    {'CIUDAD_FINAL': 'CALI', 'lat': 3.4516, 'lon': -76.5320},
    {'CIUDAD_FINAL': 'MEDELLIN', 'lat': 6.2442, 'lon': -75.5812},
    {'CIUDAD_FINAL': 'BARRANQUILLA', 'lat': 10.9639, 'lon': -74.7964},
    {'CIUDAD_FINAL': 'SIN INFORMACION', 'lat': 4.5, 'lon': -74.1}
])

conteo_ciudades = df_general['CIUDAD_FINAL'].value_counts().reset_index()
conteo_ciudades.columns = ['CIUDAD_FINAL', 'count']
ciudades_geo_original = pd.merge(conteo_ciudades, df_coordenadas, on='CIUDAD_FINAL', how='left')

# Arreglar nombres para gráfica
df_entidad = df_general['Entidad_principal'].value_counts().reset_index()
df_entidad.columns = ['Entidad', 'Cantidad']
fig_entidad_general = px.bar(
    df_entidad,
    x='Entidad', y='Cantidad',
    title='Entidad Principal',
    labels={'Entidad': 'Entidad', 'Cantidad': 'Cantidad'}
)

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H1("Tablero QNT – Priorización y Probabilidad de Contacto", className="text-center my-4"),
    dcc.Tabs([
        dcc.Tab(label='Información General', children=[
            dbc.Container([
                html.Div([
                    html.H5(f"Total clientes: {len(df_general):,}"),
                    html.H5(f"Saldo capital estimado: ${int(df_general['SMMV_Saldo_Capital_cliente'].sum() * 1400000):,}")
                ], className="mb-4 text-center"),
                dbc.Row([
                    dbc.Col(dcc.Graph(
                        figure=px.box(df_general, x='Tipo_Cliente', y='Edad__c', color='Tipo_Cliente')
                    ), md=6),
                    dbc.Col(dcc.Graph(figure=fig_entidad_general), md=6)
                ])
            ])
        ]),
        dcc.Tab(label='Clientes sin contacto', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Label("Filtrar por decil (1-10):"),
                        dcc.Dropdown(
                            id='decil-filtro',
                            options=[{'label': str(i), 'value': i} for i in range(1, 11)],
                            value=[10],
                            multi=True
                        )
                    ], width=6)
                ]),
                html.Div(id='resumen-decil'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='fig-genero'), md=6),
                    dbc.Col(dcc.Graph(id='fig-saldo'), md=6)
                ])
            ])
        ]),
        dcc.Tab(label='Predicción Cliente', children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([html.Label("Edad"), dcc.Input(id='inp-edad', type='number', value=35)], width=4),
                    dbc.Col([html.Label("Saldo Capital SMMV"), dcc.Input(id='inp-saldo', type='number', value=5)], width=4),
                    dbc.Col([html.Label("Puntaje"), dcc.Input(id='inp-puntaje', type='number', value=600)], width=4),
                ]),
                dbc.Row([
                    dbc.Col([html.Label("Tipo Cliente"), dcc.Dropdown(
                        id='inp-tipo', options=[{'label': x, 'value': x} for x in df_general['Tipo_Cliente'].unique()], value='0')], width=4),
                    dbc.Col([html.Label("Género"), dcc.Dropdown(
                        id='inp-genero', options=[{'label': x, 'value': x} for x in df_general['Genero__c'].dropna().unique()], value='MASCULINO')], width=4),
                    dbc.Col([html.Label("Entidad"), dcc.Dropdown(
                        id='inp-entidad', options=[{'label': x, 'value': x} for x in df_general['Entidad_principal'].dropna().unique()], value='ENTIDAD 1')], width=4),
                ]),
                dbc.Button("Predecir", id='btn-predecir', color='primary', className='mt-3'),
                html.Div(id='output-prediccion', className='mt-4')
            ])
        ])
    ])
])

@app.callback(
    Output('fig-genero', 'figure'),
    Output('fig-saldo', 'figure'),
    Output('resumen-decil', 'children'),
    Input('decil-filtro', 'value')
)
def actualizar_deciles(deciles):
    df_filtrado = df_con_score[df_con_score['decil'].isin(deciles)]
    fig1 = px.bar(df_filtrado, x='Genero__c', color='Tipo_Cliente', barmode='group', title='Género')
    fig2 = px.box(df_filtrado, x='Tipo_Cliente', y='SMMV_Saldo_Capital_cliente', color='Tipo_Cliente', title='Saldo')
    resumen = html.Div([
        html.H5(f"Clientes filtrados: {len(df_filtrado):,}"),
        html.H5(f"Saldo estimado: ${int(df_filtrado['SMMV_Saldo_Capital_cliente'].sum() * 1400000):,}")
    ])
    return fig1, fig2, resumen

@app.callback(
    Output('output-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    State('inp-edad', 'value'),
    State('inp-saldo', 'value'),
    State('inp-puntaje', 'value'),
    State('inp-tipo', 'value'),
    State('inp-genero', 'value'),
    State('inp-entidad', 'value')
)
def predecir(n, edad, saldo, puntaje, tipo, genero, entidad):
    if not n:
        return ""
    modelo = joblib.load("modelo_final.pkl")
    entrada = pd.DataFrame([{
        'Edad__c': edad,
        'SMMV_Saldo_Capital_cliente': saldo,
        'PUNTAJE_1_VALOR': puntaje,
        'Tipo_Cliente': tipo,
        'Genero__c': genero,
        'Entidad_principal': entidad
    }])
    entrada = pd.get_dummies(entrada).reindex(columns=modelo.feature_names_in_, fill_value=0)
    prob = modelo.predict_proba(entrada)[0][1]
    decil = next((i+1 for i in range(10) if prob <= umbrales[i+1]), 10)
    return html.Div([
        html.H5(f"Probabilidad de contacto: {prob:.2%}"),
        html.H5(f"Decil asignado: {decil}")
    ])

if __name__ == '__main__':
    app.run(debug=True,host ='0.0.0.0', port=8050)
