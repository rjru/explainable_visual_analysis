# https://github.com/andreArtelt/ContrastingExplanationDimRed
# https://stackoverflow.com/questions/71509489/dash-cytoscape-drag-and-drop-callback

# Importar las bibliotecas necesarias
import dash
import dash_table
import numpy as np
from dash.dependencies import Output, Input, State
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from sklearn.decomposition import PCA

import exp_func # estas son las funciones adecuadas de experiments.py para la visualización


dataset_list = [{'label': 'iris', 'value': 'iris'},
                {'label': 'toy', 'value': 'toy'},
                {'label': 'breastcancer', 'value': 'breastcancer'},
                {'label': 'wine', 'value': 'wine'},
                {'label': 'boston', 'value': 'boston'},
                {'label': 'diabetes', 'value': 'diabetes'},
                {'label': 'digits', 'value': 'digits'}]

dim_reduc_list = [{'label': 'PCA method', 'value': 'linear'},
                {'label': 'Autoencoders method', 'value': 'ae'},
                {'label': 'SOM method', 'value': 'som'},
                {'label': 'T-SNE method', 'value': 'tsne'}]

comp_list = [{'label': 'FANCY', 'value': 'fancy'},
             {'label': 'NO FANCY', 'value': 'NO_fancy'}]

selector_node = [ {'selector': 'node', 'style': {'background-color': 'gray', 'label': 'data(label)'}}]

stylesheet = selector_node + [  {'selector': '.clase-0', 'style': {'background-color': 'red'}},    {'selector': '.clase-1', 'style': {'background-color': 'blue'}},
                                {'selector': '.clase-2', 'style': {'background-color': 'green'}},  {'selector': '.clase-3', 'style': {'background-color': 'maroon'}},
                                {'selector': '.clase-4', 'style': {'background-color': 'purple'}}, {'selector': '.clase-5', 'style': {'background-color': 'black'}},
                                {'selector': '.clase-6', 'style': {'background-color': 'gray'}},   {'selector': '.clase-7', 'style': {'background-color': 'fushsia'}},
                                {'selector': '.clase-8', 'style': {'background-color': 'olive'}},  {'selector': '.clase-9', 'style': {'background-color': 'navy'}},
                             ]


# Crear la aplicación de Dash
app = dash.Dash(__name__)

# variable global
G_data_orig = pd.DataFrame({})
elements = []
G_X, G_y, G_X_red, G_model, G_compu_selected, G_num_exp_selected, G_model_desc, G_computation_method, G_n_explanations  = None, None, None, None, None, None, None, None, None

# Crear el grafo de Cytoscape
cyto_graph = cyto.Cytoscape(
    id='cytoscape-graph',
    layout={'name': 'preset'},
    stylesheet=stylesheet,
    elements=[],
    style={'width': '100%', 'height': '600px', 'float': 'left'}
)

# Crear el panel principal
main_panel = html.Div([
    html.H1('Grafo de Cytoscape Dash'),
    html.P('Este es un ejemplo de grafo creado con el componente Cytoscape de Dash Cytoscape.'),
    cyto_graph
], style={'width': '70%', 'float': 'left'})


# Crear el panel lateral con detalles
sidebar_panel = html.Div(
    id='details-panel',
    children=[
        html.H3('Detalles'),
        html.Hr(),
        html.P('Seleccione un nodo para ver sus detalles.'),
        html.Div(id='details-content'),
        html.Div([
            html.Label('Selecciona un conjunto de datos:'),
            dcc.Dropdown(
                id='select-box-dataset',
                options=dataset_list,
                value=None
            ),
            html.Label('Selecciona método de reducción de dimensionalidad:'),
            dcc.Dropdown(
                id='select-box-dim-reduc',
                options=dim_reduc_list,
                value=None
            ),
            html.Label('Selecciona método computacional:'),
            dcc.Dropdown(
                id='select-box-com',
                options=comp_list,
                value=None
            ),
            html.Br(),
            html.Label('Introduce número de explicaciones:'),
            dcc.Input(
                id='input-num-exp',
                type='number',
                value=3,
            ),
            html.Br(),
            html.Button('PROYECTAR DATOS', id='boton'),
            html.Div(id='output-box')
        ])
    ],
    style={'width': '25%', 'float': 'left'}
)

# Crear el DataTable en el panel inferior
table = dash_table.DataTable(
    id='details_table',
    columns=[],
    style_cell={'textAlign': 'center'},
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ]
)

# Agregar el DataTable al panel inferior
footer_panel = html.Div([
    html.H2('Panel inferior'),
    html.P('Este es un panel inferior.'),
    table
], style={'width': '100%', 'float': 'left', 'position': 'absolute', 'bottom': 0})


# Crear el diseño de la página
app.layout = html.Div([
    main_panel,
    sidebar_panel,
    footer_panel
])

# Actualizar el panel inferior cuando se selecciona un nodo
@app.callback(
    Output('details_table', 'data'),
    Input('cytoscape-graph', 'tapNode')
)

def update_table(node):
    if node is None:
        # Si no se ha seleccionado ningún nodo, mostrar una tabla vacía
        return []
    else:
        # Si se ha seleccionado un nodo, obtener los datos originales de iris correspondientes
        data = pd.DataFrame([G_data_orig.data[int(node['data']['id'])]], columns=G_data_orig.feature_names)
        print("dataxatpdic: ", data.to_dict('records'))
        # Devolver los datos en un formato compatible con el DataTable
        return data.to_dict('records')


# Definir la función de actualización del panel lateral
@app.callback(
    dash.dependencies.Output('details-content', 'children'),
    [dash.dependencies.Input('cytoscape-graph', 'tapNode')]
)
def update_details_panel(node):
    if node is None:
        return html.P('Seleccione un nodo para ver sus detalles.')
    else:
        id_act = int(node["data"]["id"])
        x_orig, y_orig = elements[id_act]["position"]["x"], elements[id_act]["position"]["y"]
        x_act, y_act = node["position"]["x"], node["position"]["y"]
        # AQUI VAMOS A COMPARAR SI CAMBIÓ DE UBICACIÓN
        pos_equal = False
        if x_act==x_orig and y_act==y_orig:
            pos_equal = True
        else:
            raw_result = exp_func.compute_count_exp(G_X, G_y, G_X_red, G_model_desc, G_model, id_act, [x_act, y_act], G_computation_method, G_n_explanations)
            print('Explicaciones: ', raw_result)

        return html.Div([
            html.P(f'ID: {id_act}'),
            html.P(f'Clase: {node["classes"]}'),
            html.P(f'Coordenadas originales: ({x_orig}, {y_orig})'),
            html.P(f'Coordenadas actuales: ({x_act}, {y_act})'),
            html.P(f'Posición igual: ({pos_equal})'),
        ])

            


@app.callback(
    [Output('cytoscape-graph', 'elements'), Output('details_table', 'columns')],
    Input('boton', 'n_clicks'),
    State('select-box-dataset', 'value'),
    State('select-box-dim-reduc', 'value'),
    State('select-box-com', 'value'),
    State('input-num-exp', 'value'),    
)

def create_nodes_vis(n_clicks, dataset_selected, dim_red_selected, compu_selected, num_exp_selected):

    global G_data_orig, elements, G_X, G_y, G_X_red, G_model, G_compu_selected, G_num_exp_selected, G_model_desc, G_computation_method, G_n_explanations 
    columns = []
    if n_clicks:
        # aplicamos las funciones de archivo python exp_func
        data_orig, X, y, X_red, model = exp_func.run_projection_data(dataset_selected, dim_red_selected)

        G_X, G_y, G_X_red, G_model = X, y, X_red, model
        G_compu_selected, G_num_exp_selected, G_model_desc, G_computation_method, G_n_explanations = compu_selected, num_exp_selected, dim_red_selected, compu_selected, num_exp_selected

        G_data_orig = data_orig
        # Crear un DataFrame con los resultados del método de reducción
        df = pd.DataFrame(data=X_red, columns=['x', 'y'])
        df['target'] = y

        # Crear el grafo de Cytoscape
        for i in range(len(df)):
            element = {'data': {'id': str(i), 'label': str(i)},
                       'position': {'x': df.x[i]*500, 'y': df.y[i]*500},
                       'classes': f'clase-{df.target[i]}'}
            elements.append(element)
        
        # cuando cargamos los datos, también asignamos los atributos en la tabla inferior. 
        columns = [{'name': e, 'id': e} for e in data_orig.feature_names] 
        #clases = np.unique(y)
    return elements, columns



if __name__ == '__main__':
    app.run_server(debug=True)

