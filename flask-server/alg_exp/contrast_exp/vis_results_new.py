# https://github.com/andreArtelt/ContrastingExplanationDimRed
# https://stackoverflow.com/questions/71509489/dash-cytoscape-drag-and-drop-callback

# Importar las bibliotecas necesarias
import dash
import dash_table
import numpy as np
from dash import Input, Output, State, ctx
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from sklearn.decomposition import PCA

from ContrastingExplanation import ContrastingExplanation # estas son las funciones adecuadas de experiments.py para la visualización


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
ce = None
elements = []
columns = []
# Crear el grafo de Cytoscape
cyto_graph = cyto.Cytoscape(
    id='cytoscape-graph',
    layout={'name': 'preset'},
    stylesheet=stylesheet,
    elements=elements,
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
            html.Div(id='output-box'),

            html.Div([
                dcc.Checklist(
                    id='add-nodes',
                    options=[{'label': 'Mostrar explicaciones proyectadas', 'value': 'on'}],
                    value=[]
                )
            ])
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

# Definir la función de actualización del panel lateral y el datatable
@app.callback(
    [Output('details-content', 'children'), Output('details_table', 'data'), Output('add-nodes', 'value')],
    [Input('cytoscape-graph', 'tapNode')]
)
def update_details_panel(node):
    if node is None:
        return html.P('Seleccione un nodo para ver sus detalles.'), [], []
    else:
        records = None
        id_act = int(node["data"]["id"])
        x_orig, y_orig = elements[id_act]["position"]["x"], elements[id_act]["position"]["y"]
        x_act, y_act = node["position"]["x"], node["position"]["y"]
        # AQUI VAMOS A COMPARAR SI CAMBIÓ DE UBICACIÓN
        pos_equal = False
        if x_act==x_orig and y_act==y_orig:
            pos_equal = True
        else:
            #raw_result = exp_func.compute_count_exp(G_X, G_y, G_X_red, G_model_desc, G_model, id_act, [x_act, y_act], G_computation_method, G_n_explanations)
            ce.generate_explanation(id_act, [x_act/500, y_act/500])
            ce.exp_node.destandardized_data()

            node_orig = [ce.pj.data_orig.data[int(node['data']['id'])]]

            exp_desest = node_orig + ce.exp_node.X_cf_D

            # Si se ha seleccionado un nodo, obtener los datos originales de iris correspondientes
            data = pd.DataFrame(np.concatenate([exp_desest]), columns=ce.pj.data_orig.feature_names)
            data['Tipo'] = ['Original', 'exp1', 'exp2', 'exp3']
            data = data.round(2)

            #print("dataxatpdic: ", data.to_dict('records'))
            # Devolver los datos en un formato compatible con el DataTable
            records = data.to_dict('records')

        return html.Div([
            html.P(f'ID: {id_act}'),
            html.P(f'Clase: {node["classes"]}'),
            html.P(f'Coordenadas originales: ({x_orig}, {y_orig})'),
            html.P(f'Coordenadas actuales: ({x_act}, {y_act})'),
            html.P(f'Posición igual: ({pos_equal})'),
        ]), records, []

@app.callback(
    [Output('cytoscape-graph', 'elements'), Output('details_table', 'columns')],
    Input('boton', 'n_clicks'),
    Input('add-nodes', 'value'),
    State('select-box-dataset', 'value'),
    State('select-box-dim-reduc', 'value'),
    State('select-box-com', 'value'),
    State('input-num-exp', 'value'),    
    prevent_initial_call=True
)

def create_nodes_vis(n_clicks, value, dataset_selected, dim_red_selected, compu_selected, num_exp_selected):
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    global ce, elements, columns

    if triggered == 'boton' and n_clicks:
        elements = []
        ce = ContrastingExplanation(dataset_selected, dim_red_selected, compu_selected, num_exp_selected, standarization=True)
        ce.create_projection()

        df = pd.DataFrame(data=ce.pj.X_red, columns=['x', 'y'])
        df['target'] = ce.pj.y

        # Crear el grafo de Cytoscape
        for i in range(len(df)):
            element = {'data': {'id': str(i), 'label': str(i)},
                    'position': {'x': df.x[i]*500, 'y': df.y[i]*500},
                    'classes': f'clase-{df.target[i]}'}
            #print("elementooo: ", element)
            elements.append(element)
        # cuando cargamos los datos, también asignamos los atributos en la tabla inferior. 
        columns = [{'name': e, 'id': e} for e in ce.pj.data_orig.feature_names] + [{'name': 'Desc', 'id': 'Tipo'}]
        #clases = np.unique(y)
        return elements, columns
    
    elif triggered == 'add-nodes' and 'on' in value:
        #print("Checkbox selecionado: ", ce.exp_node.Y_cf_pred)
        updated_elements = elements.copy()
        new_elements = []
        df = pd.DataFrame(data=np.concatenate(ce.exp_node.Y_cf_pred), columns=['x', 'y'])
        #print(df)
        # añadir al grafo de Cytoscape
        id_act = len(elements)
        for i in range(len(df)):
            element = {'data': {'id': str(id_act), 'label': str(id_act)},
                       'position': {'x': df.x[i]*500, 'y': df.y[i]*500},
                       'classes': 'clase-5'}
            id_act = id_act + 1
            new_elements.append(element)

        updated_elements.extend(new_elements)
        #print("updated_elements: ", updated_elements)
        return updated_elements, columns
    else:   
        return elements, columns

'''
# Definimos nuestra función de actualización
@app.callback(
    Output('cytoscape-graph', 'elements'),
    [Input('add-nodes', 'value')]
)
def update_elements(value):
    if 'on' in value:
        updated_elements = elements.copy()
        new_elements = []
        df = pd.DataFrame(data=ce.exp_node.Y_cf_pred, columns=['x', 'y'])
        # añadir al grafo de Cytoscape
        for i in range(len(df)):
            element = {'data': {'id': str(i), 'label': str(i)},
                       'position': {'x': df.x[i]*500, 'y': df.y[i]*500},
                       'classes': '.clase-9'}
            new_elements.append(element)

        updated_elements.extend(new_elements)
        return updated_elements
    else:
        return elements
'''


if __name__ == '__main__':
    app.run_server(debug=False)

