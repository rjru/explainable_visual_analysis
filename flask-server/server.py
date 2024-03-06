from flask import Flask
import numpy as np
from sklearn.externals import joblib
from flask_restful import Api
from flask_cors import CORS
from resources.projection_resource import *
from resources.explanation_resource import *
from alg_exp.contrast_exp.projection import *
from alg_exp.contrast_exp.explanation import *
import os

# https://www.youtube.com/watch?v=7LNl2JlZKHA

# RUTAS DE LOS DATOS
#base_data_path = 'C:/Users/rjru/Documents/proyect_initial_nodejs/data/'
#data_path = os.path.join(base_data_path, 'iris_converted_new.json')

# SE CARGA EL ARCHIVO data.json
#with open(data_path, 'r') as json_fin:
#    json_data = json.load(json_fin)

dataset_desc = "iris"
model_desc = "linear" 
computation_method = "fancy" 
standarization = True

n_explanations = 3
ind_for_exp=0
objetive_exp=np.array([0.0,  0.0])

pj = Projection(dataset_desc, standarization, model_desc)
exp_node = Explanation(pj, n_explanations, computation_method, model_desc)

"""
Flask starts here
"""
app = Flask(__name__)
api = Api(app)

# CORS for development purpose only
CORS(app, resources={r"/api/*": {"origins": "*"}})

api.add_resource(ProjectionResource,
                 '/api/projection_resource',
                 resource_class_args=(pj,))

api.add_resource(ExplanationResource,
                 '/api/explanation_resource',
                 resource_class_args=(pj, exp_node))


if __name__ == '__main__':
    port = int(5000)
    app.run(debug=True, host='127.0.0.1', port=port)