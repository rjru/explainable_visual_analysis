from flask import Flask
import numpy as np
from sklearn.externals import joblib
from flask_restful import Api
from flask_cors import CORS
import json
from resources.data import Data
from alg_exp.contrast_exp.ContrastingExplanation_api import *
# from resources.path import Path
import os

# https://www.youtube.com/watch?v=7LNl2JlZKHA

# RUTAS DE LOS DATOS
# base_data_path = 'data/samples/shuttle/'
# data_path = os.path.join(base_data_path, 'data.json')

# RUTAS DE LOS DATOS
base_data_path = 'C:/Users/rjru/Documents/proyect_initial_nodejs/data/'
data_path = os.path.join(base_data_path, 'iris_converted_new.json')

# SE CARGA EL ARCHIVO data.json
with open(data_path, 'r') as json_fin:
    json_data = json.load(json_fin)


"""
Flask starts here
"""
app = Flask(__name__)
api = Api(app)

# CORS for development purpose only
CORS(app, resources={r"/api/*": {"origins": "*"}})

api.add_resource(Data,
                 '/api/data',
                 resource_class_args=(json_data,))

api.add_resource(ExplanationResource,
                 '/api/explanation',
                 resource_class_args=(json_data,))


if __name__ == '__main__':
    port = int(5000)
    app.run(debug=True, host='127.0.0.1', port=port)