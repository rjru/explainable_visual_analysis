import numpy as np
from flask_restful import Resource, reqparse
from alg_exp.contrast_exp.ContrastingExplanation import ContrastingExplanation
from flask import request

class ExplanationResource(Resource):
    def __init__(self, json_data):
        self.json_data = json_data
        
    def post(self):
        '''
        print("hola munde")
        json_data = self.json_data
        data_vectors = json_data['dataVectors']
        print(data_vectors)

        vectors = np.array(json_data['dataVectors'])
        labels = np.array(json_data['label'])
        label_items = np.array(json_data['labelItems'])
        '''
        req_all = request.get_json(force=True)
        print(req_all)

        # No estoy haciendo uso de los datos enviados desde el servidor. 

        dataset_desc = "iris" 
        model_desc = "linear" 
        computation_method = "fancy" 
        standarization = True

        n_explanations = 3
        ind_for_exp=0
        objetive_exp=np.array([0.0,  0.0])

        ce = ContrastingExplanation(model_desc, computation_method, n_explanations)
        ce.create_projection(dataset_desc, standarization)
        #ce.generate_explanation(ind_for_exp, objetive_exp)
        #ce.exp_node.destandardized_data()

        json_output = ce.convert_to_json()
        # print(json_output)

        return True
