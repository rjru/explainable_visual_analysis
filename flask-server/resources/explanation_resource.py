from flask_restful import Resource, Api, abort
from flask import request
import numpy as np

class ExplanationResource(Resource):

    def __init__(self, pj, exp_node):
        self.pj = pj
        self.ex_node = exp_node

    def post(self):
        nodeData = request.get_json(force=True)
        
        self.ex_node.compute_node(nodeData['index'], np.array(nodeData['coordinates']))
        self.ex_node.destandardized_data()

        # Convert the explanations and other attributes to a format that can be easily serialized
        explanations_list = [[float(value) for value in explanation_instance] for explanation_instance in self.ex_node.explanations[0]]
        transformed_samples_dist_list = [float(value) for value in self.ex_node.transformed_samples_dist]
        cf_transformed_samples_error_list = [[float(value) for value in error_list] for error_list in self.ex_node.cf_transformed_sanmples_error]
        X_cf_list = [[float(value) for value in x_cf] for x_cf in self.ex_node.X_cf_]
        Y_cf_pred_list = [[float(value) for value in instance] for instance in self.ex_node.Y_cf_pred[0]]
        X_cf_D_list = [[float(value) for value in instance] for instance in self.ex_node.X_cf_D]
        
        # Create a dictionary to hold the data
        explain_node = {
            "explanations": explanations_list,
            "transformed_samples_dist": transformed_samples_dist_list,
            "cf_transformed_samples_error": cf_transformed_samples_error_list,
            "X_cf_": X_cf_list,
            "Y_cf_pred": Y_cf_pred_list,
            "X_cf_D": X_cf_D_list
        }

        return explain_node
