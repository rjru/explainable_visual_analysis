from flask_restful import Resource, Api, abort
from flask import request
import numpy as np

class ProjectionResource(Resource):

    def __init__(self, pj):
        self.pj = pj

    def get(self):
        self.pj.generate_projection()
        data_vectors_norm = self.pj.X.tolist()  #data_orig.data.tolist() en el caso de querer los datos orignales tal como se encuentran en la lib. 
        dim_names = self.pj.data_orig.feature_names
        labels_encoded = self.pj.data_orig.target.tolist()
        label_items = np.unique(self.pj.data_orig.target).tolist()
        label_names = self.pj.data_orig.target_names.tolist()

        json_data = {
            "dataVectors": data_vectors_norm,
            "dimNames": dim_names,
            "label": labels_encoded,
            "labelItems": label_items,
            "labelNames": label_names,  # En este caso, labelNames y labelItems serán iguales
            "initialProjections": {
                "pca": self.pj.X_red.tolist()
            }
        }

        return json_data

