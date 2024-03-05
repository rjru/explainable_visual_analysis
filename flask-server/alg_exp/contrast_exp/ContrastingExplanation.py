import numpy as np
import pandas as pd
import json

# alg_exp.contrast_exp. SE DEBE AGREGAR A CADA from para que pueda acceder a las librerias cuando se usa flask.
# https://stackoverflow.com/questions/4534438/typeerror-module-object-is-not-callable
#from autoencoder import AutoencoderModel, AutoencoderDimRed
from alg_exp.contrast_exp.utils import load_data, standardize_data
from alg_exp.contrast_exp.linear_dr import LinearDimRed, LinearDimRedCf
#from ae_dr import AutoEncoderDimRedCf
from alg_exp.contrast_exp.som_dr import SomDimRedCf, SomDimRed
#from tsne_dr import TsneDimRed, TsneDimRedCf
from alg_exp.contrast_exp.memory_counterfactual import MemoryCounterfactual


class Projection:
    def __init__(self, dataset_desc, standarization, model_desc):
        self.dataset_desc = dataset_desc 
        self.standarization = standarization # True or False
        self.model_desc = model_desc

        self.data_orig = None # datos originales cargados
        self.X = None
        self.y = None
        self.X_red = None
        self.model = None
        self.stand_model = None

        self.explanations = None

    def generate_projection(self):
        # Load data
        self.data_orig, self.X, self.y = load_data(self.dataset_desc)
        # standarizamos los datos
        if self.standarization == True:
            self.X, self.stand_model = standardize_data(self.X)
        
        # Fit and apply dimensionality reduction method
        if self.model_desc == "linear":
            self.model = LinearDimRed()
        #elif self.model_desc == "ae":
        #    self.model = AutoencoderDimRed(AutoencoderModel(features=[10, 2], input_dim=self.X.shape[1]))
        elif self.model_desc == "som":
            self.model = SomDimRed()
        # elif self.model_desc == "tsne":
        #     self.model = TsneDimRed()
        else:
            raise ValueError("Unknown model")

        self.X_red = self.model.fit_transform(self.X)  


class Explanation:
    def __init__(self, pj, n_explanations, computation_method, model_desc):
        self.pj = pj
        self.n_explanations = n_explanations
        self.computation_method = computation_method
        self.model_desc = model_desc

        self.explanations = []
        self.transformed_samples_dist = []
        self.cf_transformed_sanmples_error = []

        self.X_cf_ = None
        self.Y_cf_pred = None # a partir de las explicaciones, se proyectaron los datos.
        self.X_cf_D = None

    def compute_node(self, ind_for_exp, objetive_exp):
        i=int(ind_for_exp) # el indice del elemento que queremos explicar. 
        x_orig = self.pj.X[i,:] # array original de características (4 caracteristicas de iris)
        y_orig = self.pj.X_red[i,:] # array original de datos proyectados (dos datos proyectados x, y)
        y_cf =  objetive_exp #X_perturbed_red[i,:] # array de datos proyectados "perturbados" (dos datos proyectados x, y)

        # Compute counterfactual explanation of dimensionality reduction
        if self.computation_method == "fancy":
            if self.model_desc == "linear":
                explainer = LinearDimRedCf(self.pj.model, C_pred=10.)
            #elif self.pj.model_desc == "ae":
            #    explainer = AutoEncoderDimRedCf(self.pj.model)
            elif self.model_desc == "som":
                explainer = SomDimRedCf(self.pj.model)
            # elif self.pj.model_desc == "tsne":
            #    explainer = TsneDimRedCf(self.pj.model)
        else:
            explainer = MemoryCounterfactual(self.pj.X, self.pj.X_red)


        transformed_dist = np.linalg.norm(y_orig - y_cf, 2) # se calcula la distancia entre los datos "perturbados" y los la proyección original.

        self.X_cf_ = explainer.compute_diverse_explanations(x_orig, y_cf, n_explanations=self.n_explanations)
        
        if self.X_cf_ is None:
            print("Al parecer no se pudo encontar una explicabilidad")

        self.Y_cf_pred = [self.pj.model.transform(x_cf) for x_cf in self.X_cf_] # se proyectan las explicaciones. 

        cf_transformed_error = [np.linalg.norm(y_cf - y_cf_pred, 2) for y_cf_pred in self.Y_cf_pred] # resta de las 3 proyecciones en relación a la proyección de los datos perturbados

        Delta_cf = [np.abs(x_orig - x_cf) for x_cf in self.X_cf_]

        self.explanations.append(Delta_cf)
        self.transformed_samples_dist.append(transformed_dist)
        self.cf_transformed_sanmples_error.append(cf_transformed_error)

    def destandardized_data(self):
        x_mean = self.pj.stand_model.mean_
        x_std = self.pj.stand_model.scale_
        self.X_cf_D = [(ce * x_std) + x_mean for ce in self.X_cf_]
        return self.X_cf_D


class ContrastingExplanation:
    def __init__(self, model_desc, computation_method, n_explanations):
        self.model_desc = model_desc   
        self.computation_method = computation_method
        self.n_explanations = n_explanations

        self.pj = None
        self.exp_node = None

    def create_projection(self, dataset_desc, standarization):
        self.pj = Projection(dataset_desc, standarization, self.model_desc)
        self.pj.generate_projection()

    def generate_explanation(self, ind_for_exp, objetive_exp):
        self.exp_node = Explanation(self.pj, self.n_explanations, self.computation_method, self.model_desc)
        self.exp_node.compute_node(ind_for_exp, objetive_exp)

    def convert_to_json(self):
        print("aqui tiene que ir a json")
        data_orig = self.pj.data_orig

        data_vectors_norm = self.pj.X.tolist()  #data_orig.data.tolist() en el caso de querer los datos orignales tal como se encuentran en la lib. 
        dim_names = data_orig.feature_names
        labels_encoded = data_orig.target.tolist()
        label_items = np.unique(data_orig.target).tolist()
        label_names = data_orig.target_names.tolist()

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

        #json_file_path = 'data\iris_converted_new.json'
        #with open(json_file_path, 'w') as json_file:
        #    json.dump(json_data, json_file, indent=4)

        return json_data


if __name__ == "__main__":

    dataset_desc = "iris" #sys.argv[1]
    model_desc = "linear" #sys.argv[4]
    computation_method = "fancy" #sys.argv[6]
    standarization = True

    n_explanations = 3
    ind_for_exp=0
    objetive_exp=np.array([0.0,  0.0])

    ce = ContrastingExplanation(model_desc, computation_method, n_explanations)
    ce.create_projection(dataset_desc, standarization)
    #ce.convert_to_json()
    ce.generate_explanation(ind_for_exp, objetive_exp)
    ce.exp_node.destandardized_data()

    #print(ce.exp_node.Y_cf_pred)
