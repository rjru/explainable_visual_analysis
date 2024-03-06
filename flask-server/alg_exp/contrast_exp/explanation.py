import numpy as np
import pandas as pd
#from tsne_dr import TsneDimRed, TsneDimRedCf
from alg_exp.contrast_exp.memory_counterfactual import MemoryCounterfactual
from alg_exp.contrast_exp.linear_dr import LinearDimRed, LinearDimRedCf
#from ae_dr import AutoEncoderDimRedCf
from alg_exp.contrast_exp.som_dr import SomDimRedCf, SomDimRed


# PROYECTO ORIGINAL: https://github.com/andreArtelt/ContrastingExplanationDimRed
# alg_exp.contrast_exp. SE DEBE AGREGAR A CADA from para que pueda acceder a las librerias cuando se usa flask.
# https://stackoverflow.com/questions/4534438/typeerror-module-object-is-not-callable
#from autoencoder import AutoencoderModel, AutoencoderDimRed


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
