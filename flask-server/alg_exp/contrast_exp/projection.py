from alg_exp.contrast_exp.utils import load_data, standardize_data
from alg_exp.contrast_exp.linear_dr import LinearDimRed, LinearDimRedCf
#from ae_dr import AutoEncoderDimRedCf
from alg_exp.contrast_exp.som_dr import SomDimRedCf, SomDimRed
from flask_restful import Resource

# PROYECTO ORIGINAL: https://github.com/andreArtelt/ContrastingExplanationDimRed
# alg_exp.contrast_exp. SE DEBE AGREGAR A CADA from para que pueda acceder a las librerias cuando se usa flask.
# https://stackoverflow.com/questions/4534438/typeerror-module-object-is-not-callable
#from autoencoder import AutoencoderModel, AutoencoderDimRed

class Projection(Resource):
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

