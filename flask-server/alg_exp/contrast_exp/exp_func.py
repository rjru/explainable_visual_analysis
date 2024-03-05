import numpy as np

from autoencoder import AutoencoderModel, AutoencoderDimRed
from utils import load_data
from linear_dr import LinearDimRed, LinearDimRedCf
from ae_dr import AutoEncoderDimRedCf
from som_dr import SomDimRedCf, SomDimRed
from tsne_dr import TsneDimRed, TsneDimRedCf
from memory_counterfactual import MemoryCounterfactual

def run_projection_data(dataset_desc, model_desc):
    # Load data
    data_orig, X, y = load_data(dataset_desc, scaling=True) # desactibamos el scaling para evaluar los resultados de iris. 
    n_dim = X.shape[1]
    
    # Fit and apply dimensionality reduction method
    if model_desc == "linear":
        model = LinearDimRed()
    elif model_desc == "ae":
        model = AutoencoderDimRed(AutoencoderModel(features=[10, 2], input_dim=X.shape[1]))
    elif model_desc == "som":
        model = SomDimRed()
    elif model_desc == "tsne":
        model = TsneDimRed()
    else:
        raise ValueError("Unknown model")

    X_red = model.fit_transform(X)   
    return data_orig, X, y, X_red, model

def compute_count_exp(X, y, X_red, model_desc, model, ind_for_exp, objetive_exp, computation_method, n_explanations):
    # Compute counterfactual explanation of dimensionality reduction
    if computation_method == "fancy":
        if model_desc == "linear":
            explainer = LinearDimRedCf(model, C_pred=10.)
        elif model_desc == "ae":
            explainer = AutoEncoderDimRedCf(model)
        elif model_desc == "som":
            explainer = SomDimRedCf(model)
        elif model_desc == "tsne":
            explainer = TsneDimRedCf(model)
    else:
        explainer = MemoryCounterfactual(X, X_red)

    explanations = []
    transformed_samples_dist = []
    cf_transformed_sanmples_error = []
    raw_results = {"X_orig": [], "X_transformed": [], "X_perturbed": [], "X_cf": [], "Y_cf": [], "X_cf_transformed": [], "class": []}
    
    i=int(ind_for_exp) # el indice del elemento que queremos explicar. 
    x_orig = X[i,:] # array original de características (4 caracteristicas de iris)
    y_orig = X_red[i,:] # array original de datos proyectados (dos datos proyectados x, y)
    y_cf =  objetive_exp #X_perturbed_red[i,:] # array de datos proyectados "perturbados" (dos datos proyectados x, y)

    transformed_dist = np.linalg.norm(y_orig - y_cf, 2) # se calcula la distancia entre los datos "perturbados" y los la proyección original.

    X_cf_ = explainer.compute_diverse_explanations(x_orig, y_cf, n_explanations=n_explanations)
    if X_cf_ is None:
        print("Al parecer no se pudo encontar una explicabilidad")
    Y_cf_pred = [model.transform(x_cf) for x_cf in X_cf_] # se proyectan las explicaciones. 

    cf_transformed_error = [np.linalg.norm(y_cf - y_cf_pred, 2) for y_cf_pred in Y_cf_pred] # resta de las 3 proyecciones en relación a la proyección de los datos perturbados

    Delta_cf = [np.abs(x_orig - x_cf) for x_cf in X_cf_]

    explanations.append(Delta_cf)
    transformed_samples_dist.append(transformed_dist)
    cf_transformed_sanmples_error.append(cf_transformed_error)

    raw_results["X_orig"].append(x_orig) # vector original
    raw_results["X_transformed"].append(y_orig) # vector original proyectado
    raw_results["Y_cf"].append(y_cf) # vector distorcionado proyectado
    raw_results["X_cf"].append(X_cf_) # explicaciones (tres)
    raw_results["X_cf_transformed"].append(Y_cf_pred) # a partir de las explicaciones, se proyectaron los datos. 
    raw_results["class"].append(y[i]) 

    #print(explanations)

    return raw_results


#'''

if __name__ == "__main__":

    dataset_desc = "iris" #sys.argv[1]
    model_desc = "linear" #sys.argv[4]
    raw_results_out = "exp_results/" #sys.argv[5]
    computation_method = "fancy" #sys.argv[6]
    standarization = 'yes'

    print(dataset_desc,model_desc,raw_results_out,computation_method)


    data_orig, X, y, X_red, model = run_projection_data(dataset_desc, model_desc)

    n_explanations = 3
    ind_for_exp=0
    objetive_exp=np.array([0.0,  0.0])

    raw_results = compute_count_exp(X, y, X_red, model_desc, model, ind_for_exp, objetive_exp, computation_method, n_explanations)

    print(raw_results)
                  
#'''