import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston, load_diabetes, load_digits


non_zero_threshold = 1e-5


def scale_standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def create_toy_data(n_samples=500, n_dim=10):
    X = []
    y = []

    # Random centers [0,100) of Gaussian distributions
    centers = 100. * np.random.random(n_dim)

    # Generate data by sampling from a different Gaussian distribution for every dimension
    for mu in centers:
        X.append(np.random.normal(mu, size=n_samples))
    X = np.array(X).T

    # Cluster data into two groups -- labeling according to cluster assignment
    model = KMeans(n_clusters=2)
    model.fit(X)
    y = model.predict(X)

    return X, y

def load_data(data_desc):
    if data_desc == "iris":
        data_orig = load_iris(return_X_y=False)
    elif data_desc == "toy":
        data_orig = create_toy_data() # es una dataset sintética, por el momento queda sin utilizar.
    elif data_desc == "breastcancer":
        data_orig = load_breast_cancer(return_X_y=False)
    elif data_desc == "wine":
        data_orig = load_wine(return_X_y=False)
    elif data_desc == "boston":
        data_orig = load_boston(return_X_y=False)
    elif data_desc == "diabetes":
        data_orig = load_diabetes(return_X_y=False)
    elif data_desc == "digits":
        data_orig = load_digits(return_X_y=False)
    else:
        raise ValueError(f"Unknown data set '{data_desc}'")
    
    X, y = data_orig.data, data_orig.target
        
    return data_orig, X, y

def standardize_data(X):
    X, stand_model = scale_standardize_data(X)
    return X, stand_model