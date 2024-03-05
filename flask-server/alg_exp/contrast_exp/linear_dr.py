import numpy as np
import cvxpy as cp
from sklearn.decomposition import PCA

from alg_exp.contrast_exp.utils import non_zero_threshold
from alg_exp.contrast_exp.dim_red import DimRed
from alg_exp.contrast_exp.counterfactual import CounterfactualExplanation


default_solver = cp.SCS

class LinearDimRed(DimRed):
    def __init__(self, **kwds):
        self.model = PCA(n_components=2)

        super().__init__(**kwds)
    
    def fit(self, X):
        self.model.fit(X)

    def transform(self, X):
        if len(X.shape) == 1:
            return self.model.transform(X.reshape(1, -1))
        else:
            return self.model.transform(X)


class LinearDimRedCf(CounterfactualExplanation):
    def __init__(self, model, C_pred=1., **kwds):
        if not isinstance(model, LinearDimRed):
            raise TypeError(f"'model' must be an instance of 'LinearDimRed' not of {type(model)}")

        self.A = model.model.components_ # varlores específicos de pca
        self.b = np.zeros(self.A.shape[0])
        self.C_pred = C_pred
        self.solver = default_solver
        self.solver_verbosity = False

        super().__init__(**kwds)

    def __compute_counterfactual(self, x_orig, y_cf, features_whitelist=None): # ejecuta la explicación contractual
        try:
            dim = x_orig.shape[0] 
            if features_whitelist is None: # si es None, entonces features_whitelist se completará con los indices tal es la dimensión, en este caso, [0, 1, 2, 3]
                features_whitelist = list(range(dim))

            x = cp.Variable(dim)
            xi = cp.Variable()

            constraints = [xi >= 0] # Requested output
            constraints += [cp.norm(self.A @ x + self.b - y_cf, 2) <= xi]

            A = []  # Some features must not change!, con el fin de que haya diversidad. 
            a = []
            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
                    
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)
                constraints += [A @ x == a]

            f = cp.Minimize(cp.norm(x_orig - x, 1) + self.C_pred * xi)
            prob = cp.Problem(f, constraints)

            prob.solve(solver=self.solver, verbose=self.solver_verbosity)

            return x.value
        except Exception as ex:
            print(ex)
            return None

    def _compute_diverse_counterfactual(self, x_orig, y_cf, X_cf):
        features_whitelist = [] # Diversity: Already used features must not be used again
                                # las caracteristicas usadas no deben volverse a usar
        for x_cf in X_cf:
            delta_cf = np.abs(x_orig - x_cf) # la diferencia entre el vector original y la primera explicación contractual
            features_whitelist += [int(idx) for idx in np.argwhere(delta_cf > non_zero_threshold)] # argwhere toma todos los valores de una matriz que sean mayores que cero. 
            # Si la diferencia es cero en delta_cf, significa que no ubo variación por lo cual esa característica no se usó. 
            # Si la diferencia es mayor que cero, significa que la caracteristica ha sido modificada por lo tanto se usó. 
        features_whitelist = list(set(list(range(x_orig.shape[0]))) - set(features_whitelist))

        return self.__compute_counterfactual(x_orig, y_cf, features_whitelist)

    def compute_explanation(self, x_orig, y_cf):
        return self.__compute_counterfactual(x_orig, y_cf)
