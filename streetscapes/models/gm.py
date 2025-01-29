import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture

import plotly.graph_objects as go

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GM:
    def __init__(self, max_iter: int =1000, tol: float =1e-6):
        """
        Initializes a Gaussian Mixture Model (GMM).

        ## Parameters:
        - K: int
            The number of mixture components.
        - max_iter: int, optional (default=1000)
            The maximum number of iterations for the GMM algorithm.
        - tol: float, optional (default=1e-6)
            The tolerance for convergence of the GMM algorithm.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.gmm = None
        self.K = None
        self.centroids = None
        self.covariances = None
        self.confidence_regions = None

    def avg_gradient_change(self, values, current_index, window_size):
        gradients = [(values[i] - values[i-1]) / abs(values[i-1]) if values[i-1] != 0 else 0
                    for i in range(current_index - window_size + 1, current_index + 1)]
        avg_grad = sum(gradients) / window_size
        return avg_grad

    def find_optimal_clusters(self, data: torch.tensor, dgrad: float, max_k: int, window_size: int =3) -> dict:
        """
        Finds the optimal number of clusters using Gaussian Mixture Models (GMM).

        ## Args:
            data (array-like): The input data used for clustering.
            dgrad (float): The threshold for the average gradient change of AIC, BIC, and log likelihood.
            max_k (int): The maximum number of clusters to consider.
            window_size (int, optional): The size of the sliding window used to calculate average gradient change. Defaults to 3.

        ## Returns:
            dict: A dictionary containing the following keys:
                - 'optimal_k' (int): The optimal number of clusters.
                - 'aic_values' (list): A list of AIC values for each number of clusters.
                - 'bic_values' (list): A list of BIC values for each number of clusters.
                - 'log_likelihood_values' (list): A list of log likelihood values for each number of clusters.
        """
        aics = []
        bics = []
        ll = []
        k_values = list(range(1, max_k + 1))
        optimal_k = None

        for index, ki in enumerate(k_values):
            self.gmm = GaussianMixture(n_components=ki, covariance_type='full', 
                                    max_iter=self.max_iter, tol=self.tol, init_params='kmeans')
            self.gmm.fit(data)
            
            aics.append(self.gmm.aic(data))
            bics.append(self.gmm.bic(data))
            ll.append(self.gmm.score(data) * len(data))

            if index >= window_size:
                aic_avg_grad = self.avg_gradient_change(aics, index, window_size)
                bic_avg_grad = self.avg_gradient_change(bics, index, window_size)
                
                if (aic_avg_grad < dgrad or bic_avg_grad < dgrad):
                    optimal_k = k_values[index - window_size // 2]  # Middle of  window
                    break 

        return optimal_k, {
            'aic_values': aics,
            'bic_values': bics,
            'log_likelihood_values': ll
        }

    def fit(self, X: torch.tensor, k: int, optimise = False, dgrad: float = None,):
        """
        Fits the Gaussian Mixture Model to the given data.

        ## Parameters:
        - X: torch.tensor, shape (n_samples, n_features)
            The input data.
        - k: int
            In optimisation mode -> The maximum number of clusters to consider.
            In standard mode -> The number of clusters to fit.
        - dgrad: float, optional (default=70)
            The gradient threshold for determining the optimal number of clusters.

        ## Returns:
        None
        """
        if optimise:
            optimal, dict_ = self.find_optimal_clusters(X, dgrad, k)
            self.K = optimal
            self.gmm = GaussianMixture(n_components=self.K, covariance_type='full', max_iter=self.max_iter, tol=self.tol, init_params='kmeans')
            self.gmm.fit(X)
            return dict_
        else:
            self.K = k
            self.gmm = GaussianMixture(n_components=self.K, covariance_type='full', max_iter=self.max_iter, tol=self.tol, init_params='kmeans')
            self.gmm.fit(X)
        
    def sample(self, n_samples: int):
        """
        Generates samples from the fitted Gaussian Mixture Model.

        ## Parameters:
        - n_samples: int
            The number of samples to generate.

        ## Returns:
        - samples: array-like, shape (n_samples, n_features)
            The generated samples.
        """
        return self.gmm.sample(n_samples)

    def predict(self, X: torch.tensor):
        """
        Computes the posterior probabilities of each sample belonging to each component.

        ## Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data.

        ## Returns:
        - z_nk: array-like, shape (n_samples, K)
            The posterior probabilities of each sample belonging to each component.
        """
        return self.gmm.predict_proba(X)