import torch
from sklearn.mixture import GaussianMixture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianMixture:
    def __init__(self, K: int, max_iter: int =1000, tol: float =1e-6):
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
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.gmm = None

    def fit(self, X: torch.tensor):
        """
        Fits the Gaussian Mixture Model to the given data.

        ## Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data.
        """
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
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.sample(n_samples)

    def predict_proba(self, X: torch.tensor):
        """
        Computes the posterior probabilities of each sample belonging to each component.

        ## Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data.

        ## Returns:
        - z_nk: array-like, shape (n_samples, K)
            The posterior probabilities of each sample belonging to each component.
        """
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.predict_proba(X)

    @property
    def means_(self):
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.means_

    @property
    def covariances_(self):
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.covariances_

    @property
    def weights_(self):
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.weights_

    def score(self, X: torch.tensor):
        """
        Computes the log-likelihood of the fitted GMM model.

        ## Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data.

        ## Returns:
        - loglikelihood: float
            The log-likelihood of the fitted GMM model.
        """
        if self.gmm is None:
            raise ValueError("GMM has not been fitted yet. Please call the fit method first.")
        return self.gmm.score(X) * len(X)  # gmm.score returns average log likelihood per sample
