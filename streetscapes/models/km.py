import torch
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import plotly.graph_objects as go

class KM:
    def __init__(self, max_iter: int = 10000, tol: float = 1e-20, n_init: int = 10):
        """
        Initializes a K-means clustering model.

        Parameters:
        - max_iter: int, optional (default=300)
            Maximum number of iterations for the K-means algorithm.
        - tol: float, optional (default=1e-4)
            Tolerance for declaring convergence.
        - n_init: int, optional (default=10)
            Number of time the k-means algorithm will be run with different centroid seeds.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.kmeans = None
        self.K = None
        self.centroids = None

    def fit(self, X: torch.Tensor, k: int):
        """
        Fits the K-means model to the given data.

        Parameters:
        - X: torch.Tensor
            Input data.
        - k: int
            Number of clusters to fit.

        Returns:
        - None
        """
        self.K = k
        self.kmeans = KMeans(
            n_clusters=self.K, max_iter=self.max_iter, tol=self.tol, n_init=self.n_init
        )
        self.kmeans.fit(X)
        self.centroids = self.kmeans.cluster_centers_

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts cluster assignments and returns soft probabilities based on distance.

        Parameters:
        - X: torch.Tensor
            Input data.

        Returns:
        - torch.Tensor: Probability-like scores for each cluster based on distance.
        """
        distances = cdist(X, self.centroids)
        exp_neg_dist = np.exp(-distances)
        probabilities = exp_neg_dist / exp_neg_dist.sum(axis=1, keepdims=True)
        return torch.from_numpy(probabilities)

    def get_cluster_assignments(self, X: torch.Tensor) -> torch.Tensor:
        """
        Gets hard cluster assignments for input data.

        Parameters:
        - X: torch.Tensor
            Input data.

        Returns:
        - torch.Tensor: Cluster assignments.
        """
        return torch.from_numpy(self.kmeans.predict(X))

    def calculate_confidence_regions(self, X: torch.Tensor, confidence_interval: float = 0.99):
        """
        Calculates confidence regions for each cluster based on point distances.

        Parameters:
        - X: torch.Tensor
            Input data used to calculate confidence regions.
        - confidence_interval: float
            Confidence level (default: 0.95).
        Returns:
        - dict: Confidence radii for each cluster.
        """
        labels = self.kmeans.predict(X)
        confidence_regions = {}

        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - self.centroids[k], axis=1)
                radius = np.percentile(distances, confidence_interval * 100)
                confidence_regions[k] = radius
            else:
                confidence_regions[k] = 0.0

        return confidence_regions
    
    def calculate_elbo_score(self, X: torch.Tensor, k: int) -> float:
       """
       Calculates an ELBO-like score for a given k.
       
       Parameters:
       - X: torch.Tensor - Input data
       - k: int - Number of clusters
       
       Returns:
       - float: Score (higher is better)
       """
       # Fit K-means for this k
       kmeans = KMeans(n_clusters=k, max_iter=self.max_iter, 
                      tol=self.tol, n_init=self.n_init)
       kmeans.fit(X)
       
       # Calculate within-cluster variance (reconstruction term)
       distances = cdist(X, kmeans.cluster_centers_)
       min_distances = np.min(distances, axis=1)
       elbo_score = -np.sum(min_distances**2)
       
       return elbo_score
    
    def find_optimal_k(self, X: torch.Tensor, k_range: range, threshold: float = 100) -> int:
       """
       Finds optimal number of clusters using ELBO-like criterion.
       
       Parameters:
       - X: torch.Tensor - Input data
       - k_range: range - Range of k values to try
       - Threshold: float - Derivative value to determine the plateauing in divergence between scores
       
       Returns:
       - int: Optimal number of clusters
       """
       scores = []
       for k in k_range:
           score = self.calculate_elbo_score(X, k)
           scores.append(score)
       
       self.scores = np.array(scores)
       
       derivatives = np.diff(scores)
       der_diff = derivatives - derivatives.max()/ threshold
       flat_indices = np.where(der_diff < 0)[0]
       
       optimal_index = flat_indices[0] if len(flat_indices) > 0 else np.argmax(scores)
       self.optimal_k = optimal_index + 1
       
       return self.optimal_k
    

    def plot_elbo_scores(self, k_range: range):
       """
       Plots ELBO scores vs number of clusters.
       
       Parameters:
       - k_range: range - Range of k values tried
       """
       if self.scores is None:
           raise ValueError("Run find_optimal_k first")
           
       fig = go.Figure()
       fig.add_trace(go.Scatter(x=list(k_range), y=self.scores, mode='lines+markers', name='ELBO Score', marker=dict(color='blue')))
       fig.add_vline(x=self.optimal_k, line_color='red', line_dash='dash', 
                     annotation_text=f'Optimal k={self.optimal_k}', annotation_position='top right')
       fig.update_layout(title='Model Selection Criterion vs Number of Clusters',
                         xaxis_title='Number of Clusters (k)',
                         yaxis_title='ELBO-like Score',
                         template='plotly_white')
       fig.show()
