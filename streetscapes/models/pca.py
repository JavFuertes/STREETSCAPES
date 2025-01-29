import numpy as np
import plotly.graph_objects as go

class PCA:
    def __init__(self, n_components: int):
        """
        Principal Component Analysis (PCA) model.

        Args:
            n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the data.

        Args:
            X (np.ndarray): The input data of shape [n_samples, n_features].
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance

        self.components_ = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data to the PCA space.

        Args:
            X (np.ndarray): The input data of shape [n_samples, n_features].

        Returns:
            np.ndarray: The transformed data of shape [n_samples, n_components].
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the PCA model and transform the data.

        Args:
            X (np.ndarray): The input data of shape [n_samples, n_features].

        Returns:
            np.ndarray: The transformed data of shape [n_samples, n_components].
        """
        self.fit(X)
        return self.transform(X)

    def find_optimal_components(self, data: np.ndarray, max_components: int = None, threshold: float = 0.95):
        """
        Determines the optimal number of PCA components using explained variance ratio
        and plots the cumulative explained variance.

        Parameters:
            data (np.ndarray): Input data.
            max_components (int): Maximum number of components to consider.
            threshold (float): Minimum cumulative explained variance ratio (default 0.95).

        Returns:
            int: Optimal number of components.
            np.ndarray: Array of explained variance ratios.
        """
        n_samples, n_features = data.shape
        max_components = min(n_samples, n_features) if max_components is None else max_components

        self.n_components = max_components
        self.fit(data)

        cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio_)
        optimal_n = np.argmax(cumulative_variance_ratio >= threshold) + 1

        self.plot_explained_variance(cumulative_variance_ratio, optimal_n, threshold)

        return optimal_n, self.explained_variance_ratio_

    def plot_explained_variance(self, cumulative_variance_ratio, optimal_n, threshold):
        """
        Plots the cumulative explained variance using Plotly.

        Parameters:
            cumulative_variance_ratio (np.ndarray): Array of cumulative explained variance ratios.
            optimal_n (int): Optimal number of components.
            threshold (float): Minimum cumulative explained variance ratio.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(cumulative_variance_ratio) + 1),
                                 y=cumulative_variance_ratio,
                                 mode='lines+markers',
                                 name='Cumulative Explained Variance',
                                 marker=dict(color='blue')))
        fig.add_hline(y=threshold, line_color='red', line_dash='dash',
                      annotation_text=f'{threshold*100:.1f}% Variance Threshold',
                      annotation_position='top right')
        fig.add_vline(x=optimal_n, line_color='green', line_dash='dash',
                      annotation_text=f'Optimal Components: {optimal_n}',
                      annotation_position='top right')
        fig.update_layout(title="Cumulative Explained Variance",
                          xaxis_title="Number of Components",
                          yaxis_title="Cumulative Variance Ratio",
                          template="plotly_white")
        fig.show()
