import numpy as np
import plotly.graph_objects as go
import plotly.express as px 

# --------------------------- Vis Gaussian Mixture --------------------------- #
def gmc_space(features, latd, gmm):
    """
    Visualizes the Gaussian Mixture Model clusters in 3D with simplified boundaries.
    
    Args:
    - features (torch.Tensor): The latent features from the VAE.
    - latd (list of int): The indices of the dimensions to plot.
    - gmm (GaussianMixture): The fitted Gaussian Mixture Model.
    """
    
    z_cpu = features
    cluster_labels = gmm.predict(z_cpu)
    
    # Define colors for each cluster
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    # Plot data points with colors based on cluster membership
    for i in range(gmm.n_components):
        cluster_data = z_cpu[cluster_labels == i]
        fig.add_trace(go.Scatter3d(
            x=cluster_data[:, latd[0]],
            y=cluster_data[:, latd[1]],
            z=cluster_data[:, latd[2]],
            mode='markers',
            marker=dict(size=3, color=colors[i % len(colors)]),
            name=f'Cluster {i+1}'
        ))

    # Add spheres to indicate cluster centers (simplified boundary visualization)
    for mean, color in zip(gmm.means_, colors[:gmm.n_components]):
        sphere = create_sphere(mean[latd[0]], mean[latd[1]], mean[latd[2]], radius=0.5, color=color)
        fig.add_trace(sphere)
    
    # Update plot layout
    fig.update_layout(
        title_text="GMM Clusters in Latent Space",
        scene=dict(
            xaxis_title=f'Dimension {latd[0]}',
            yaxis_title=f'Dimension {latd[1]}',
            zaxis_title=f'Dimension {latd[2]}'
        ),
        template='plotly_dark'
    )
    
    return fig.show()

def create_sphere(x_center, y_center, z_center, radius, color):
    """
    Generates a sphere surface centered at (x_center, y_center, z_center).
    
    Args:
    - x_center, y_center, z_center (float): Center of the sphere.
    - radius (float): Radius of the sphere.
    - color (str): Color of the sphere.
    
    Returns:
    - A Plotly figure object representing the sphere.
    """
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    
    x = radius * np.sin(theta) * np.cos(phi) + x_center
    y = radius * np.sin(theta) * np.sin(phi) + y_center
    z = radius * np.cos(theta) + z_center
    
    return go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        alphahull=0,
        opacity=0.2,
        color=color
    )