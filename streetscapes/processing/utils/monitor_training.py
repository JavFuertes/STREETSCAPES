import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------ Monitoring VAE ------------------------------ #
def training_losses(vloss, tloss):
    vloss_cpu = [tensor.cpu() for tensor in vloss]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=tloss, mode='lines', name='training loss'))
    fig.add_trace(go.Scatter(y=vloss_cpu, mode='lines', name='validation loss'))
    fig.update_layout(title='VAE Training and Validation Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template = "plotly_white")

    return fig.show()

def lf_space(features, latd):
        
    z_cpu = features.detach().cpu().numpy()
    fig = go.Figure(data=[go.Scatter3d(
        x=z_cpu[:, latd[0]],  # X axis data
        y=z_cpu[:, latd[1]],  # Y axis data
        z=z_cpu[:, latd[2]],  # Z axis data
        mode='markers',
        marker=dict(
            size=5,  
            opacity=0.8,  
        )
    )])

    fig.update_layout(
        title_text = "Image feature distribution in latent space",
        margin=dict(l=0, r=0, b=0, t=0),  
        scene=dict(
            xaxis_title=f'Dimension {latd[0]}', 
            yaxis_title=f'Dimension {latd[1]}', 
            zaxis_title=f'Dimension {latd[2]}'
        ),
        template = 'plotly_dark'
    )
    return fig.show()

# ------------------------ Monitoring Gaussian Mixture ----------------------- #
def gm_elbo(aic_values: list, bic_values: list, log_likelihood_values: list):
    k_values = list(range(1, len(aic_values) + 1))
    
    trace_aic = go.Scatter(x=k_values, y=aic_values, mode='lines+markers', name='AIC')
    trace_bic = go.Scatter(x=k_values, y=bic_values, mode='lines+markers', name='BIC')
    trace_log_likelihood = go.Scatter(x=k_values, y=log_likelihood_values, mode='lines+markers', name='Log Likelihood')

    fig = make_subplots(rows=1, cols=2, subplot_titles=('AIC & BIC Scores per K', 'Log Likelihood per K'))

    fig.add_trace(trace_aic, row=1, col=1)
    fig.add_trace(trace_bic, row=1, col=1)
    fig.add_trace(trace_log_likelihood, row=1, col=2)

    fig.update_xaxes(title_text='Number of clusters (K)', row=1, col=1)
    fig.update_xaxes(title_text='Number of clusters (K)', row=1, col=2)
    fig.update_yaxes(title_text='Score', row=1, col=1)
    fig.update_yaxes(title_text='Log Likelihood', row=1, col=2)

    fig.update_layout(height=600, width=1200, title_text='GMM Model Criteria per Number of Clusters', showlegend=True)

    fig.show()


