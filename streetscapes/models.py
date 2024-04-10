import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models 
from torchsummary import summary
from torch.nn import functional
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import torch
import torch.nn as nn
from torchvision import models

class ResNet152(nn.Module):
    """
    ResNet152 model implementation.

    This class defines the ResNet152 model architecture and initializes it with pretrained weights.
    """

    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, dataloader: object, device: object) -> torch.Tensor:
        """
        Extracts features from the given dataloader using the model.

        ## Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
            device (torch.device): The device to perform the computation on.

        ## Returns:
            torch.Tensor: The extracted features as a tensor.
        """
        features = []
        self.eval()  # Put the model in evaluation mode
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computations
            for inputs in dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                features.append(outputs)
        features = torch.cat(features, dim=0)  # Concatenate tensors along dimension 0
        return features

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, hidden_num: int, beta: float =0.001):
        """
        Initializes the layers that make up the two halves of the autoencoder.

        ## Receives:
        'input_dim': Data dimensionality in the original space (D)
        'latent_dim': Dimensionality of the latent space z (M)
        'hidden_dim': Number of units of each hidden layer
        'hidden_num': Number of hidden layers of each half of the model
        'beta': The (optional) weighing of the KL divergence loss term

        ## Stores:
        'self.encoder': The encoder model, ready to be called
        'self.decoder': The decoder model, ready to be called
        """

        super(VAE, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.beta = beta
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        ).to(self.device)

        self.hidtomu = nn.Linear(hidden_dim, latent_dim).to(self.device)
        self.hidtovar = nn.Linear(hidden_dim, latent_dim).to(self.device)

        # Decoder
        decoder = [
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU()
        ]

        decoder.append(nn.Linear(hidden_dim, input_dim))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder).to(self.device)

    def encode(self, x: torch.Tensor): 
        """
        Encodes data from real space x to a Gaussian approximation q(z) of p(z|x)

        ## Receives:
        'x': a set of input features [B x D]

        ## Returns:
        'mu': Vector with posterior means of q(z) [B x M]
        'log_var': Vector with posterior covariances of q(z) [B x M]
        """
        result = self.encoder(x)
        mu = self.hidtomu(result)
        log_var = self.hidtovar(result)
        return mu, log_var

    def decode(self, z):
        """
        Decodes data from latent space z to real space x

        Receives:
        'z': samples of q(z) of [B x M]

        Returns:
        'xtilde': decoded reconstructions (means of p(x|z)) [B x D]
        """

        result = self.decoder(z)

        return result

    def reparameterize(self, mu, logvar):
        """
        Performs the 'reparametrization trick' to draw a sample from q(z) while
        actually sampling N(0,I). The variance is exponentiated to learn a log-scaled
        version of it. This enforces non-negativity and ensures q(z) is a valid Gaussian.

        Receives:
        'mu': Mean of the latent Gaussian q(z) [B x M]
        'logvar': Log of (diagonal) variance of the latent Gaussian q(z) [B x M]

        Returns:
        'z': one sample of q(z) for each minibatch point [B x M]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        
        return z

    def forward(self, x):
        """
        Performs a complete forward pass through the autoencoder

        Receives:
        'x': A set of input features [B x D]

        Returns:
        'xtilde': A set of reconstructed inputs [B x D]
        'mu': The means of the posteriors q(z) [B x M]
        'log_var': The (naturally log-scaled) variances of the posteriors q(z) [B x M]
        """

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        xtilde = self.decode(z)
       
        return xtilde, mu, log_var

    def encoparam(self, x):
        """
        Perform half a forward pass of the variational autoencoder to obtain an encoded latent space.

        Parameters:
        - x: Input data to be encoded.

        Returns:
        - z: Encoded latent space representation.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return z

    def loss_function(self, x, xtilde, mu, log_var):
        """
        Computes the loss function of the VAE
        """

        reconstruction_loss = functional.mse_loss(xtilde, x)
        
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss + self.beta * kld_loss
        
        return {
            "loss": loss,
            "recon": reconstruction_loss.detach(),
            "kld": kld_loss.detach(),
        }

    @staticmethod
    def train_one_epoch(model, loader: object, optimizer: object):  
        """
        Trains the model for one epoch using the given data loader and optimizer.

        Args:
            model (nn.Module): The model to be trained.
            loader (DataLoader): The data loader containing the training data.
            optimizer (Optimizer): The optimizer used for updating the model's parameters.

        Returns:
            float: The average loss per batch for the epoch.
        """

        model.train()
        running_loss = 0.0
        last_loss = 0.0
        n_batch = loader.batch_size
        n_data = len(loader.dataset)

        write_every = max(int(n_data / n_batch / 10), 1)

        for i, [data, _] in enumerate(loader):
            
            x = data.to(model.device)  # Ensure data is on the correct device
            
            optimizer.zero_grad()

            xtilde, mu, log_var = model(x)

            output = model.loss_function(x, xtilde, mu, log_var)
            loss = output["loss"]
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % write_every == write_every - 1:
                last_loss = running_loss / write_every  # loss per batch
                print(
                    f"loss: {last_loss:5f}, "
                    f"reconstruction loss: {output['recon']:5f}, "
                    f"kld loss: {output['kld']:5f}"
                )
                running_loss = 0.0

        return last_loss

    @staticmethod
    def tain(model,optimizer,loader,epochs):

        epoch_number = 0

        for epoch in range(epochs):
            print("--------------------------------------------")
            print("EPOCH {}:".format(epoch_number + 1))

            model.train(True)
            avg_loss = model.train_one_epoch(model,loader[0],optimizer)

            running_vloss = 0.0
            model.eval()

            with torch.no_grad():
                for i, [x, _] in enumerate(loader[1]):
                    xtilde, mu, log_var = model(x)
                    vloss = model.loss_function(x, xtilde, mu, log_var)["loss"]
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss:5f} LOSS valid {avg_vloss:5f}")
            train_loss.append(avg_loss)
            val_loss.append(avg_vloss)

            # Track best validation performance with early stopping, optionally save the model with lowest loss
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                wait = 0  # Reset wait counter after improvement
                if save_model:
                    torch.save(model.state_dict(), os.path.join(model_path, "model_best.pt"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break  # Stop training if wait exceeds patience

            epoch_number += 1

    @staticmethod
    def plot_one_sample(model, loader):
        model.eval()
        idcs = np.random.permutation(loader.dataset.indices)[:1]
        targets = loader.dataset.dataset[idcs].to(model.device)
        recons = model.forward(targets)[0].detach().cpu().detach()

        # create figure and plot reconstructions
        fig, ax = plt.subplots(1, 2, figsize=(2 * 2.5, 2.5))

        ax[0].imshow(targets[0].reshape(2048, 2048))
        ax[1].imshow(recons[0].reshape(25, 25))

        # adapt layout
        [axs.set_axis_off() for axs in ax.flat]
        ax[0].set_title("target")
        ax[1].set_title("reconstruction")

        plt.show()