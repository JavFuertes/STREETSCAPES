import os

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional
from tabulate import tabulate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, hidden_num: int, beta: float =0.001, save_model: bool = False):
        """
        Variational Autoencoder (VAE) model.

        ## Args:
            input_dim (int): Dimensionality of the input data.
            latent_dim (int): Dimensionality of the latent space.
            hidden_dim (int): Dimensionality of the hidden layers.
            hidden_num (int): Number of hidden layers.
            beta (float, optional): Beta parameter for the Kullback-Leibler divergence term in the loss function. Defaults to 0.001.
            save_model (bool, optional): Whether to save the trained model. Defaults to False.
        """
        super(VAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.beta = beta
        self.latent_dim = latent_dim

        if save_model:
            model_path = os.path.join(
                os.getcwd(), "_data\\models\\vae", f"lat-{latent_dim}"
            )
            os.makedirs(model_path, exist_ok=True)
            self.save_model = True
            self.model_path = model_path

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

        Args:
            x (torch.Tensor): A set of input features [B x D]

        Returns:
            mu (torch.Tensor): Vector with posterior means of q(z) [B x M]
            log_var (torch.Tensor): Vector with posterior covariances of q(z) [B x M]
        """
        result = self.encoder(x)
        mu = self.hidtomu(result)
        log_var = self.hidtovar(result)
        return mu, log_var

    def decode(self, z):
        """
        Decodes data from latent space z to real space x

        Args:
            z: Samples of q(z) of [B x M]

        Returns:
            result: Decoded reconstructions (means of p(x|z)) [B x D]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        """
        Performs the 'reparametrization trick' to draw a sample from q(z) while
        actually sampling N(0,I). The variance is exponentiated to learn a log-scaled
        version of it. This enforces non-negativity and ensures q(z) is a valid Gaussian.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian q(z) [B x M]
            logvar (torch.Tensor): Log of (diagonal) variance of the latent Gaussian q(z) [B x M]

        Returns:
            z (torch.Tensor): One sample of q(z) for each minibatch point [B x M]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        """
        Performs a complete forward pass through the autoencoder

        Args:
            x (torch.Tensor): A set of input features [B x D]

        Returns:
            xtilde (torch.Tensor): A set of reconstructed inputs [B x D]
            mu (torch.Tensor): The means of the posteriors q(z) [B x M]
            log_var (torch.Tensor): The (naturally log-scaled) variances of the posteriors q(z) [B x M]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        xtilde = self.decode(z)
        return xtilde, mu, log_var

    def encoparam(self, x):
        """
        Perform half a forward pass of the variational autoencoder to obtain an encoded latent space.

        Args:
            x: Input data to be encoded.

        Returns:
            z: Encoded latent space representation.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z

    def loss_function(self, x, xtilde, mu, log_var):
        """
        Computes the loss function of the VAE

        Args:
            x (torch.Tensor): The original input data [B x D]
            xtilde (torch.Tensor): The reconstructed input data [B x D]
            mu (torch.Tensor): The means of the posteriors q(z) [B x M]
            log_var (torch.Tensor): The (naturally log-scaled) variances of the posteriors q(z) [B x M]

        Returns:
            dict: A dictionary containing the loss values
                - loss (torch.Tensor): The total loss
                - recon (torch.Tensor): The reconstruction loss
                - kld (torch.Tensor): The Kullback-Leibler divergence loss
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

    def train_one_epoch(self, loader: object, optimizer: object):  
        """
        Trains the model for one epoch using the given data loader and optimizer.

        Args:
            model (nn.Module): The model to be trained.
            loader (DataLoader): The data loader containing the training data.
            optimizer (Optimizer): The optimizer used for updating the model's parameters.

        Returns:
            float: The average loss per batch for the epoch.
        """

        self.train()
        running_loss = 0.0
        last_loss = 0.0
        n_batch = loader.batch_size
        n_data = len(loader.dataset)

        write_every = max(int(n_data / n_batch / 10), 1)

        for i, [data, _] in enumerate(loader):
            
            x = data.to(self.device)   
            optimizer.zero_grad()

            xtilde, mu, log_var = self(x)

            output = self.loss_function(x, xtilde, mu, log_var)
            loss = output["loss"]
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % write_every == write_every - 1:
                last_loss = running_loss / write_every  # loss per batch
                # print(
                #     f"loss: {last_loss:5f}, "
                #     f"reconstruction loss: {output['recon']:5f}, "
                #     f"kld loss: {output['kld']:5f}"
                # )
                running_loss = 0.0

        return last_loss

    def train_(self, optimizer: object, loader: object, epochs: int, patience: int, wait: int):
        """
        Train the model using the given optimizer and data loader for a specified number of epochs.

        ## Args:
            optimizer (object): The optimizer used for training.
            loader (object): The data loader used for training and validation.
            epochs (int): The number of epochs to train the model.
            patience (int): The number of epochs to wait for improvement in validation loss before early stopping.
            wait (int): The current number of epochs without improvement in validation loss.

        ## Returns:
            None
        """
        train_loss = []
        val_loss = []
        
        epoch_number = 0
        best_vloss = float('inf')
        best_epoch = float('inf')
        best_loss = None

        for epoch in range(epochs):

            self.train(True)
            avg_loss = self.train_one_epoch(loader[0],optimizer)

            running_vloss = 0.0
            self.eval()

            with torch.no_grad():
                for i, [x, _] in enumerate(loader[1]):
                    x = x.to(self.device)
                    xtilde, mu, log_var = self(x)
                    vloss = self.loss_function(x, xtilde, mu, log_var)["loss"]
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            train_loss.append(avg_loss)
            val_loss.append(avg_vloss)

            # Track best validation performance with early stopping, optionally save the model with lowest loss
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                wait = 0  # Reset wait counter after improvement
                best_loss = avg_loss
                best_epoch = epoch
                if self.save_model:
                    torch.save(self.state_dict(), os.path.join(self.model_path, "model_best.pt"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break  # Stop training if wait exceeds patience

            epoch_number += 1
            last_values = [epoch, avg_loss, float(avg_vloss), best_loss, best_epoch]
            print(tabulate([last_values], headers=['Epoch', 'Train Loss', 'Val Loss', 'Best Loss', 'Best Epoch'], tablefmt='grid'))

        return train_loss, val_loss