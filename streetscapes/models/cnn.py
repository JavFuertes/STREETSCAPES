import torch
import torch.nn as nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    """
    Implementation of CNN of choice.

    ## Args:
        model (str): The name of the model architecture to use. Default is 'resnet152'.
        train (bool): Whether to train the model or not. Default is True.
    """
    def __init__(self, model= models.resnet152, train=True):
        super(CNN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = model(pretrained=train).to(device)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, dataloader: object) -> torch.Tensor:
        """
        Extracts features from the given dataloader using the model.

        ## Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.

        ## Returns:
            torch.Tensor: The extracted features as a tensor.
        """
        features = []
        self.eval()  # Put the model in evaluation mode
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computations
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                features.append(outputs)
        features = torch.cat(features, dim=0)  # Concatenate tensors along dimension 0
        return features