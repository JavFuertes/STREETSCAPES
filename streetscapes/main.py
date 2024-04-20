import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from streetscapes.models import CNN, VAE, GaussianMixture
from streetscapes.processing import processing


# Import/process images
directory = '_data/imagedb'
n_images = 150

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to your desired dimensions
    transforms.ToTensor(),               # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image if necessary
])

labels = []
dataset = []
listdir = os.listdir(directory)
for i in range(n_images):
    filename = listdir[i]
    if filename.endswith('.png'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as image:
            image = Image.open(file_path)
            image_tensor = transform(image)
            dataset.append(image_tensor.to(device))
            labels.append(filename)

## Part 1: Image feature extraction through CNN
# Instantiate model and extract features
STREETSCAPES01 = CNN()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
features = STREETSCAPES01.extract_features(dataloader)

# Save features and 'labels'
dir = '_data\\sc_data'
file_name = 'DelftSV_imgfeatuers.h5'
path = os.path.join(dir,file_name )
processing.features(dataset, path, mode = 'save')

## Part 2: Image feature dimensionality reduction through VAE
# Prepare dataset for NN
dset = list(zip(features, labels))

data_test, data_train = random_split(
    dset, [test_rat, 1 - test_rat], generator=g
)
data_train, data_val = random_split(
    list(data_train), [validation_rat, 1 - validation_rat], generator=g
)

test_loader = DataLoader(
    data_test, batch_size=batch_size, shuffle=True, generator=g
)
training_loader = DataLoader(
    data_train, batch_size=batch_size, shuffle=True, generator=g
)
validation_loader = DataLoader(
    data_val, batch_size=batch_size, shuffle=True, generator=g
)

# Define VAE model
torch.manual_seed(0) # Set rng for reproducability
g = torch.Generator()
g.manual_seed(0)

# Set params for dataset handling
batch_size = 50
test_rat = 0.8
validation_rat = 0.7

# Model parameters
latent_dim = 3

kld_weight = 5e-04
hidden_dim = 100
hidden_num = 5
epochs = 250
save_model = True

#Early stopping param
patience = 25  # Number of epochs to wait for improvement before stopping
delta = 0.001  # Minimum change to signify an improvement
wait = 0  # Counter for epochs waited since last improvement

# create the model
model = VAE(
    input_dim= features.shape[1],
    latent_dim= latent_dim,
    hidden_dim= hidden_dim,
    hidden_num= hidden_num,
    beta= kld_weight,
    save_model = save_model
)

# Train the model
optimizer = torch.optim.Adam(model.parameters())
loaders = [training_loader, validation_loader]

tloss, vloss = model.train_(optimizer,loaders,epochs,patience,wait)

monitor_VAE = False
if monitor_VAE:
    from streetscapes.processing.utils import monitor_training, lf_space
    monitor_training(vloss,tloss)
    lf_space(features, [0,1,2])





monitor_GM = False
if monitor_GM:
    from streetscapes.processing.utils import gmc_space 
    gmc_space(data, [0,1,2], gmm)
