import sys
sys.dont_write_bytecode = True

import os
import random

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .processing import *

print("Setting up device and loading dependencies...")
device = torch.device('cpu')

directory = '_data\\geo_json\\panoids'
directory = '_data/imagedb'

BATCH_SIZE = 0  
FEATURES_FILE = '_data/processed/features.h5'
LABELS_FILE = '_data/processed/labels.txt'

gpd_df = {}

if __name__ == "__main__":

    print("Loading GeoJSON files...")
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.geojson'):
                gpd_df[filename] = geojson(root, filename)
    print(f"Loaded {len(gpd_df)} GeoJSON files")

    os.makedirs('_data/processed', exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("\nStarting image processing pipeline...")
    listdir = os.listdir(directory)
    random.shuffle(listdir)
    print(f"Found {len(listdir)} total files")
    
    for i in tqdm(range(0, len(listdir), BATCH_SIZE)):
        batch_files = listdir[i:i+BATCH_SIZE]
        features, labels = process_image_batch(batch_files, i)
        
        if features is not None:
            save_features_and_labels(features, labels, i, 'w' if i == 0 else 'a')
    
    process_all_images()