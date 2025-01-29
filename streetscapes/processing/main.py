import os
import numpy as np
import h5py
import geopandas as gpd
import torch
from PIL import Image  
from torchvision import transforms  
from torch.utils.data import DataLoader  

def geojson(directory: str, filename: str) -> gpd.GeoDataFrame:
    """
    Process a GeoJSON file and return a GeoDataFrame.

    Parameters:
    directory (str): The directory where the GeoJSON file is located.
    filename (str): The name of the GeoJSON file.

    Returns:
    gpd.GeoDataFrame: The processed GeoDataFrame containing the geometries and attributes from the GeoJSON file.
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        gpd_df = gpd.read_file(file_path)
    return gpd_df

def handle_features(path: str, mode: str, dataset: torch.tensor = None, labels: torch.tensor = None):
    """
    Process and save or load features and labels.

    Args:
        path (str): The path to save or load the processed data.
        mode (str): The mode of operation. Can be 'save' or 'load'.
        dataset (torch.tensor, optional): The dataset to process. Required when mode is 'save'.
        labels (torch.tensor, optional): The labels to process. Required when mode is 'save'.

    Returns:
        None or tuple: Returns None when saving data. Returns a tuple of features and labels when loading data.
    """
    if mode == 'save':
        dataset = dataset.detach().cpu().numpy()
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('features', data=dataset)
            hf.create_dataset('labels', data=labels)
        return 'Data saved successfully!'
    if mode == 'load':
        with h5py.File(path, 'r') as hf:
            features = hf['features'][:]
            labels = hf['labels'][:]
            labels = [label.decode('utf-8') for label in labels]
        return features, labels
    
def process_image_batch(image_files, start_idx):
    """
    Process a batch of images and extract features.

    Args:
        image_files (list): List of image file names to process.
        start_idx (int): The starting index for the batch being processed.

    Returns:
        tuple: A tuple containing the extracted features as a tensor and the corresponding labels as a list.
    """
    print(f"\nProcessing batch starting at index {start_idx}...")
    temp_dataset = []
    temp_labels = []
    
    for filename in image_files:
        if filename.endswith('.png') and not '_s_' in filename:
            image = Image.open(os.path.join(directory, filename)).convert('RGB')
            temp_dataset.append(transform(image))
            temp_labels.append(filename)
        
    print(f"Found {len(temp_dataset)} valid images")    
    dataset = torch.stack(temp_dataset)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
    
    features = []
    model = VIT()
    model.eval()
    
    print("Extracting features...")
    with torch.no_grad():
        for batch in dataloader:
            batch_features = model.forward(batch)
            features.append(batch_features)
    
    return torch.cat(features, dim=0), temp_labels

def save_features_and_labels(features, labels, start_idx, mode='a'):
    """
    Save features and labels to disk.

    Args:
        features (torch.Tensor): The features to save.
        labels (list): The labels corresponding to the features.
        start_idx (int): The starting index for the batch being saved.
        mode (str): The mode for saving the features file ('w' for write, 'a' for append).

    Returns:
        None
    """
    print(f"Saving features and labels for batch starting at index {start_idx}...")
    with h5py.File(FEATURES_FILE, mode) as f:
        if start_idx == 0:
            f.create_dataset('features', data=features.numpy(), maxshape=(None, features.shape[1]))
        else:
            f['features'].resize((f['features'].shape[0] + features.shape[0]), axis=0)
            f['features'][-features.shape[0]:] = features.numpy()
    
    write_mode = 'w' if start_idx == 0 else 'a'
    with open(LABELS_FILE, write_mode) as f:
        for label in labels:
            f.write(f"{label}\n")
    print("Save complete")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))
