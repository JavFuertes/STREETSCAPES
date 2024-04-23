import os
import h5py
import geopandas as gpd

def geojson(directory: str, filename: str) -> gpd.GeoDataFrame:
    """
    Process a GeoJSON file and return a GeoDataFrame.

    ## Parameters:
    directory (str): The directory where the GeoJSON file is located.
    filename (str): The name of the GeoJSON file.

    ## Returns:
    gpd.GeoDataFrame: The processed GeoDataFrame.

    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        gpd_df = gpd.read_file(file_path)
    return gpd_df


def features(dataset, path: str, mode: str):
    """
    Process and save or load features and labels.

    Args:
        dataset: The dataset to process.
        path (str): The path to save or load the processed data.
        mode (str): The mode of operation. Can be 'save' or 'load'.

    Returns:
        None
    """
    dataset = features.detach().cpu().numpy()
    
    if mode == 'save':
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('features', data= dataset)
            hf.create_dataset('labels', data=labels)
    if mode == 'load':
        with h5py.File(path, 'r') as hf:
            features = hf['features'][:]
            labels = hf['labels'][:]