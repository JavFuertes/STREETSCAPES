import os
import geopandas as gpd

def process_geojson(directory: str, filename: str) -> gpd.GeoDataFrame:
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