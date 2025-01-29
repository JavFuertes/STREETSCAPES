import folium
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keplergl import KeplerGl

def plot_feature_classes_responsibilities(latitude: np.array, longitude: np.array, responsibilities: np.array) -> folium.Map:
    """
    Creates a folium map with circle markers representing the feature classification based on responsibilities.

    Parameters:
    latitude (numpy.ndarray): Array of latitude values.
    longitude (numpy.ndarray): Array of longitude values.
    responsibilities (numpy.ndarray): Matrix of shape (n_points, n_clusters) representing the cluster responsibilities.

    Returns:
    folium.Map: Folium map object with circle markers representing the feature classification.
    """
    m = folium.Map(location=[latitude.mean(), longitude.mean()], zoom_start=15)

    folium.TileLayer('Esri.WorldImagery').add_to(m)
    n_clusters = responsibilities.shape[1]

    colors = plt.get_cmap('viridis', n_clusters)
    for lat, lon, resp in zip(latitude, longitude, responsibilities):
        r, g, b, a = 0, 0, 0, 0
        for i in range(n_clusters):
            rgba = colors(i)  
            r += rgba[0] * resp[i]
            g += rgba[1] * resp[i]
            b += rgba[2] * resp[i]
            a += resp[i] / n_clusters  

        a = min(max(a, 0), 1)    
        final_color = matplotlib.colors.to_hex([r, g, b, a])
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=final_color,
            fill=True,
            fill_color=final_color,
            popup=f'Responsibilities: {resp}',
            fill_opacity=a
        ).add_to(m)

    return m

def plot_feature_classes_kmeans(latitude: np.array, longitude: np.array, cluster_labels: np.array, image_labels: np.array) -> folium.Map:
    """
    Creates a folium map with circle markers representing the feature classification based on KMeans clustering.
    Each marker includes the cluster label and an image label in its popup.

    Parameters:
    latitude (numpy.ndarray): Array of latitude values.
    longitude (numpy.ndarray): Array of longitude values.
    cluster_labels (numpy.ndarray): Array of cluster labels for each point.
    image_labels (numpy.ndarray): Array of image labels corresponding to each point.

    Returns:
    folium.Map: Folium map object with circle markers representing the feature classification.
    """
    m = folium.Map(location=[latitude.mean(), longitude.mean()], zoom_start=15)

    folium.TileLayer('Esri.WorldImagery').add_to(m)
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.get_cmap('viridis', n_clusters)
    for lat, lon, cluster_label, image_label in zip(latitude, longitude, cluster_labels, image_labels):
        rgba = colors(cluster_label % n_clusters)  # Get RGBA values for the cluster
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=matplotlib.colors.to_hex(rgba[:3]),  # Use RGB for color
            fill=True,
            fill_color=matplotlib.colors.to_hex(rgba[:3]),
            popup=f'Cluster: {cluster_label}<br>Label: {image_label}',  # Include image label in popup
            fill_opacity=0.6
        ).add_to(m)

    return m

def map_geojson(gpds,save: bool = False, save_path = '_maps\\STREETSCAPES01.html'):
    map = KeplerGl(height=600)
    for gdf_name in gpds:
        map.add_data(gpds[gdf_name], name= gdf_name )

    if save:
        map.save_to_html(file_name= save_path, config={
            'mapState': {
                'latitude': 52.01153531997234,
                'longitude': 4.3588424177636185,
                'zoom': 16
            }
        })
    return map
