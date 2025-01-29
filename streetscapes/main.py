import models
import processing

import numpy as np
from collections import Counter

device = torch.device('cpu')
directory = '_data\\geo_json\\panoids'
gpd_df = {}

for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.geojson'):
            gpd_df[filename] = geojson(root, filename)
print(f"Loaded {len(gpd_df)} GeoJSON files")

with h5py.File(FEATURES_FILE, 'r') as f:
    features_np = f['features'][:]
with open(LABELS_FILE, 'r') as f:
    labels = [line.strip() for line in f]

pca_model = PCA(n_components=2)
optimal_n, variance_ratios = pca_model.find_optimal_components(features_np, threshold=0.95)
pca_model.n_components = optimal_n
data = pca_model.fit_transform(features_np)

km = KM()
km.fit(data, 4)
cluster_labels = km.get_cluster_assignments(data)

STREETSCAPES_df = gpd_df['panoids.geojson'].drop(columns=['year', 'month', 'owner', 'ask_lng', 'ask_lat', 'consulted', 'url_side_a', 'url_front', 'url_side_b', 'url_back'])
cols = ['im_side_a', 'im_front', 'im_side_b', 'im_back']
new_cols = [f'{col}_cluster' for col in cols]
STREETSCAPES_df[new_cols] = pd.DataFrame([[None]*len(new_cols)], index=STREETSCAPES_df.index)

label_to_cluster = dict(zip(labels, cluster_labels))

for col, new_col in zip(cols, new_cols):
    for idx, row in STREETSCAPES_df.iterrows():
        image_label = row[col]
        if image_label in label_to_cluster:
            STREETSCAPES_df.at[idx, new_col] = label_to_cluster[image_label]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

coords = STREETSCAPES_df[['lat', 'lng']].values
threshold_distance = 20
cluster_cols = ['im_side_a_cluster', 'im_front_cluster', 'im_side_b_cluster', 'im_back_cluster']
used = set()
groups = []

for i in range(len(coords)):
    if i in used: continue
    group = [i]
    used.add(i)
    for j in range(i + 1, len(coords)):
        distance = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        if distance < threshold_distance:
            group.append(j)
            used.add(j)
        else: break
    if len(group) > 1: groups.append(group)

STREETSCAPES_df['grouped_cluster'] = None
STREETSCAPES_df['grouped_labels'] = None

for group in groups:   
    all_clusters = [c for idx in group for c in STREETSCAPES_df.loc[STREETSCAPES_df.index[idx], cluster_cols].dropna().values]
    if all_clusters:
        majority_cluster = Counter(all_clusters).most_common(1)[0][0]
        all_labels = [l for idx in group for l in STREETSCAPES_df.loc[STREETSCAPES_df.index[idx], ['im_side_a', 'im_front', 'im_side_b', 'im_back']].dropna().values]
        first_idx = STREETSCAPES_df.index[group[0]]
        STREETSCAPES_df.at[first_idx, 'grouped_cluster'] = majority_cluster
        STREETSCAPES_df.at[first_idx, 'grouped_labels'] = ','.join(all_labels)

indices_to_drop = [idx for group in groups for idx in group[1:]]
STREETSCAPES_df = STREETSCAPES_df.drop(STREETSCAPES_df.index[indices_to_drop]).drop(columns=cluster_cols)
STREETSCAPES_df = STREETSCAPES_df[STREETSCAPES_df['grouped_cluster'].notnull()]

plot_feature_classes_kmeans(STREETSCAPES_df['lat'].values, STREETSCAPES_df['lng'].values, STREETSCAPES_df['grouped_cluster'].values, STREETSCAPES_df['grouped_labels'].values)