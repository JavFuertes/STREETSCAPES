from keplergl import KeplerGl

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