import os
import pandas as pd
import numpy as np
from math import radians, degrees, sin, cos, sqrt, atan2
from tqdm import tqdm

# Initialiser tqdm pour pandas
tqdm.pandas()

# DataFrame des antennes
antennas_df = pd.DataFrame({
    'antenna': ['0QRDE8F0010496', '0QRDJCAR037P0N', '0QRDJCAR037P0N', '0QRDKC2R03J32P', '0QRDKC2R038370', 
                '0QRDE8F0010496', '0QRDE8F0010496', '0QRDKC2R03J32P', '0QRDKC2R038370', '0QRDKC2R038370', 
                '0QRDJCAR037P0N'],
    'start_at': [1626739200, 1655164800, 1677628800, 1687219200, 1690848000, 
                 1683504000, 1697500800, 1708387200, 1676332800, 1712534400, 
                 1719878400],
    'end_at': [1682208000, 1677542400, 1714694400, 1706054400, 1696118400, 
               1694736000, np.nan, np.nan, 1706054400, np.nan, 
               np.nan],
    'latitude': [48.6, 49.01, 49.6, 50.8, 44.8, 
                 54.7, 51.52, 35.16, 49.01, 48.10, 
                 44.57],
    'longitude': [2.35, 2.55, 6.2, 4.4, -0.7, 
                  -6.2, -0.05, 33.28, 2.55, 16.59, 
                  26.1],
    'elevation': [89, 119, 376, 56, 49, 
                  82, 6, 3, 119, 183, 
                  96],
    'name': ['EIH', 'CDG', 'LUX', 'BRU', 'BDX', 
             'BEL', 'LON', 'CYP', 'CDG', 'AUS', 
             'BUC']
})

def load_and_merge_files(directory):
    # Liste pour stocker tous les DataFrames
    dataframes = []
    
    # Parcourir tous les fichiers dans le dossier
    for filename in tqdm(os.listdir(directory), desc="Chargement des fichiers"):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            dataframes.append(df)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            dataframes.append(df)
    
    # Fusionner tous les DataFrames en un seul
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
    else:
        print("Aucun fichier valide trouvé dans le dossier.")
        return None
    
    return merged_df

def determine_station_elevation(antenna_id, timestamp):
    # Trouver la station correspondant à la ligne de données actuelle
    matching_antenna = antennas_df[
        (antennas_df['antenna'] == antenna_id) & 
        (antennas_df['start_at'] <= timestamp) & 
        ((antennas_df['end_at'] >= timestamp) | pd.isna(antennas_df['end_at']))
    ]
    if not matching_antenna.empty:
        return matching_antenna.iloc[0]['elevation']
    return np.nan

def process_data_vectorized(df):
    # Filtrer les données : supprimer les lignes où la source est 'wi' ou où l'altitude est NaN
    df = df[df['source'] != 'wi']
    df = df.dropna(subset=['altitude'])
    
    # Supprimer les colonnes spécifiques
    columns_to_drop = ['journey', 'ident', 'model', 'source', 'location', 'timestamp', 'gps', 
                       'rssi', 'home_lat', 'home_lon', 'home_height', 'speed', 
                       'heading', 'year', 'month']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 1. Ajouter l'élévation de la station pour chaque ligne en vectorisé
    df['station_elevation'] = df.progress_apply(lambda row: determine_station_elevation(row['station_name'], row['time']), axis=1)
    
    # 2. Calculer la hauteur du drone par rapport à la station
    df['relative_altitude'] = df['altitude'] - df['station_elevation']
    
    # 3. Calcul vectorisé de la distance en utilisant Haversine
    lat1, lon1, lat2, lon2 = map(np.radians, [df['latitude'], df['longitude'], df['station_latitude'], df['station_longitude']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6371.0
    df['distance_km'] = r * c
    
    # Calcul vectorisé de l'azimut
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    initial_bearing = np.degrees(np.arctan2(x, y))
    df['azimuth_deg'] = np.round((initial_bearing + 360) % 360)
    
    # 4. Calcul de l'angle d'élévation en utilisant l'altitude relative
    df['elevation_angle_deg'] = np.round(np.degrees(np.arctan2(df['relative_altitude'], df['distance_km'] * 1000)) * 10)
    
    return df

def save_processed_data(df, output_file):
    # Sauvegarder le DataFrame dans un fichier CSV ou Parquet
    print("Sauvegarde du fichier...")
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    else:
        print("Format de fichier non supporté. Utilisez .csv ou .parquet")
    print("Fichier sauvegardé avec succès.")

def main():
    directory = input("Entrez le chemin du dossier contenant les fichiers CSV et Parquet : ")
    output_file = input("Entrez le nom du fichier de sortie (avec extension .csv ou .parquet) : ")
    
    merged_df = load_and_merge_files(directory)
    if merged_df is not None:
        processed_df = process_data_vectorized(merged_df)
        save_processed_data(processed_df, output_file)
        print(f"Traitement terminé. Fichier sauvegardé sous {output_file}")

if __name__ == "__main__":
    main()
