import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st

def load_data(uploaded_file):
    # Détecter le type de fichier et charger les données
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.parquet'):
        return pd.read_parquet(uploaded_file)
    else:
        st.error("Unsupported file format. Please select a CSV or Parquet file.")
        return None

def main():
    st.set_page_config(layout="wide")

    st.title('Polar diagram')

    # Ajout d'une option pour sélectionner un fichier
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'parquet'])

    if uploaded_file is not None:
        # Charger les données
        df = load_data(uploaded_file)
        if df is not None:
            # Convertir les timestamps en datetime pour la gestion des dates
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Ajustement des angles de azimuth_deg
            df['azimuth_deg'] = df['azimuth_deg'] % 360  # Garder l'angle tel quel

            unique_city_names = df['station_name'].unique()
            city_options = [name for name in unique_city_names]
            selected_city_name = st.selectbox('Select a city', city_options)

            # Filtrage des dates les plus tôt et les plus tardives
            min_date = df[df['station_name'] == selected_city_name]['time'].min()
            max_date = df[df['station_name'] == selected_city_name]['time'].max()
            selected_date_range = st.date_input('Select date range', [min_date, max_date], min_value=min_date, max_value=max_date)

            # Filtrer les données en fonction de la plage de dates sélectionnée
            filtered_df = df[(df['station_name'] == selected_city_name) & (df['time'] >= pd.to_datetime(selected_date_range[0])) & (df['time'] <= pd.to_datetime(selected_date_range[1]))]

            altitudes = [-100, 0, 50, 100, 150, 200, 300, 500, 1000, 10000]
            altitude_pairs = [(altitudes[i], altitudes[i + 1]) for i in range(len(altitudes) - 1)]
            selected_altitude_pair = st.radio('Select altitude range (in meters)', altitude_pairs, index=2, format_func=lambda x: f"{x[0]} to {x[1]}", horizontal=True)

            rayon = [10, 20, 30, 40, 50, 60, 70, 80, 100]
            selected_rayon = st.radio('Select antenna radius (in KM)', rayon, index=4, format_func=lambda x: f"{x} KM", horizontal=True)

            use_true_altitude = False   
            display_all_drones = st.checkbox('Display every drones (it will display drone from -100 meters to 10000 meters)', value=False)
            angle_input_method = st.radio('Angle selection method', ['Number', 'Bar'], horizontal=True)
            if angle_input_method == 'Number':
                selected_angle = st.number_input('Select an angle', min_value=0, max_value=359, value=0, step=1)
            else:
                selected_angle = st.slider('Select an angle', min_value=0, max_value=359, value=0, step=1)

            if selected_city_name and (selected_altitude_pair or display_all_drones):
                col1, col2 = st.columns(2)
                with col1:
                    plot_radiation_diagram_all(selected_city_name, selected_altitude_pair, selected_rayon, display_all_drones, selected_angle, use_true_altitude, filtered_df)
                with col2:
                    plot_radiation_diagram_elevation(selected_city_name, selected_angle, selected_rayon, filtered_df)

def plot_radiation_diagram_all(city_name, altitude_range, distance_max, display_all_drones, selected_angle, use_true_altitude, df):
    distance_per_degree, angles, altitudes, num_data_points = compute_radiation_data_all(city_name, altitude_range, distance_max, display_all_drones, use_true_altitude, df)

    if distance_per_degree is None:
        st.write("No data available for the selected city.")
        return

    st.write(f"Number of data points displayed: {num_data_points}")

    hover_texts = [f"Angle: {angle}°, Distance: {distance:.2f} km, Altitude: {altitude:.2f} m"
                   for angle, distance, altitude in zip(angles, distance_per_degree, altitudes)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=distance_per_degree,
        theta=angles,  # Utiliser les angles originaux
        mode='lines',
        line=dict(color='black', width=1),
        name=f'Distances from {city_name}',
        text=hover_texts,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[0, distance_max],
        theta=[selected_angle, selected_angle],
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, distance_max],
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 360, 10)),
                ticktext=[f'{i}°' for i in range(0, 360, 10)],
                rotation=0  # Orienter 0° au nord
            )
        ),
        showlegend=True,
        title=f'Polar diagram in {city_name}',
        width=800,
        height=800,
        margin=dict(l=100, r=20, t=50, b=50)
    )

    st.plotly_chart(fig)

def compute_radiation_data_all(city_name, altitude_range, distance_max, display_all_drones, use_true_altitude, df):
    filtered_df_city = df[df['station_name'] == city_name]

    if filtered_df_city.empty:
        return None, None, None, 0

    altitude_column = 'altitude'

    if not display_all_drones:
        alt_min, alt_max = altitude_range
        filtered_df_city = filtered_df_city[(filtered_df_city[altitude_column] >= alt_min) & (filtered_df_city[altitude_column] <= alt_max)]

    # Grouper par angle (azimuth_deg) et prendre la distance maximale pour chaque angle
    grouped = filtered_df_city.groupby('azimuth_deg').agg({'distance_km': 'max', 'altitude': 'first'}).reset_index()

    # Appliquer la limite d'affichage à la distance maximale
    grouped['distance_km'] = np.where(grouped['distance_km'] > distance_max, distance_max, grouped['distance_km'])

    angles = grouped['azimuth_deg'].tolist()
    distances = grouped['distance_km'].tolist()
    altitudes = grouped['altitude'].tolist()

    # Initialiser une liste de distances avec 0 pour chaque angle de 0 à 359
    distance_per_degree = np.zeros(360)
    altitude_per_degree = np.zeros(360)

    # Remplir les distances et altitudes pour les angles où il y a des données
    for angle, distance, altitude in zip(angles, distances, altitudes):
        degree = int(angle)
        distance_per_degree[degree] = distance
        altitude_per_degree[degree] = altitude

    # Remplir les angles manquants avec des distances de 0 km
    for i in range(360):
        if distance_per_degree[i] == 0:
            altitude_per_degree[i] = 0  # ou np.nan si vous préférez
            distance_per_degree[i] = 0

    return distance_per_degree, list(range(360)), altitude_per_degree, len(angles)

def plot_radiation_diagram_elevation(city_name, selected_angle, distance_max, df):
    distances, elevations, num_data_points = compute_radiation_data_elevation(city_name, selected_angle, distance_max, df)

    if distances is None:
        st.write("No data available for the selected city and angle.")
        return

    st.write(f"Number of data points displayed: {num_data_points}")

    hover_texts = [f"Elevation: {elevation}°, Distance: {distance:.2f} km"
                   for distance, elevation in zip(distances, elevations)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=distances,
        theta=elevations,  # Utiliser les angles d'élévation
        mode='lines',
        line=dict(color='blue', width=1),
        name=f'Distances from {city_name} at {selected_angle}° azimuth',
        text=hover_texts,
        hoverinfo='text'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, distance_max],
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 360, 10)),
                ticktext=[f'{i}°' for i in range(0, 360, 10)],
                rotation=0  # Orienter 0° au nord
            )
        ),
        showlegend=True,
        title=f'Polar diagram of elevation angles at {selected_angle}° azimuth in {city_name}',
        width=800,
        height=800,
        margin=dict(l=100, r=20, t=50, b=50)
    )

    st.plotly_chart(fig)

def compute_radiation_data_elevation(city_name, selected_angle, distance_max, df):
    filtered_df_city = df[(df['station_name'] == city_name) & (df['azimuth_deg'] == selected_angle)]

    if filtered_df_city.empty:
        return None, None, 0

    # Grouper par angle d'élévation et prendre la distance maximale pour chaque angle d'élévation
    grouped = filtered_df_city.groupby('elevation_angle_deg').agg({'distance_km': 'max'}).reset_index()

    elevations = grouped['elevation_angle_deg'].tolist()
    distances = grouped['distance_km'].tolist()

    num_data_points = len(grouped)

    return distances, elevations, num_data_points

if __name__ == "__main__":
    main()
