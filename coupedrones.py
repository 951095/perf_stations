# VPA ( visualisation performance antenna )
#
# This is an app in python to visualize the performance of every antenna used since the start of the ACUTE project
#
# Armand Bécot
# Projet ACUTE, Europe, France, Le plessi-pâté


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import os


# Lecture du fichier CSV
df = pd.read_csv('../CSV/drones_with_real_alt_recalculated_and_vertical_angle.csv')

# Ajustement des angles de angle_from_station
df['angle_from_station'] = (df['angle_from_station'] - 90) % 360

def compute_radiation_data_all(city_name, altitude_range, distance_max, display_all_drones, use_true_altitude):
    filtered_df_city = df[df['ville'] == city_name]

    if filtered_df_city.empty:
        return None, None, None, 0

    altitude_column = 'real_altitude'

    distances = filtered_df_city['distance_to_station'].tolist()
    angles = filtered_df_city['angle_from_station'].tolist()
    altitudes = filtered_df_city[altitude_column].tolist()

    if not display_all_drones:
        alt_min, alt_max = altitude_range
        filtered_df_city = filtered_df_city[((filtered_df_city[altitude_column]) >= alt_min) & ((filtered_df_city[altitude_column]) <= alt_max)]
        distances = filtered_df_city['distance_to_station'].tolist()
        angles = filtered_df_city['angle_from_station'].tolist()
        altitudes = filtered_df_city[altitude_column].tolist()
    else:
        distances = [min(distance, distance_max) if distance > distance_max else distance for distance in distances]

    num_data_points = len(filtered_df_city)
    distance_per_degree = np.zeros(360)

    for angle, distance in zip(angles, distances):
        if distance <= distance_max:
            degree = int(angle)
            distance_per_degree[degree] = max(distance_per_degree[degree], distance)

    return distance_per_degree, angles, altitudes, num_data_points

def plot_radiation_diagram_all(city_name, altitude_range, distance_max, display_all_drones, selected_angle, use_true_altitude):
    distance_per_degree, angles, altitudes, num_data_points = compute_radiation_data_all(city_name, altitude_range, distance_max, display_all_drones, use_true_altitude)

    if distance_per_degree is None:
        st.write("No data available for the selected city.")
        return

    st.write(f"Number of data points displayed: {num_data_points}")

    # Inverser les angles pour le sens horaire pour le tracé, mais garder les angles originaux pour l'affichage hoverinfo
    inverted_angles = [(360 - angle) % 360 for angle in angles]

    # Ajuster les angles pour l'affichage hoverinfo
    hover_texts = [f"Angle: {angle}°, Distance: {distance:.2f} km, Altitude: {altitude:.2f} m"
                for angle, distance, altitude in zip(angles, distance_per_degree, altitudes)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=distance_per_degree,
        theta=list(range(360)),  # Utiliser une liste de 0 à 359 pour les angles
        mode='lines',
        line=dict(color='black', width=1),
        name=f'Distances from {city_name}',
        text=hover_texts,
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[-max(distance_per_degree), max(distance_per_degree)],
        theta=[(360 - selected_angle) % 360, (360 - selected_angle) % 360],
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
                ticktext=[f'{360 - i}°' for i in range(0, 360, 10)],
                rotation=90  # Orienter 0° au nord
            )
        ),
        showlegend=True,
        title=f'Polar diagram in {city_name}',
        width=800,
        height=800,
        margin=dict(l=100, r=20, t=50, b=50)
    )

    st.plotly_chart(fig)

def compute_radiation_data(city_name, distance_max, selected_angle, use_true_altitude):
    filtered_df_city = df[(df['ville'] == city_name) & (df['angle_from_station'] == selected_angle)]

    if filtered_df_city.empty:
        return None, None, None, 0

    altitude_column = 'real_altitude'
    filtered_df_city = filtered_df_city[filtered_df_city[altitude_column].notna()]

    distances = filtered_df_city['distance_to_station'].tolist()
    angles = filtered_df_city['vertical_angle_degrees'].tolist()
    altitudes = filtered_df_city[altitude_column].tolist()

    num_data_points = len(filtered_df_city)
    distance_per_degree = np.zeros(360)

    for angle, distance in zip(angles, distances):
        if not np.isnan(angle) and distance <= distance_max:
            degree = int(angle)
            distance_per_degree[degree] = max(distance_per_degree[degree], distance)

    return distance_per_degree, angles, altitudes, num_data_points

def plot_radiation_diagram(city_name, distance_max, selected_angle, use_true_altitude):
    distance_per_degree, angles, altitudes, num_data_points = compute_radiation_data(city_name, distance_max, selected_angle, use_true_altitude)

    if distance_per_degree is None:
        st.write("No data available for the selected city and angle.")
        return

    st.write(f"Number of data points displayed: {num_data_points} ")

    # Inverser les angles pour le sens horaire
    angles = [(360 - angle) % 360 for angle in angles]

    # Ajuster les angles pour l'affichage hoverinfo
    hover_texts = [f"Distance: {distance:.2f} km, Altitude: {altitude:.2f} m"
                for distance, altitude in zip(distance_per_degree, altitudes)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=distance_per_degree,
        theta=list(range(360)),
        mode='lines',
        line=dict(color='blue', width=1),
        name=f'Distances from {city_name}',
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
                ticktext=[f'{360 - i}°' for i in range(0, 360, 10)],
                rotation=0
            )
        ),
        showlegend=True,
        title=f'Polar diagram in {city_name} for {selected_angle}° cap (in this diagram altitude is not taken into account)',
        width=800,
        height=800,
        margin=dict(l=100, r=20, t=50, b=50)
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(layout="wide")

    st.title('Polar diagram')

    unique_city_names = df['ville'].unique()
    city_options = [name for name in unique_city_names]
    selected_city_name = st.selectbox('Select a city', city_options)

    altitudes = [-100, 0, 50, 100, 150, 200, 300, 500, 1000, 10000]
    altitude_pairs = [(altitudes[i], altitudes[i + 1]) for i in range(len(altitudes) - 1)]
    selected_altitude_pair = st.radio('Select altitude range (in meters)', altitude_pairs, index=2, format_func=lambda x: f"{x[0]} to {x[1]}", horizontal=True)

    rayon = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    selected_rayon = st.radio('Select antenna radius (in KM)', rayon, index=4, format_func=lambda x: f"{x} KM", horizontal=True)

    use_true_altitude = False   
    display_all_drones = st.checkbox('Display every drones ( it will display drone from -100 meters to 10000 meters)', value=False)
    angle_input_method = st.radio('Angle selection method', ['Number', 'Bar'], horizontal=True)
    if angle_input_method == 'Number':
        selected_angle = st.number_input('Select an angle', min_value=0, max_value=359, value=0, step=1)
    else:
        selected_angle = st.slider('Select an angle', min_value=0, max_value=359, value=0, step=1)

    if selected_city_name and (selected_altitude_pair or display_all_drones):
        col1, col2 = st.columns(2)
        with col1:
            plot_radiation_diagram_all(selected_city_name, selected_altitude_pair, selected_rayon, display_all_drones, selected_angle, use_true_altitude)
        with col2:
            plot_radiation_diagram(selected_city_name, selected_rayon, selected_angle, use_true_altitude)

if __name__ == "__main__":
    main()