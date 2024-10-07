import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pydeck as pdk
from utils.constants import *


def get_address_density(df: pd.DataFrame, selected_types) -> pd.DataFrame:
    # Filter the data based on selected call types
    filtered_data = df[df['Call Type'].isin(selected_types)]
    
    # Calculate density (number of calls) for each address
    address_density = filtered_data.groupby(['Address', 'Latitude', 'Longitude']).size().reset_index(name='Density')
    
    # Sort by density and get the top ten places
    top_ten_addresses = address_density.nlargest(10, 'Density')
    st.dataframe(top_ten_addresses)
    return top_ten_addresses

def assign_colors(densities):
    # Define a color gradient from blue to red
    colors = []
    max_density = max(densities)
    min_density = min(densities)
    for density in densities:
        ratio = (density - min_density) / (max_density - min_density) if max_density != min_density else 0.5
        red = int(255 * ratio)        
        green = int(255 * (1 - ratio))
        colors.append([red, green, 0, 160])
    return colors


def show_bubble_map(df: pd.DataFrame, selected_types) -> None:
    top_ten_addresses = get_address_density(df, selected_types)
    
    # Apply color based on density
    top_ten_addresses['color'] = assign_colors(top_ten_addresses['Density'])
    
    # Define the ScatterplotLayer for the bubble map
    layer = pdk.Layer(
        "ScatterplotLayer",
        top_ten_addresses,
        get_position=["Longitude", "Latitude"],
        get_radius='Density * 3',
        get_fill_color='color',
        get_line_color=[0, 0, 0],
        pickable=True
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        latitude=top_ten_addresses['Latitude'].median(),
        longitude=top_ten_addresses['Longitude'].median(),
        zoom=10.5,
        pitch=0
    )
    
    # Render the map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Address:</b> {Address}<br>"
                    "<b>Density:</b> {Density}"
        }
    )
    st.pydeck_chart(r)


def get_map_data(df: pd.DataFrame, selected_types) -> pd.DataFrame:
    # Create a DataFrame to store the map data
    map_data = pd.DataFrame()
    map_data['lat'] = df['Latitude'].astype(float)
    map_data['lon'] = df['Longitude'].astype(float)
    map_data['B-Party'] = df['B-Party'].astype(str)
    map_data['Call Type'] = df['Call Type'].astype(str)
    map_data['Date & Time'] = pd.to_datetime(df['Date & Time'])
    map_data['Duration'] = df['Duration']
    map_data['Address'] = df['Address'].astype(str)
    
    # Extract date and time from 'Date & Time' for display
    map_data['Date'], map_data['Time'] = zip(*map_data['Date & Time'].astype(str).apply(lambda x: x.split(' ')))
    
    # Filter by selected call types
    map_data = map_data[map_data['Call Type'].isin(selected_types)]
    
    # Identify the latest entry by 'Date & Time'
    latest_entry_index = map_data['Date & Time'].idxmax()
    
    # Assign colors based on Call Type
    color_data = map_data['Call Type'].apply(
        lambda x: [255, 0, 0, 255] if x == 'InComing' else  # Red
                ([0, 0, 255, 255] if x == 'Outgoing' else  # Blue
                ([0, 128, 0, 255] if x == 'Incoming SMS' else  # Green
                    ([255, 165, 0, 255] if x == 'Outgoing SMS' else [255, 255, 255, 160])))  # Orange
    ).tolist()
    
    map_data['color'] = color_data
    
    # Set radius (size) for the points, with a larger radius for the latest entry
    map_data['radius'] = 120  # Default radius
    map_data.loc[latest_entry_index, 'radius'] = 250  # Increase the radius for the latest location
    
    return map_data


def show_map(df: pd.DataFrame) -> None:
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Allow users to select call types
        unique_call_types = df['Call Type'].unique().tolist()
        st.info("INFO: Select the call types from the dropdown to display on the map.")
        selected_types = st.multiselect('Select Call Types:', unique_call_types, default="Outgoing", key="call_map") 
        
        if selected_types:
            map_data = get_map_data(df, selected_types)
            # Define the ScatterplotLayer
            layer = pdk.Layer(
                "ScatterplotLayer",
                map_data,
                get_position=["lon", "lat"],
                get_color="color",  # Use the color column
                get_radius="radius",  # Use the radius column
                pickable=True, 
                auto_highlight=True,
                highlight_color=[255, 255, 255, 255], 
            )
            # Set the viewport location (focus on the most frequent location)
            view_state = pdk.ViewState(
                latitude=map_data['lat'].mode()[0],
                longitude=map_data['lon'].mode()[0],
                zoom=11,
                pitch=0,
                mode="geospatial" 
            )
            # Render the map
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>B-Party:</b> {B-Party}<br>"
                            "<b>Call Type:</b> {Call Type}<br>"
                            "<b>Date:</b> {Date}<br>"
                            "<b>Time:</b> {Time}<br>"
                            "<b>Duration (secs):</b> {Duration}<br>"
                            "<b>Address:</b> {Address}"
                }
            )
            st.pydeck_chart(r)
        else:
            st.warning("Please select at least one call type to display the map.")
    else:
        st.error("Latitude and Longitude columns are missing in the dataframe.")

def show_density_map(df: pd.DataFrame) -> None:
    # Get the address density data
    unique_call_types = df['Call Type'].unique().tolist()
    st.info("INFO: Select the call types from the dropdown to display on the map.")
    selected_types = st.multiselect('Select Call Types:', unique_call_types, default="Outgoing", key='density_map') 
    
    top_ten_addresses = get_address_density(df, selected_types)
    
    # Apply color based on density
    top_ten_addresses['color'] = assign_colors(top_ten_addresses['Density'])
    
    # Define the ScatterplotLayer for the density map
    layer = pdk.Layer(
        "ScatterplotLayer",
        top_ten_addresses,
        get_position=["Longitude", "Latitude"],
        get_radius='Density * 3',  # Radius based on density
        get_fill_color='color',    # Color based on density
        get_line_color=[0, 0, 0],  # Border color for the circles
        pickable=True,             # Enable interaction
    )
    
    # Set the viewport location to center the map on the most dense location
    view_state = pdk.ViewState(
        latitude=top_ten_addresses['Latitude'].median(),
        longitude=top_ten_addresses['Longitude'].median(),
        zoom=10.5,
        pitch=0
    )
    
    # Render the density map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Address:</b> {Address}<br>"
                    "<b>Density:</b> {Density}"  # Display density in the tooltip
        }
    )
    st.pydeck_chart(r)

def show_heat_map(df: pd.DataFrame) -> None:
    # Filter data based on selected call types
    unique_call_types = df['Call Type'].unique().tolist()
    st.info("INFO: Select the call types from the dropdown to display on the map.")
    selected_types = st.multiselect('Select Call Types:', unique_call_types, default="Outgoing", key='heat_map') 
    map_data = df[df['Call Type'].isin(selected_types)]
    
    # Ensure the data contains latitude and longitude
    if 'Latitude' in map_data.columns and 'Longitude' in map_data.columns:
        # Create a new DataFrame for the heatmap with necessary columns
        heatmap_data = pd.DataFrame()
        heatmap_data['lat'] = map_data['Latitude'].astype(float)
        heatmap_data['lon'] = map_data['Longitude'].astype(float)
        heatmap_data['Call Type'] = map_data['Call Type'].astype(str)
        heatmap_data['Date & Time'] = pd.to_datetime(map_data['Date & Time'])
        heatmap_data['date_time_str'] = heatmap_data['Date & Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Define the HeatmapLayer
        layer = pdk.Layer(
            "HeatmapLayer",
            data=heatmap_data,
            get_position=["lon", "lat"],
            aggregation='MEAN',
            radiusPixels=60,  # Adjust the radius for heat intensity
            opacity=0.6,      # Set opacity for the heatmap layer
        )
        
        # Set the initial view state (zoom into the main area of activity)
        view_state = pdk.ViewState(
            latitude=heatmap_data['lat'].median(),
            longitude=heatmap_data['lon'].median(),
            zoom=10.5,
            pitch=0,
        )
        
        # Render the heatmap
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Call Type:</b> {Call Type}<br>"
                        "<b>Date & Time:</b> {date_time_str}<br>"
                        "<b>Latitude:</b> {lat}<br>"
                        "<b>Longitude:</b> {lon}"
            }
        )
        st.pydeck_chart(r)
    else:
        st.error("Latitude and Longitude columns are missing in the dataframe.")

def show_time_bound_map(df: pd.DataFrame) -> None:
    # Filter data based on selected call types
    unique_call_types = df['Call Type'].unique().tolist()
    st.info("INFO: Select the call types from the dropdown to display on the map.")
    selected_types = st.multiselect('Select Call Types:', unique_call_types, default="Outgoing", key='time_bound_map') 
    map_data = df[df['Call Type'].isin(selected_types)]
    map_data['date_time_str'] = map_data['Date & Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
     # Ensure 'Date & Time' column is in datetime format
    map_data['Date & Time'] = pd.to_datetime(map_data['Date & Time'])

    # Define the minimum and maximum dates in the dataset
    min_date = map_data['Date & Time'].min()
    max_date = map_data['Date & Time'].max()

    # Create a time range slider in Streamlit for users to select the time frame
    st.info("INFO: Adjust the slider to select the time range for displaying locations.")
    selected_time_range = st.slider(
        "Select Date & Time Range:",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD HH:mm:ss"
    )
    
    # Filter data based on the selected time range
    time_filtered_data = map_data[
        (map_data['Date & Time'] >= pd.to_datetime(selected_time_range[0])) &
        (map_data['Date & Time'] <= pd.to_datetime(selected_time_range[1]))
    ]

    # Check if the filtered dataset has any data
    if time_filtered_data.empty:
        st.warning("No data available for the selected time range.")
        return
    
    # Create a new DataFrame for plotting with necessary columns
    plot_data = pd.DataFrame()
    plot_data['lat'] = time_filtered_data['Latitude'].astype(float)
    plot_data['lon'] = time_filtered_data['Longitude'].astype(float)
    plot_data['Call Type'] = time_filtered_data['Call Type'].astype(str)
    plot_data['Date & Time'] = time_filtered_data['Date & Time']
    
    # Define the ScatterplotLayer for the time-bound map
    layer = pdk.Layer(
        "ScatterplotLayer",
        plot_data,
        get_position=["lon", "lat"],
        get_radius=120,  # Adjust the radius if needed
        get_fill_color=[255, 140, 0, 160],  # Orange color for all points
        pickable=True
    )
    
    # Set the viewport location based on the filtered data
    view_state = pdk.ViewState(
        latitude=plot_data['lat'].median(),
        longitude=plot_data['lon'].median(),
        zoom=11,
        pitch=0
    )
    
    # Render the map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Call Type:</b> {Call Type}<br>"
                    "<b>Date & Time:</b> {date_time_str}<br>"
                    "<b>Latitude:</b> {lat}<br>"
                    "<b>Longitude:</b> {lon}"
        }
    )
    st.pydeck_chart(r)


def top_locations(data: pd.DataFrame):
    st.subheader("Top 10 Locations")
    st.write("This analysis displays the top locations for calls based on the selected call types. "
             "It provides insights into the geographical distribution of call activities, helping to identify "
             "hotspots for different types of calls.")
    st.write("")
     # Filter out rows without latitude and longitude
    df = data.dropna(subset=['Latitude', 'Longitude', 'Address'])
    
    # Convert latitude and longitude to float
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    
    # Get unique call types
    unique_call_types = df['Call Type'].unique().tolist()
    
    # Streamlit widget for user selection
    selected_types = st.multiselect('Select Call Types:', unique_call_types, default=unique_call_types)
    
    if selected_types:
        show_bubble_map(df, selected_types)
    else:
        st.warning("Please select at least one call type to display the plot.")
        
        


def show_line_tracking_chart(df: pd.DataFrame) -> None:
    # Filter data to ensure it contains 'Date & Time', 'Latitude', and 'Longitude'
    if 'Date & Time' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        st.error("The dataframe is missing required columns: 'Date & Time', 'Latitude', or 'Longitude'.")
        return

    df['date_time_str'] = df['Date & Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Ensure the 'Date & Time' column is in datetime format
    df.loc[:, 'Date & Time'] = pd.to_datetime(df['Date & Time'], errors='coerce')
    df = df.dropna(subset=['Date & Time'])  # Drop rows where 'Date & Time' is NaT

    # Check for valid Latitude and Longitude data
    df.loc[:, 'Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df.loc[:, 'Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Convert 'Date & Time' to string for the tooltip
    df.loc[:, 'Date & Time'] = df['Date & Time'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string format

    # Create the start and end positions for each movement
    df['start'] = df[['Longitude', 'Latitude']].shift(1).apply(list, axis=1)  # Previous location as list [Longitude, Latitude]
    df = df.dropna()  # Drop rows where 'start' is NaN

    # Normalize time for color gradient
    df['time_norm'] = (df['Date & Time'].astype('datetime64[ns]') - df['Date & Time'].astype('datetime64[ns]').min()) / \
                      (df['Date & Time'].astype('datetime64[ns]').max() - df['Date & Time'].astype('datetime64[ns]').min())
    df['color'] = df['time_norm'].apply(lambda x: [255 * (1 - x), 128 * x, 255 * x, 200])  # RGBA gradient based on time

    # Define the line layer and scatterplot layer
    scatterplot = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[255, 140, 0],  # Static color for points
        get_radius=100,
        pickable=True,
    )

    line_layer = pdk.Layer(
        "LineLayer",
        data=df,
        get_source_position="start",  # The previous point as a list
        get_target_position=["Longitude", "Latitude"],  # The current point
        get_color="color",  # Use the color calculated based on time
        get_width=5,
        auto_highlight=True,
        pickable=True,
    )

    # Set the initial view state (zoom into the main area of activity)
    view_state = pdk.ViewState(
        latitude=df['Latitude'].mean(),
        longitude=df['Longitude'].mean(),
        zoom=11,
        pitch=50,
    )

    # Render the line tracking chart
    r = pdk.Deck(
        layers=[scatterplot, line_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Date & Time:</b> {date_time_str}<br>"
                    "<b>Latitude:</b> {Latitude}<br>"
                    "<b>Longitude:</b> {Longitude}"
        }
    )
    st.pydeck_chart(r)
