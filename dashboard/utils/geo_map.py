import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pydeck as pdk
from utils.constants import *


def select_call_types(map_data: pd.DataFrame, key: str) -> list:
    """
    Displays a multiselect dropdown for users to select call types from the provided map data.

    Parameters:
    - map_data (pd.DataFrame): The DataFrame containing call data.
    - default (str): The default call type to be selected.
    - key (str): The unique key for the Streamlit widget.

    Returns:
    - list: A list of selected call types.
    """
    unique_call_types = map_data["Call Type"].unique().tolist()
    selected_types = st.multiselect(
        "Select Call Types:", unique_call_types, default=unique_call_types[0], key=key
    )
    return selected_types


def select_date_range(data: pd.DataFrame, start_key: str, end_key: str, latest_only=False):
    """
    Displays date input fields for users to select a date range based on the provided data.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing date information.
    - start_key (str): The unique key for the start date input.
    - end_key (str): The unique key for the end date input.

    Returns:
    - tuple: A tuple containing the selected start and end dates.
    """
    
    min_date = data["Date & Time"].min().date()
    max_date = data["Date & Time"].max().date()
    
    end=max_date
    
    if latest_only:
        start = max(max_date - pd.Timedelta(days=2), min_date)
    else:
        start = min_date
    start_date = st.date_input(
        "Select start date",
        start,
        min_value=min_date,
        max_value=max_date,
        key=start_key,
    )
    end_date = st.date_input(
        "Select end date",
        end,
        min_value=min_date,
        max_value=max_date,
        key=end_key,
    )
    return start_date, end_date


def calculate_address_density(data: pd.DataFrame, selected_types) -> pd.DataFrame:
    """
    Calculates the density of calls at each address based on the selected call types.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing call data.
    - selected_types (list): The list of selected call types.

    Returns:
    - pd.DataFrame: A DataFrame containing the top ten addresses with the highest density.
    """
    filtered_data = data[data["Call Type"].isin(selected_types)]
    address_density = (
        filtered_data.groupby(["Address", "Latitude", "Longitude"])
        .size()
        .reset_index(name="Density")
    )
    return address_density.nlargest(10, "Density")


def assign_colors_by_density(densities):
    """
    Assigns colors to data points based on their density using a gradient from blue to red.

    Parameters:
    - densities (pd.Series): A series containing density values.

    Returns:
    - list: A list of colors corresponding to the density values.
    """
    if densities.empty:
        return []

    # Proceed if densities are not empty
    max_density, min_density = max(densities), min(densities)

    # If all densities are the same (max_density == min_density), assign a neutral color
    color_range = [
        (
            255 * (density - min_density) / (max_density - min_density)
            if max_density != min_density
            else 0.5
        )
        for density in densities
    ]
    
    return [[int(red), int(255 * (1 - red / 255)), 0, 160] for red in color_range]



def render_map_with_layer(data: pd.DataFrame, layer: pdk.Layer, tooltip_template: str):
    """
    Renders a map with the specified layer and tooltip template using PyDeck.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing data to be displayed on the map.
    - layer (pdk.Layer): The layer to be rendered on the map.
    - tooltip_template (str): The HTML template for the tooltip.
    """
    view_state = pdk.ViewState(
        latitude=data["Latitude"].median(),
        longitude=data["Longitude"].median(),
        zoom=10.5,
        pitch=0,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip_template,
    )
    st.pydeck_chart(r)


def show_bubble_map(data: pd.DataFrame, selected_types) -> None:
    """
    Displays a bubble map representing the density of calls at various addresses.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing call data.
    - selected_types (list): The list of selected call types.
    """
    bubble_map_data = pd.DataFrame()
    bubble_map_data["Latitude"] = data["Latitude"].astype(float)
    bubble_map_data["Longitude"] = data["Longitude"].astype(float)
    bubble_map_data["Address"] = data["Address"].astype(str)
    bubble_map_data["Call Type"] = data["Call Type"].astype(str)

    top_ten_addresses = calculate_address_density(bubble_map_data, selected_types)
    st.dataframe(top_ten_addresses)

    top_ten_addresses["color"] = assign_colors_by_density(top_ten_addresses["Density"])

    bubble_layer = pdk.Layer(
        "ScatterplotLayer",
        top_ten_addresses,
        get_position=["Longitude", "Latitude"],
        get_radius="Density * 3",
        get_fill_color="color",
        get_line_color=[0, 0, 0],
        pickable=True,
    )

    render_map_with_layer(
        top_ten_addresses,
        bubble_layer,
        {"html": "<b>Address:</b> {Address}<br><b>Density:</b> {Density}"},
    )


def prepare_map_data(
    data: pd.DataFrame, selected_types, start_date, end_date
) -> pd.DataFrame:
    """
    Prepares the map data by filtering based on selected call types and date range.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing call data.
    - selected_types (list): The list of selected call types.
    - start_date (datetime.date): The start date for filtering.
    - end_date (datetime.date): The end date for filtering.

    Returns:
    - pd.DataFrame: The filtered DataFrame ready for mapping.
    """
    data = data[data["Date & Time"].notna()]
    data["Date"], data["Time"] = zip(
        *data["Date & Time"].astype(str).apply(lambda x: x.split(" "))
    )

    data = data[data["Call Type"].isin(selected_types)]

    data = data[
        (data["Date & Time"].dt.date >= start_date)
        & (data["Date & Time"].dt.date <= end_date)
    ]

    color_mapping = {
        "InComing": [255, 0, 0, 255],  # Red
        "Outgoing": [0, 0, 255, 255],  # Blue
        "InComing SMS": [0, 128, 0, 255],  # Green
        "OutGoing SMS": [255, 165, 0, 255],  # Orange
    }
    data["color"] = data["Call Type"].map(
        lambda x: color_mapping.get(x, [255, 255, 255, 160])
    )

    latest_entry_index = data["Date & Time"].idxmax()
    data["radius"] = 120
    data.loc[latest_entry_index, "radius"] = 250

    return data


def show_scatter_map(df: pd.DataFrame) -> None:
    """
    Displays a scatter map based on the selected call types and date range.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing call data.
    """
    selected_types = select_call_types(df, key="scatter_map")
    start_date, end_date = select_date_range(
        df, "scater_map_date_start", "scater_map_date_end"
    )

    if not selected_types:
        st.warning(no_selected_type_warning)
        return

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    scatter_map_data = pd.DataFrame()
    scatter_map_data["Latitude"] = df["Latitude"].astype(float)
    scatter_map_data["Longitude"] = df["Longitude"].astype(float)
    scatter_map_data["B-Party"] = df["B-Party"].astype(str)
    scatter_map_data["Call Type"] = df["Call Type"].astype(str)
    scatter_map_data["Date & Time"] = pd.to_datetime(df["Date & Time"])
    scatter_map_data["Duration"] = df["Duration"]
    scatter_map_data["Address"] = df["Address"].astype(str)
    map_data = prepare_map_data(scatter_map_data, selected_types, start_date, end_date)

    if map_data.empty:
        st.warning("No data available for the selected date range.")
        return

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["Longitude", "Latitude"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 255],
    )
    render_map_with_layer(
        map_data,
        scatter_layer,
        {
            "html": "<b>B-Party:</b> {B-Party}<br><b>Call Type:</b> {Call Type}<br>"
            "<b>Date:</b> {Date}<br><b>Time:</b> {Time}<br>"
            "<b>Duration (secs):</b> {Duration}<br><b>Address:</b> {Address}",
        },
    )


def show_density_map(df: pd.DataFrame) -> None:
    """
    Displays a density map based on the selected call types.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing call data.
    """
    selected_types = select_call_types(df, key="density_map")

    if not selected_types:
        st.warning(no_selected_type_warning)

    density_map_data = pd.DataFrame()
    density_map_data["Latitude"] = df["Latitude"].astype(float)
    density_map_data["Longitude"] = df["Longitude"].astype(float)
    density_map_data["Address"] = df["Address"].astype(str)
    density_map_data["Call Type"] = df["Call Type"].astype(str)
    density_map_data = density_map_data[
        density_map_data["Call Type"].isin(selected_types)
    ]
    top_ten_addresses = calculate_address_density(
        density_map_data, df["Call Type"].unique()
    )
    st.dataframe(top_ten_addresses)
    top_ten_addresses["color"] = assign_colors_by_density(top_ten_addresses["Density"])
    density_layer = pdk.Layer(
        "ScatterplotLayer",
        top_ten_addresses,
        get_position=["Longitude", "Latitude"],
        get_radius="Density * 3",
        get_fill_color="color",
        get_line_color=[0, 0, 0],
        pickable=True,
    )
    render_map_with_layer(
        top_ten_addresses,
        density_layer,
        {"html": "<b>Address:</b> {Address}<br><b>Density:</b> {Density}"},
    )


def show_heat_map(df: pd.DataFrame) -> None:
    """
    Displays a heat map based on the selected call types.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing call data.
    """
    selected_types = select_call_types(df, key="heat_map")

    if not selected_types:
        st.warning(no_selected_type_warning)

    heatmap_data = pd.DataFrame()
    heatmap_data["Latitude"] = df["Latitude"].astype(float)
    heatmap_data["Longitude"] = df["Longitude"].astype(float)
    heatmap_data["Call Type"] = df["Call Type"].astype(str)
    heatmap_data["Date & Time"] = pd.to_datetime(df["Date & Time"])
    heatmap_data["date_time_str"] = heatmap_data["Date & Time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    heatmap_data = heatmap_data[heatmap_data["Call Type"].isin(selected_types)]

    layer = pdk.Layer(
        "HeatmapLayer",
        data=heatmap_data,
        get_position=["Longitude", "Latitude"],
        aggregation="MEAN",
        radiusPixels=60,
        opacity=0.6,
    )
    render_map_with_layer(
        heatmap_data,
        layer,
        {
            "html": "<b>Call Type:</b> {Call Type}<br>"
            "<b>Date & Time:</b> {date_time_str}<br>"
            "<b>Latitude:</b> {Latitude}<br>"
            "<b>Longitude:</b> {Longitude}"
        },
    )


def show_location_tracking_chart(df: pd.DataFrame) -> None:
    st.markdown(line_tracking_map_guide)

    map_data = pd.DataFrame()
    map_data["Latitude"] = df["Latitude"].astype(float)
    map_data["Longitude"] = df["Longitude"].astype(float)
    map_data["B-Party"] = df["B-Party"].astype(str)
    map_data["Call Type"] = df["Call Type"].astype(str)
    map_data["Date & Time"] = pd.to_datetime(df["Date & Time"])
    map_data["Duration"] = df["Duration"]
    map_data["Address"] = df["Address"].astype(str)

    start_date, end_date = select_date_range(
        map_data, start_key="LocationMapDateStart", end_key="LocationMapDateEnd", latest_only=True
    )

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    # Filter for valid entries in the required columns
    df_filtered = map_data[
        (map_data["Date & Time"].dt.date >= start_date) &
        (map_data["Date & Time"].dt.date <= end_date) &
        map_data["Latitude"].notna() &
        map_data["Longitude"].notna() &
        map_data["Date & Time"].notna()
    ]

    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    df_filtered["start"] = df_filtered[["Longitude", "Latitude"]].shift(1).apply(list, axis=1)
    df_filtered = df_filtered.dropna().iloc[1:]

    # Normalize 'Date & Time' for color scaling
    time_min, time_max = df_filtered["Date & Time"].min(), df_filtered["Date & Time"].max()
    df_filtered["time_norm"] = (df_filtered["Date & Time"] - time_min) / (time_max - time_min)

    df_filtered["source_color"] = df_filtered["time_norm"].apply(
        lambda x: [int(255 * x), int(0 * (1 - x)), int(255 * (1 - x)), 160]
    )
    df_filtered["target_color"] = df_filtered["time_norm"].apply(
        lambda x: [int(255 * (1 - x)), int(255 * x), int(0 * x), 160]
    )

    df_filtered["date_time_str"] = df_filtered["Date & Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    scatterplot = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtered,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[255, 140, 0],
        get_radius=100,
        pickable=True,
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=df_filtered,
        get_source_position="start",
        get_target_position=["Longitude", "Latitude"],
        get_source_color="source_color",
        get_target_color="target_color",
        get_width=4,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=df_filtered["Latitude"].mean(),
        longitude=df_filtered["Longitude"].mean(),
        zoom=11,
        pitch=50,
    )

    r = pdk.Deck(
        layers=[scatterplot, arc_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Date & Time:</b> {date_time_str}<br>"
                    "<b>Latitude:</b> {Latitude}<br>"
                    "<b>Longitude:</b> {Longitude}"
        },
    )

    st.pydeck_chart(r)


def show_time_bound_map(df: pd.DataFrame) -> None:
    # Filter data based on selected call types
    unique_call_types = df["Call Type"].unique().tolist()
    st.info("INFO: Select the call types from the dropdown to display on the map.")
    selected_types = st.multiselect(
        "Select Call Types:",
        unique_call_types,
        default="OutGoing",
        key="time_bound_map",
    )
    map_data = df[df["Call Type"].isin(selected_types)]
    map_data["date_time_str"] = map_data["Date & Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure 'Date & Time' column is in datetime format
    map_data["Date & Time"] = pd.to_datetime(map_data["Date & Time"])

    # Define the minimum and maximum dates in the dataset
    min_date = map_data["Date & Time"].min()
    max_date = map_data["Date & Time"].max()

    # Create a time range slider in Streamlit for users to select the time frame
    st.info(
        "INFO: Adjust the slider to select the time range for displaying locations."
    )
    selected_time_range = st.slider(
        "Select Date & Time Range:",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        format="YYYY-MM-DD HH:mm:ss",
    )

    # Filter data based on the selected time range
    time_filtered_data = map_data[
        (map_data["Date & Time"] >= pd.to_datetime(selected_time_range[0]))
        & (map_data["Date & Time"] <= pd.to_datetime(selected_time_range[1]))
    ]

    # Check if the filtered dataset has any data
    if time_filtered_data.empty:
        st.warning("No data available for the selected time range.")
        return

    # Create a new DataFrame for plotting with necessary columns
    plot_data = pd.DataFrame()
    plot_data["lat"] = time_filtered_data["Latitude"].astype(float)
    plot_data["lon"] = time_filtered_data["Longitude"].astype(float)
    plot_data["Call Type"] = time_filtered_data["Call Type"].astype(str)
    plot_data["Date & Time"] = time_filtered_data["Date & Time"]

    # Define the ScatterplotLayer for the time-bound map
    layer = pdk.Layer(
        "ScatterplotLayer",
        plot_data,
        get_position=["lon", "lat"],
        get_radius=120,  # Adjust the radius if needed
        get_fill_color=[255, 140, 0, 160],  # Orange color for all points
        pickable=True,
    )

    # Set the viewport location based on the filtered data
    view_state = pdk.ViewState(
        latitude=plot_data["lat"].median(),
        longitude=plot_data["lon"].median(),
        zoom=11,
        pitch=0,
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
        },
    )
    st.pydeck_chart(r)