import streamlit as st
import pandas as pd
from utils.data_helper import load_data
from utils.geo_map import (
    show_heat_map,
    show_location_tracking_chart,
    show_scatter_map,
    show_density_map,
    show_time_bound_map,
)
from utils.constants import sections, scatter_map_guide, required_columns
from utils.analysis import (
    display_dataset_highlights,
    handle_multi_file_analysis,
    plot_call_frequency_by_time,
    plot_longest_calls,
    show_b_party_analysis,
    two_file_comparative_analysis,
)
from utils.data_processing import *

# Set page configuration
st.set_page_config(page_title="CDR Dashboard", layout="wide")

# Sidebar for analysis type selection
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type", ["Single File Analysis", "Two File Analysis", "Multi-File Analysis"]
)
rerun = None
if analysis_type == "Single File Analysis":
    # Single file analysis code
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False
    else:
        rerun = st.button("Rerun")

    if rerun:
        st.session_state.clear()
        st.rerun()

    if not st.session_state["file_uploaded"]:
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=["xlsx", "xls", "csv"],
            disabled=st.session_state["file_uploaded"],
        )

        if uploaded_file is not None:
            # Load the data
            df = load_data(uploaded_file)

            # Handle non-relevant rows, preprocess, and transform data
            df = find_table_start(df)
            df = preprocess_file(df)

            # Store the data in session state
            st.session_state["df"] = df
            st.session_state["file_uploaded"] = True  # Mark file as uploaded
            st.success(f"{uploaded_file.name} uploaded and data loaded successfully!")

    if st.session_state["file_uploaded"]:
        dataset = st.session_state["df"]
        display_dataset_highlights(dataset)

        st.title("CDR Dashboard: maps")

        if required_columns.issubset(dataset.columns):
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Geolocation Map", "Density Map", "Heat Map", "Location Tracking", "Time Tracking"]
            )

            with tab1:
                st.title("Geolocation Map :clock1:")
                st.markdown(scatter_map_guide)
                show_scatter_map(dataset)

            with tab2:
                st.title("Density Map :bar_chart:")
                show_density_map(dataset)

            with tab3:
                st.title("Heat Map :fire:")
                show_heat_map(dataset)

            with tab4:
                st.title("Location Tracking Chart: :round_pushpin:")
                show_location_tracking_chart(dataset)
                
            with tab5:
                st.title("Time Tracking Map")
                show_time_bound_map(dataset)

        else:
            st.error(
                "The dataframe is missing required columns: 'Date & Time', 'Latitude', or 'Longitude'."
            )

        st.title("Analysis :bar_chart:")
        plot_call_frequency_by_time(dataset)
        plot_longest_calls(dataset)
        show_b_party_analysis(dataset)

elif analysis_type == "Two File Analysis":
    # Two file analysis code
    st.title("Two File Analysis")
    
    if "file_uploaded" not in st.session_state:
        st.session_state["file_uploaded"] = False
    else:
        rerun = st.button("Rerun")

    if rerun:
        st.session_state.clear()
        st.rerun()

    uploaded_file1 = st.file_uploader(
        "Choose the first file :file_folder:", key="file1"
    )
    uploaded_file2 = st.file_uploader(
        "Choose the second file :file_folder:", key="file2"
    )

    if uploaded_file1 and uploaded_file2:
        # Load both files
        df1 = load_data(uploaded_file1)
        df2 = load_data(uploaded_file2)

        # Preprocess each file
        df1 = find_table_start(df1)
        df1 = preprocess_file(df1)

        df2 = find_table_start(df2)
        df2 = preprocess_file(df2)

        st.success(
            f"{uploaded_file1.name} and {uploaded_file2.name} uploaded and data loaded successfully!"
        )

        # Placeholder for comparison analysis
        two_file_comparative_analysis(df1, df2)

elif analysis_type == "Multi-File Analysis":
    handle_multi_file_analysis()