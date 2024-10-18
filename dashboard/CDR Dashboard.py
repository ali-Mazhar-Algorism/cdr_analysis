import streamlit as st
import pandas as pd
from utils.data_helper import load_data, transform_data
from utils.geo_map import show_line_tracking_chart, show_heat_map, show_scatter_map, show_density_map
from utils.constants import sections, scatter_map_guide, required_columns
from utils.analysis import analyze_b_party, display_b_party_analysis, display_dataset_highlights, plot_call_frequency_by_time, plot_longest_calls, show_b_party_analysis
from utils.data_processing import *

# Set page configuration
st.set_page_config(page_title="CDR Dashboard", layout="wide")

if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False   

# File uploader
if not st.session_state['file_uploaded']:
    uploaded_file = st.file_uploader("Choose a file :file_folder:")
    
    if uploaded_file is not None:
        # Load the data
        df = None
        df = load_data(uploaded_file)
        
        # Handle non-relevant rows
        df = find_table_start(df)
        df = preprocess_file(df)
        
        # Standardize column names
        df = standardize_columns(df)
        
        # Prune columns
        df = prune_columns(df)
        st.write(df)
        
        # Transform data and format phone numbers
        df = transform_data(df)
        df['A-Party'] = df['A-Party'].apply(format_phone_number)
        
        # Store the data in session state
        st.session_state['df'] = df
        st.session_state['file_uploaded'] = True
        st.success(f"{uploaded_file.name} uploaded and data loaded successfully!")


if st.session_state['file_uploaded']:
    # Display the title
    
    dataset = st.session_state['df']
    
    display_dataset_highlights(dataset)

    st.title("CDR Dashboard: maps")
    
    if required_columns.issubset(dataset.columns):
        tab1, tab2, tab3, tab4 = st.tabs(["Geolocation Map", "Density Map", "Heat Map", "Location Tracking"])
        # Geolocation Map tab
        with tab1:
            st.title("Geolocation Map :clock1:")
            st.markdown(scatter_map_guide)
            show_scatter_map(dataset)
        
        # Density Map tab
        with tab2:
            st.title("Density Map :bar_chart:")
            st.markdown('\n\n')
            show_density_map(dataset)
        
        # Heat Map tab
        with tab3:
            st.title("Heat Map :fire:")
            st.markdown('\n\n')
            show_heat_map(dataset)

        # Location Tracking Chart
        with tab4:
            st.title("Location Tracking Chart: :round_pushpin:")
            st.markdown('\n\n')
            show_line_tracking_chart(dataset)
            
        
    else:
        st.error(
            "The dataframe is missing required columns: 'Date & Time', 'Latitude', or 'Longitude'."
        )
        
    # elif page == "Analysis":
    st.title("Analysis :bar_chart:")
    plot_call_frequency_by_time(dataset)
    plot_longest_calls(dataset)
    show_b_party_analysis(dataset)
    # top_locations(st.session_state['df'])

    


