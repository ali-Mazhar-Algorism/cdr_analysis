import streamlit as st
import pandas as pd
from utils.data_helper import load_data, transform_data
from utils.geo_map import show_line_tracking_chart, show_heat_map, show_map, show_time_bound_map, top_locations, show_density_map
from utils.constants import sections, geo_markdown
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
        df = load_data(uploaded_file)
        
        # Handle non-relevant rows
        df = find_table_start(df)
        
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
    
    display_dataset_highlights(st.session_state['df'])
    
    # Create tabs for each map
    st.title("CDR Dashboard: maps")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Geolocation Map", "Density Map", "Heat Map", "Time Bound Map", "Location Tracking"])
    
    # Geolocation Map tab
    with tab1:
        st.title("Geolocation Map :round_pushpin:")
        st.markdown(geo_markdown)
        show_map(st.session_state['df'])
    
    # Density Map tab
    with tab2:
        st.title("Density Map :bar_chart:")
        st.markdown('\n\n')
        show_density_map(st.session_state['df'])
    
    # Heat Map tab
    with tab3:
        st.title("Heat Map :fire:")
        st.markdown('\n\n')
        show_heat_map(st.session_state['df'])
    
    # Time Bound Map tab
    with tab4:
        st.title("Time Bound Map :clock1:")
        st.markdown('\n\n')
        show_time_bound_map(st.session_state['df'])

    # Location Tracking Chart
    with tab5:
        st.title("Location Tracking Chart:")
        st.markdown('\n\n')
        show_line_tracking_chart(st.session_state['df'])
        
        
    # elif page == "Analysis":
    st.title("Analysis :bar_chart:")
    plot_call_frequency_by_time(st.session_state['df'])
    analyze_b_party(st.session_state['df'])
    display_b_party_analysis(st.session_state['df'])
    plot_longest_calls(st.session_state['df'])
    show_b_party_analysis(st.session_state['df'])
    # top_locations(st.session_state['df'])

    


