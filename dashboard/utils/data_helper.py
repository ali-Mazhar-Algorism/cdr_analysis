import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    file_extension = os.path.splitext(filepath.name)[1]
    
    if file_extension == '.csv':
        data = pd.read_csv(filepath)
    else:
        data = pd.read_excel(filepath)
    
    st.session_state["raw_data"] = data
    return data

@st.cache_data
def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing address
    # data.dropna(inplace=True, subset=["Address"])
    data['A-Party'] = data['A-Party'].astype(str)
    data['B-Party'] = data['B-Party'].astype(str)
    st.session_state["transformed_data"] = data
    return data