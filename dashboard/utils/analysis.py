from typing import Tuple, List
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

def categorize_time(hour):
    if 6 <= hour < 12:
        return 'Morning (06:00-12:00)'
    elif 12 <= hour < 18:
        return 'Afternoon (12:00-18:00)'
    else:
        return 'Evening/Night (18:00-06:00)'

@st.cache_data
def build_frequency_df(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Check if 'Date & Time' column exists before processing it
    if 'Date & Time' in data.columns:
        # Extract hour from 'Date & Time' column and create a new 'Hour' column
        data['Hour'] = pd.to_datetime(data['Date & Time']).dt.hour
        # Apply the function to create a 'Time of Day' column
        data['Time of Day'] = data['Hour'].apply(categorize_time)
    else:
        st.warning("The 'Date & Time' column is missing in the data.")
    
    # Get unique call types for multiselect
    call_types = data['Call Type'].unique() if 'Call Type' in data.columns else []

    return data, call_types


def plot_call_frequency_by_time(data: pd.DataFrame):
    st.subheader("Call Frequency by Phase of Day")
    st.write("This analysis will show the frequency of calls by phase of day (morning, afternoon, evening/night).") 
    # Build the frequency dataframe
    data, call_types = build_frequency_df(data)
    st.info("INFO: Please select the call types you want to analyze.")
    # Multiselect for call types
    selected_call_types = st.multiselect("Select Call Types", call_types, default="InComing")
    
    # Show a warning if no call type is selected
    if not selected_call_types:
        st.warning("Please select at least one Call Type.")
    else:
        # Filter data based on selection
        filtered_data = data[data['Call Type'].isin(selected_call_types)]
        # Group by 'Time of Day' and 'Call Type' and count the occurrences
        grouped_data = filtered_data.groupby(['Time of Day', 'Call Type']).size().unstack(fill_value=0)
        # Plot the data
        sns.set(style="darkgrid")
        grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title('Call Type Frequency by Phase of Day', color='white')
        plt.xlabel('Phase of Day', color='white')
        plt.ylabel('Frequency', color='white')
        plt.xticks(rotation=0, color='white')
        plt.yticks(color='white')
        plt.legend(title='Call Type')
        plt.tight_layout()
        # Change plot background and grid color for dark mode
        plt.gca().set_facecolor('#2e2e2e')
        plt.gcf().patch.set_facecolor('#2e2e2e')
        plt.grid(color='#444444')

        # Display the plot
        st.pyplot(plt)

def plot_longest_calls(data: pd.DataFrame):
    st.subheader(f"Longest Talk Times")
    st.write("Each bar represents a call with a duration in minutes. The x-axis shows the B-Party number and the y-axis shows the duration of the call. Hover over the bar to see the details.")
    
    # Filter data for call types "InComing" and "Outgoing"
    filtered_data = data[data['Call Type'].isin(['InComing', 'Outgoing'])]
    
    # Remove rows where B-Party number has length less than 9
    filtered_data = filtered_data[filtered_data['B-Party'].apply(lambda x: len(str(x)) >= 9)]
    
    # Convert Duration from seconds to minutes
    filtered_data['Duration'] = filtered_data['Duration']
    
    # Create a slider to select the number of longest calls to display, max 20, default 10
    num_calls = st.slider("Select the number of longest calls to display", min_value=1, max_value=20, value=10)
    longest_calls = filtered_data.sort_values(by='Duration', ascending=False).head(num_calls)

    # Add an identifier for each call
    longest_calls['Call ID'] = longest_calls.index
    
    # Create a 'Call Pair' column for better labeling
    longest_calls['Call Pair'] = longest_calls['A-Party'] + ' -> ' + longest_calls['B-Party']
    
    # Plot the data using Altair
    chart = alt.Chart(longest_calls, autosize="fit").mark_bar().encode(
        x=alt.X('B-Party:N', sort=None, title='B-Party Number', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Duration:Q', title='Duration (minutes)'),
            color=alt.Color('Call ID:N', legend=None),
            tooltip=['A-Party', 'B-Party', 'Duration', 'Call Pair']
        ).properties(
            width=900,
            height=600
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).interactive()

    st.altair_chart(chart)

def show_b_party_analysis(data: pd.DataFrame):
    st.subheader("Calendar Heatmap of Calls")
    st.write("The following analysis provides a detailed analysis for selected B-Party numbers and call types. It includes a calendar heatmap visualizing the distribution of calls over different days and months.")
    
    # Convert 'Date & Time' to datetime
    data['Date & Time'] = pd.to_datetime(data['Date & Time'])
    
    # Extract year, month, and date from 'Date & Time'
    data['Year'] = data['Date & Time'].dt.year
    data['Month'] = data['Date & Time'].dt.month
    data['Month-Year'] = data['Date & Time'].dt.to_period('M')
    data['Date'] = data['Date & Time'].dt.date
    
    # Get unique B-Party numbers and call types
    b_party_numbers = data['B-Party'].unique()
    call_types = data['Call Type'].unique()
    
    # Streamlit widgets for user selection
    selected_b_party = st.selectbox("Select B-Party Number", b_party_numbers)
    selected_call_type = st.selectbox("Select Call Type", call_types)
    
    # Filter data based on user selection
    filtered_data = data[(data['B-Party'] == selected_b_party) & (data['Call Type'] == selected_call_type)]
    
    if filtered_data.empty:
        st.warning("No {} found for the selected B-Party number {}".format(selected_call_type, selected_b_party))
    else:
        # Display the filtered data
        st.write(f"Calls for B-Party: {selected_b_party}, Call Type: {selected_call_type}")
        st.write(filtered_data)
        
        # Group data by date and month-year
        grouped_data = filtered_data.groupby(['Year', 'Month', 'Date']).size().reset_index(name='Number of Calls')
        
        # Remove months with zero calls
        month_year_counts = grouped_data.groupby(['Year', 'Month'])['Number of Calls'].sum().reset_index()
        non_zero_months = month_year_counts[month_year_counts['Number of Calls'] > 0]
        
        grouped_data = grouped_data.merge(non_zero_months[['Year', 'Month']], on=['Year', 'Month'])
        
        # Create a calendar heatmap using Altair
        heatmap = alt.Chart(grouped_data).mark_rect().encode(
            x=alt.X('yearmonth(Date):T', title='Month-Year'),
            y=alt.Y('date(Date):O', title='Day of Month'),
            color=alt.Color('Number of Calls:Q', scale=alt.Scale(scheme='viridis'), title='Number of Calls'),
            tooltip=['Year', 'Month', 'Date', 'Number of Calls']
        ).properties(
            width=800,
            height=400,
            title='Calendar Heatmap of Calls'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        ).interactive()
        
        st.altair_chart(heatmap, use_container_width=True)



def analyze_b_party(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function analyzes the 'B-Party' column for known phone numbers and provides
    statistics on call frequency, call locations, and call durations.
    
    Args:
    - df (pd.DataFrame): DataFrame containing call data.
    
    Returns:
    - pd.DataFrame: DataFrame containing statistics for each known phone number.
    """
    # Filter out rows where 'B-Party' is empty or unknown (assuming 'Unknown' or empty values mean unknown)
    df_filtered = df[df['B-Party'].notna() & (df['B-Party'] != 'Unknown')]
    
    # Group the data by 'B-Party' to calculate the desired metrics
    b_party_stats = df_filtered.groupby('B-Party').agg(
        Call_Frequency=('B-Party', 'size'),  # Count the frequency of each 'B-Party'
        Avg_Call_Duration=('Duration', 'mean'),  # Calculate the average duration of calls
        Total_Call_Duration=('Duration', 'sum'),  # Calculate the total duration of calls
        Locations=('Address', lambda x: ', '.join(x.dropna().astype(str).unique())),  # Concatenate unique locations (addresses)
        Latitude=('Latitude', lambda x: x.mean()),  # Get the average latitude (for central location)
        Longitude=('Longitude', lambda x: x.mean())  # Get the average longitude (for central location)
    ).reset_index()

    # Rename columns for clarity
    b_party_stats.columns = ['B-Party', 'Call Frequency', 'Avg Call Duration (s)', 'Total Call Duration (s)', 'Call Locations', 'Avg Latitude', 'Avg Longitude']
    
    return b_party_stats



def display_b_party_analysis(df: pd.DataFrame) -> None:
    """
    Streamlit function to display the B-Party analysis in the app.
    """
    st.subheader("B-Party Analysis")
    st.write("This section provides insights into the known phone numbers (B-Party) "
             "in terms of call frequency, locations, and durations.")
    
    # Check if the DataFrame has the necessary columns
    required_columns = ['B-Party', 'Duration', 'Address', 'Latitude', 'Longitude']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns: {', '.join([col for col in required_columns if col not in df.columns])}")
        return
    
    # Perform the B-Party analysis
    b_party_stats = analyze_b_party(df)
    
    if b_party_stats.empty:
        st.warning("No known phone numbers (B-Party) found in the dataset.")
    else:
        # Display the data in a table format
        st.dataframe(b_party_stats)
        
        # Optionally, visualize some aspects (e.g., call frequency)
        st.subheader("Call Frequency by B-Party")
        b_party_stats_chart = alt.Chart(b_party_stats).mark_bar().encode(
            x=alt.X('B-Party', sort='-y', title='Phone Number'),
            y=alt.Y('Call Frequency', title='Frequency of Calls'),
            tooltip=['B-Party', 'Call Frequency', 'Avg Call Duration (s)', 'Total Call Duration (s)', 'Call Locations']
        ).properties(
            width=700,
            height=400,
            title="Call Frequency for Known B-Parties"
        )
        st.altair_chart(b_party_stats_chart, use_container_width=True)