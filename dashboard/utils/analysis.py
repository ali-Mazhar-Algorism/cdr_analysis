from typing import Tuple, List
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning (06:00-12:00)"
    elif 12 <= hour < 18:
        return "Afternoon (12:00-18:00)"
    else:
        return "Evening/Night (18:00-06:00)"


@st.cache_data
def build_frequency_df(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Check if 'Date & Time' column exists before processing it
    if "Date & Time" in data.columns:
        # Extract hour from 'Date & Time' column and create a new 'Hour' column
        data["Hour"] = pd.to_datetime(data["Date & Time"]).dt.hour
        # Apply the function to create a 'Time of Day' column
        data["Time of Day"] = data["Hour"].apply(categorize_time)
    else:
        st.warning("The 'Date & Time' column is missing in the data.")

    # Get unique call types for multiselect
    call_types = data["Call Type"].unique() if "Call Type" in data.columns else []

    return data, call_types


def display_call_duration_per_day(df: pd.DataFrame) -> None:
    """
    Streamlit function to display a bar chart of total call duration per day with interactive tooltips.
    """
    # Ensure 'Date & Time' is in datetime format

    st.subheader("Call Frequency by Dates")

    df["Date & Time"] = pd.to_datetime(df["Date & Time"])

    call_types = df["Call Type"].unique().tolist()
    # Multiselect for call types
    selected_call_types = st.multiselect(
        "Select Call Types", call_types, default="InComing", key="multi_day"
    )

    # Show a warning if no call type is selected
    if not selected_call_types:
        st.warning("Please select at least one Call Type.")
        return

    # Filter data based on selected call types
    filtered_data = df[df["Call Type"].isin(selected_call_types)]

    cdr_start = filtered_data["Date & Time"].min()
    cdr_end = filtered_data["Date & Time"].max()

    start_date = st.date_input(
        "Select start date",
        cdr_start,
        min_value=cdr_start,
        max_value=cdr_end,
        key="CallDataDateStart",
    )
    end_date = st.date_input(
        "Select end date",
        cdr_end,
        min_value=cdr_start,
        max_value=cdr_end,
        key="CallDataDateEnd",
    )

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return
    # Create a full date range
    filtered_date_range = pd.date_range(start=start_date, end=end_date).date

    # Group by date and sum the 'Duration' column
    filtered_data["Date"] = filtered_data["Date & Time"].dt.date
    daily_duration = filtered_data.groupby("Date")["Duration"].sum().reset_index()

    # Find missing dates and append them to daily_duration with 0 duration
    all_dates = pd.DataFrame(filtered_date_range, columns=["Date"])
    daily_duration = pd.merge(all_dates, daily_duration, on="Date", how="left").fillna(
        0
    )

    # Rename 'Duration' for clarity
    daily_duration = daily_duration.rename(
        columns={"Duration": "Total Call Duration (s)"}
    )

    if daily_duration.empty:
        st.warning("No call data available.")
    else:
        # Plot the data using Altair
        call_duration_chart = (
            alt.Chart(daily_duration)
            .mark_bar()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y(
                    "Total Call Duration (s):Q", title="Total Call Duration (seconds)"
                ),
                tooltip=["Date:T", "Total Call Duration (s):Q"],
            )
            .properties(width=700, height=400, title="Total Call Duration per Day")
            .configure_axis(labelColor="white", titleColor="white")
            .configure_title(color="white")
            .configure_view(strokeWidth=0, stroke=None)
            .configure(background="#2e2e2e")
        )

        st.altair_chart(call_duration_chart, use_container_width=True)


def display_call_frequency_by_phase(df: pd.DataFrame) -> None:
    """
    Streamlit function to display the call frequency by phase of day and call type.
    """
    st.subheader("Call Frequency by Phase of Day")
    st.write(
        "This analysis will show the frequency of calls by phase of day (morning, afternoon, evening/night)."
    )

    # Ensure 'Date & Time' is in datetime format
    df["Date & Time"] = pd.to_datetime(df["Date & Time"])

    # Define time-of-day phases
    def get_phase_of_day(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Evening/Night"

    # Add a 'Phase of Day' column
    df["Phase of Day"] = df["Date & Time"].dt.hour.apply(get_phase_of_day)

    # Extract unique call types for selection
    call_types = df["Call Type"].unique().tolist()

    # Inform the user to select call types
    st.info("INFO: Please select the call types you want to analyze.")

    # Multiselect for call types
    selected_call_types = st.multiselect(
        "Select Call Types", call_types, default="InComing", key="multi_day_phase"
    )

    # Show a warning if no call type is selected
    if not selected_call_types:
        st.warning("Please select at least one Call Type.")
        return

    # Filter data based on selected call types
    filtered_data = df[df["Call Type"].isin(selected_call_types)]

    # Group by 'Phase of Day' and 'Call Type' and count occurrences
    phase_call_count = (
        filtered_data.groupby(["Phase of Day", "Call Type"])
        .size()
        .reset_index(name="Count")
    )

    # If the data is empty after filtering, show a warning
    if phase_call_count.empty:
        st.warning("No data available for the selected call types.")
        return

    # Create a bar chart using Altair
    phase_of_day_chart = (
        alt.Chart(phase_call_count)
        .mark_bar()
        .encode(
            x=alt.X(
                "Phase of Day:N",
                title="Phase of Day",
                sort=["Morning", "Afternoon", "Evening/Night"],
            ),
            y=alt.Y("Count:Q", title="Frequency of Calls"),
            color=alt.Color("Call Type:N", title="Call Type"),
            tooltip=["Phase of Day:N", "Call Type:N", "Count:Q"],
        )
        .properties(width=700, height=400, title="Call Frequency by Phase of Day")
        .configure_axis(labelColor="white", titleColor="white")
        .configure_title(color="white")
        .configure_view(strokeWidth=0, stroke=None)
        .configure_legend(labelColor="white", titleColor="white")
        .configure(background="#2e2e2e")
    )

    # Display the chart
    st.altair_chart(phase_of_day_chart, use_container_width=True)


def plot_call_frequency_by_time(data: pd.DataFrame):
    # Missing Dates
    if "Date & Time" in data.columns:

        display_call_duration_per_day(data)
        display_call_frequency_by_phase(data)


def plot_longest_calls(data: pd.DataFrame):
    st.subheader(f"Longest Talk Times")
    st.write(
        "Each bar represents a call with a duration in minutes. The x-axis shows the B-Party number and the y-axis shows the duration of the call. Hover over the bar to see the details."
    )

    # Filter data for call types "InComing" and "Outgoing"
    filtered_data = data[data["Call Type"].isin(["InComing", "OutGoing"])]

    # Remove rows where B-Party number has length less than 9
    filtered_data = filtered_data[
        filtered_data["B-Party"].apply(lambda x: len(str(x)) >= 9)
    ]

    # Convert Duration from seconds to minutes
    filtered_data["Duration"] = filtered_data["Duration"]

    # Create a slider to select the number of longest calls to display, max 20, default 10
    num_calls = st.slider(
        "Select the number of longest calls to display",
        min_value=1,
        max_value=20,
        value=10,
    )
    longest_calls = filtered_data.sort_values(by="Duration", ascending=False).head(
        num_calls
    )

    # Add an identifier for each call
    longest_calls["Call ID"] = longest_calls.index

    # Create a 'Call Pair' column for better labeling
    longest_calls["Call Pair"] = (
        longest_calls["A-Party"] + " -> " + longest_calls["B-Party"]
    )

    # Plot the data using Altair
    chart = (
        alt.Chart(longest_calls, autosize="fit")
        .mark_bar()
        .encode(
            x=alt.X(
                "B-Party:N",
                sort=None,
                title="B-Party Number",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("Duration:Q", title="Duration (minutes)"),
            color=alt.Color("Call ID:N", legend=None),
            tooltip=["A-Party", "B-Party", "Duration", "Call Pair"],
        )
        .properties(width=900, height=600)
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .interactive()
    )

    st.altair_chart(chart)


def show_b_party_analysis(data: pd.DataFrame):
    st.title("B-Party Analysis")
    st.write(
        "The following analysis provides a detailed breakdown for selected B-Party numbers and call types. It visualizes the number of calls and the total call duration over time."
    )

    display_b_party_analysis(data)

    # Convert 'Date & Time' to datetime
    data["Date & Time"] = pd.to_datetime(data["Date & Time"])

    # Extract year, month, and date from 'Date & Time'
    data["Year"] = data["Date & Time"].dt.year
    data["Month"] = data["Date & Time"].dt.month
    data["Month-Year"] = data["Date & Time"].dt.to_period("M")
    data["Date"] = data["Date & Time"].dt.date

    # Get unique B-Party numbers and call types
    b_party_numbers = data["B-Party"].unique()
    call_types = data["Call Type"].unique()

    # Streamlit widgets for user selection
    selected_b_party = st.multiselect(
        "Select B-Party Numbers", b_party_numbers
    )  # Multiselect for B-Party numbers
    selected_call_type = st.multiselect(
        "Select Call Type", call_types, default="OutGoing", key="multiselect_b_party"
    )

    if not selected_b_party:
        st.warning("Please select at least one B-Party number.")
        return

    # Filter data based on user selection
    filtered_data = data[
        (data["B-Party"].isin(selected_b_party))
        & (data["Call Type"].isin(selected_call_type))
    ]

    if filtered_data.empty:
        st.warning(f"No calls found for the selected B-Party numbers and call types.")
    else:
        # Display the filtered data
        st.write(f"Calls for selected B-Party numbers, Call Type: {selected_call_type}")
        st.write(filtered_data)

        # Group data by date to get the number of calls and total call duration
        grouped_data = (
            filtered_data.groupby(["Date"])
            .agg(
                Number_of_Calls=("Date & Time", "size"),
                Total_Call_Duration=("Duration", "sum"),
            )
            .reset_index()
        )

        # Create a bar chart for the number of calls
        bar_chart = (
            alt.Chart(grouped_data)
            .mark_bar()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Number_of_Calls:Q", title="Number of Calls"),
                tooltip=["Date:T", "Number_of_Calls:Q", "Total_Call_Duration:Q"],
            )
            .properties(width=800, height=400, title="Number of Calls per Day")
        )

        # Create a line chart for the total call duration
        line_chart = (
            alt.Chart(grouped_data)
            .mark_line(color="red")
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Total_Call_Duration:Q", title="Total Call Duration (seconds)"),
                tooltip=["Date:T", "Number_of_Calls:Q", "Total_Call_Duration:Q"],
            )
        )

        # Combine the bar and line charts
        combined_chart = (
            alt.layer(bar_chart, line_chart)
            .resolve_scale(
                y="independent"  # Independent y-axis for the line and bar chart
            )
            .configure_axis(labelFontSize=12, titleFontSize=14)
            .configure_title(fontSize=16)
            .interactive()
        )

        st.altair_chart(combined_chart, use_container_width=True)


def analyze_b_party(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function analyzes the 'B-Party' column for known phone numbers and provides
    statistics on call frequency, call locations, and call durations.

    Args:
    - df (pd.DataFrame): DataFrame containing call data.

    Returns:
    - pd.DataFrame: DataFrame containing statistics for each known phone number.
    """
    # Filter out rows where 'B-Party' is empty or unknown, and exclude overly large B-Party numbers
    df_filtered = df[df["B-Party"].notna() & (df["B-Party"] != "Unknown")]
    df_filtered = df_filtered[df_filtered["B-Party"].apply(lambda x: len(str(x)) <= 13)]

    # Ensure 'Latitude' and 'Longitude' are numeric
    df_filtered["Latitude"] = pd.to_numeric(df_filtered["Latitude"], errors="coerce")
    df_filtered["Longitude"] = pd.to_numeric(df_filtered["Longitude"], errors="coerce")
    # Drop rows where 'Duration' is NaN as they can't be used in aggregation
    df_filtered = df_filtered.dropna(subset=["Duration"])

    # Group the data by 'B-Party' to calculate the desired metrics
    b_party_stats = (
        df_filtered.groupby("B-Party")
        .agg(
            Call_Frequency=("B-Party", "size"),
            Avg_Call_Duration=("Duration", "mean"),
            Total_Call_Duration=("Duration", "sum"),
            Locations=("Address", lambda x: ", ".join(x.dropna().astype(str).unique())),
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
        )
        .reset_index()
    )

    # Rename columns for clarity
    b_party_stats.columns = [
        "B-Party",
        "Call Frequency",
        "Avg Call Duration (s)",
        "Total Call Duration (s)",
        "Call Locations",
        "Avg Latitude",
        "Avg Longitude",
    ]

    return b_party_stats


def display_b_party_analysis(df: pd.DataFrame) -> None:
    """
    Streamlit function to display the B-Party analysis in the app.
    """

    # Check if the DataFrame has the necessary columns
    required_columns = ["B-Party", "Duration", "Address", "Latitude", "Longitude"]
    if not all(col in df.columns for col in required_columns):
        st.error(
            f"Missing required columns: {', '.join([col for col in required_columns if col not in df.columns])}"
        )
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
        b_party_stats_chart = (
            alt.Chart(b_party_stats)
            .mark_bar()
            .encode(
                x=alt.X("B-Party", sort="-y", title="Phone Number"),
                y=alt.Y("Call Frequency", title="Frequency of Calls"),
                tooltip=[
                    "B-Party",
                    "Call Frequency",
                    "Avg Call Duration (s)",
                    "Total Call Duration (s)",
                    "Call Locations",
                ],
            )
            .properties(
                width=700, height=400, title="Call Frequency for Known B-Parties"
            )
        )
        st.altair_chart(b_party_stats_chart, use_container_width=True)


def display_dataset_highlights(df: pd.DataFrame):
    """
    Display important highlights from the dataset including:
    - Last known location
    - A-Party number
    - Most called B-Party
    - Longest Call B-Party
    - CDR start and end date-time
    - IMEI(s) used, IMSI(s) used
    - Important International numbers in B-Party
    - Afg numbers in B-Party
    - Total GSM activity, Internet usage, etc.
    - Number of unique B-Party numbers
    - Longest incoming and outgoing calls with details
    """

    def display_metric_card(title, value):
        st.markdown(
            f"""
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;margin:5px;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <p style="margin:0;color:#4a4a4a;font-weight:bold">{title}</p>
            <p style="font-size:22px;color:#007acc;margin:4px 0;font-weight:600">{value}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        
    def display_date_cards(startDate, endDate):
        st.markdown(
            f"""
        <div style="display:flex;justify-content:space-between;margin:5px">
            <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin:0;font-size:16px;">Start Date</h3>
                <p style="font-size:20px;font-weight:bold;margin-top:5px">{startDate}</p>
            </div>
            <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin:0;font-size:16px;">End Date</h3>
                <p style="font-size:20px;font-weight:bold;margin-top:5px">{endDate}</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Basic processing (same as before)

    # 1. Last Known Location
    last_known_location = (
        df["Address"].dropna().iloc[-1] if "Address" in df.columns else "N/A"
    )

    # 2. A-Party number
    a_party_number = (
        df["A-Party"].dropna().unique().tolist() if "A-Party" in df.columns else "N/A"
    )

    # 3. Most called B-Party
    most_called_b_party = df["B-Party"].mode()[0] if "B-Party" in df.columns else "N/A"

    # 4. Longest Call B-Party
    longest_call_b_party = (
        df.loc[df["Duration"].idxmax()]["B-Party"]
        if "Duration" in df.columns and "B-Party" in df.columns
        else "N/A"
    )

    # 5. CDR Start and End Date-Time
    if "Date & Time" in df.columns:
        df["Date & Time"] = pd.to_datetime(df["Date & Time"])
        cdr_start = df["Date & Time"].min()
        cdr_end = df["Date & Time"].max()
    else:
        cdr_start, cdr_end = "N/A", "N/A"

    # 6. Unique B-Party Numbers
    unique_b_parties = df["B-Party"].nunique() if "B-Party" in df.columns else "N/A"

    # 7. Longest Incoming and Outgoing Calls
    if (
        "Call Type" in df.columns
        and "Duration" in df.columns
        and "B-Party" in df.columns
    ):
        incoming_calls = df[df["Call Type"] == "InComing"]
        outgoing_calls = df[df["Call Type"] == "OutGoing"]

        longest_incoming_call = incoming_calls.loc[incoming_calls["Duration"].idxmax()]
        longest_outgoing_call = outgoing_calls.loc[outgoing_calls["Duration"].idxmax()]

        longest_incoming_b_party = longest_incoming_call["B-Party"]
        longest_incoming_duration = longest_incoming_call["Duration"]

        longest_outgoing_b_party = longest_outgoing_call["B-Party"]
        longest_outgoing_duration = longest_outgoing_call["Duration"]
    else:
        longest_incoming_b_party, longest_incoming_duration = "N/A", "N/A"
        longest_outgoing_b_party, longest_outgoing_duration = "N/A", "N/A"

    # Display the highlights using a combination of tables and metrics
    st.title("CDR Dataset Highlights")

    # CDR Date Range
    st.subheader("CDR Date Range")
    display_date_cards(cdr_start, cdr_end)

    # Key metrics section
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_metric_card("Last Known Location", last_known_location)
    with col2:
        display_metric_card("A-Party Number", a_party_number[0])
    with col3:
        display_metric_card("Most Called B-Party", most_called_b_party)
    with col4:
        display_metric_card("Longest Call B-Party", longest_call_b_party)

        # Display additional metrics for B-Party
    st.markdown(
            f"""
        <div style="display:flex;justify-content:space-between;margin:5px">
            <div style="background-color:#d4e5ff;padding:20px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin:0;font-size:16px;">Unique B-Party Numbers</h3>
                <p style="font-size:18px;font-weight:bold;margin-top:5px">{unique_b_parties}</p>
            </div>
            <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin:0;font-size:16px;">Longest Incoming Call</h3>
                <p style="font-size:18px;font-weight:bold;margin-top:5px">{longest_incoming_b_party} ({longest_incoming_duration} seconds)</p>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;margin:5px">
            <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
                <h3 style="margin:0;font-size:16px;">Longest Outgoing Call</h3>
                <p style="font-size:18px;font-weight:bold;margin-top:5px">{longest_outgoing_b_party} ({longest_outgoing_duration} seconds)</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
    )

    # IMEI and IMSI Usage
    st.subheader("IMEI and IMSI Usage")

    imei_used = df["IMEI"].unique().tolist() if "IMEI" in df.columns else ["N/A"]
    imsi_used = df["IMSI"].unique().tolist() if "IMSI" in df.columns else ["N/A"]

    st.markdown(
        f"""
    <div style="display:flex;justify-content:space-between;margin:5px">
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">IMEI(s) Used</h3>
            <p style="font-size:18px;font-weight:bold;margin-top:5px">{', '.join(map(str, imei_used))}</p>
        </div>
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">IMSI(s) Used</h3>
            <p style="font-size:18px;font-weight:bold;margin-top:5px">{', '.join(map(str, imsi_used))}</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # International and Afg Numbers in B-Party
    st.subheader("International and Afghan Numbers in B-Party")
    international_numbers = (
        df[df["B-Party"].str.startswith("+")]["B-Party"].unique().tolist()
        if "B-Party" in df.columns
        else "N/A"
    )
    afg_numbers = (
        df[df["B-Party"].str.startswith("93")]["B-Party"].unique().tolist()
        if "B-Party" in df.columns
        else "N/A"
    )

    st.markdown(
        f"""
    <div style="display:flex;justify-content:space-between;margin:5px;">
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">International Numbers</h3>
            <p style="font-size:16px;font-weight:bold;margin-top:5px;overflow-wrap:anywhere;">{', '.join(map(str, international_numbers))}</p>
        </div>
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">Afghan Numbers</h3>
            <p style="font-size:16px;font-weight:bold;margin-top:5px;overflow-wrap:anywhere;">{', '.join(map(str, afg_numbers))}</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Additional HTML to show the total counts of IMEI and IMSI used

    # GSM Activity
    st.subheader("GSM Activity")
    total_gsm_activity = len(df)
    outgoing_calls_sms = (
        len(df[df["Call Type"] == "OutGoing"]) if "Call Type" in df.columns else "N/A"
    )
    incoming_calls_sms = (
        len(df[df["Call Type"] == "InComing"]) if "Call Type" in df.columns else "N/A"
    )

    st.markdown(
        f"""
    <div style="background-color:#cfe2ff;padding:20px;border-radius:10px;text-align:center;margin:5px;width:100%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
        <h3 style="margin:0;font-size:18px;">Total GSM Activity (Calls/SMS)</h3>
        <p style="font-size:22px;font-weight:bold;color:#004085;margin:5px 0;">{total_gsm_activity}</p>
    </div>
    <div style="display:flex;justify-content:space-between;margin:5px">
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">Outgoing Calls/SMS</h3>
            <p style="font-size:18px;font-weight:bold;margin-top:5px">{outgoing_calls_sms}</p>
        </div>
        <div style="background-color:#d4e5ff;padding:15px;border-radius:8px;text-align:center;color:#004b8d;width:48%;box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:16px;">Incoming Calls/SMS</h3>
            <p style="font-size:18px;font-weight:bold;margin-top:5px">{incoming_calls_sms}</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# Utility function to categorize time of day
def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning (06:00-12:00)"
    elif 12 <= hour < 18:
        return "Afternoon (12:00-18:00)"
    else:
        return "Evening/Night (18:00-06:00)"
