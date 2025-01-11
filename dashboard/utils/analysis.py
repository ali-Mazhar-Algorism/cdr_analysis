from typing import Tuple, List
import pandas as pd
import streamlit as st
from utils.data_helper import load_data
from utils.data_processing import find_table_start, preprocess_file
import numpy as np
import pydeck as pdk
import altair as alt
from .constants import (
    day_freq_heading,
    phase_freq_heading,
    phase_freq_sub_heading,
    phase_freq_sub_info,
    calls_plot_heading,
    calls_plot_sub_heading,
    b_analysis_heading,
    b_analysis_sub_heading,
    no_data_warning,
    no_selected_type_warning,
    no_date_selected,
)


def categorize_time(hour):
    """Categorizes the given hour into a time of day."""
    if 6 <= hour < 12:
        return "Morning (06:00-12:00)"
    elif 12 <= hour < 18:
        return "Afternoon (12:00-18:00)"
    else:
        return "Evening/Night (18:00-06:00)"


@st.cache_data
def build_frequency_df(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Builds a frequency DataFrame from the provided data, extracting the hour and categorizing the time of day."""
    if "Date & Time" in data.columns:
        data["Hour"] = pd.to_datetime(data["Date & Time"]).dt.hour
        data["Time of Day"] = data["Hour"].apply(categorize_time)
    else:
        st.warning(no_date_selected)

    call_types = data["Call Type"].unique() if "Call Type" in data.columns else []

    return data, call_types


def display_call_duration_per_day(df: pd.DataFrame) -> None:
    """
    Streamlit function to display a bar chart of total call duration per day with interactive tooltips.
    """

    st.subheader(day_freq_heading)

    df["Date & Time"] = pd.to_datetime(df["Date & Time"])

    call_types = df["Call Type"].unique().tolist()
    selected_call_types = st.multiselect(
        "Select Call Types",
        call_types,
        default=call_types[0] if call_types else None,
        key="multi_day",
    )

    if not selected_call_types:
        st.warning(no_selected_type_warning)
        return

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
    filtered_date_range = pd.date_range(start=start_date, end=end_date).date

    filtered_data["Date"] = filtered_data["Date & Time"].dt.date
    daily_duration = filtered_data.groupby("Date")["Duration"].sum().reset_index()

    all_dates = pd.DataFrame(filtered_date_range, columns=["Date"])
    daily_duration = pd.merge(all_dates, daily_duration, on="Date", how="left").fillna(
        0
    )

    daily_duration = daily_duration.rename(
        columns={"Duration": "Total Call Duration (s)"}
    )

    if daily_duration.empty:
        st.warning("No call data available.")
    else:
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
    st.subheader(phase_freq_heading)
    st.write(phase_freq_sub_heading)

    df["Date & Time"] = pd.to_datetime(df["Date & Time"])

    def get_phase_of_day(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Evening/Night"

    df["Phase of Day"] = df["Date & Time"].dt.hour.apply(get_phase_of_day)

    call_types = df["Call Type"].unique().tolist()

    st.info(phase_freq_sub_info)

    selected_call_types = st.multiselect(
        "Select Call Types", call_types, default=call_types[0], key="multi_day_phase"
    )

    if not selected_call_types:
        st.warning(no_selected_type_warning)
        return

    filtered_data = df[df["Call Type"].isin(selected_call_types)]

    phase_call_count = (
        filtered_data.groupby(["Phase of Day", "Call Type"])
        .size()
        .reset_index(name="Count")
    )

    if phase_call_count.empty:
        st.warning(no_data_warning)
        return

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

    st.altair_chart(phase_of_day_chart, use_container_width=True)


def plot_call_frequency_by_time(data: pd.DataFrame):
    if "Date & Time" in data.columns:

        display_call_duration_per_day(data)
        display_call_frequency_by_phase(data)


def plot_longest_calls(data: pd.DataFrame):
    st.subheader(calls_plot_heading)
    st.write(calls_plot_sub_heading)

    filtered_data = data[
        data["Call Type"].str.contains("InComing|OutGoing", case=False, na=False)
    ]

    filtered_data = filtered_data[
        filtered_data["B-Party"].apply(lambda x: len(str(x)) >= 9)
    ]

    num_calls = st.slider(
        "Select the number of longest calls to display",
        min_value=1,
        max_value=20,
        value=10,
    )
    longest_calls = filtered_data.sort_values(by="Duration", ascending=False).head(
        num_calls
    )

    longest_calls["Call ID"] = longest_calls.index

    longest_calls["Call Pair"] = (
        longest_calls["A-Party"] + " -> " + longest_calls["B-Party"]
    )

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
    st.title(b_analysis_heading)
    st.write(b_analysis_sub_heading)

    display_b_party_analysis(data)

    data["Date & Time"] = pd.to_datetime(data["Date & Time"])

    data["Year"] = data["Date & Time"].dt.year
    data["Month"] = data["Date & Time"].dt.month
    data["Month-Year"] = data["Date & Time"].dt.to_period("M")
    data["Date"] = data["Date & Time"].dt.date

    b_party_numbers = data["B-Party"].unique()
    call_types = data["Call Type"].unique()

    selected_b_party = st.multiselect("Select B-Party Numbers", b_party_numbers)

    selected_call_type = st.multiselect(
        "Select Call Type", call_types, default=call_types[0], key="multiselect_b_party"
    )

    if not selected_b_party:
        st.warning("Please select at least one B-Party number.")
        return

    filtered_data = data[
        (data["B-Party"].isin(selected_b_party))
        & (data["Call Type"].isin(selected_call_type))
    ]

    if filtered_data.empty:
        st.warning(no_data_warning)
    else:
        st.write(f"Calls for selected B-Party numbers, Call Type: {selected_call_type}")
        st.write(filtered_data)

        grouped_data = (
            filtered_data.groupby(["Date"])
            .agg(
                Number_of_Calls=("Date & Time", "size"),
                Total_Call_Duration=("Duration", "sum"),
            )
            .reset_index()
        )

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

        line_chart = (
            alt.Chart(grouped_data)
            .mark_line(color="red")
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Total_Call_Duration:Q", title="Total Call Duration (seconds)"),
                tooltip=["Date:T", "Number_of_Calls:Q", "Total_Call_Duration:Q"],
            )
        )

        combined_chart = (
            alt.layer(bar_chart, line_chart)
            .resolve_scale(y="independent")
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
    df_filtered = df[df["B-Party"].notna() & (df["B-Party"] != "Unknown")]
    df_filtered = df_filtered[df_filtered["B-Party"].apply(lambda x: len(str(x)) <= 13)]

    df_filtered["Latitude"] = pd.to_numeric(df_filtered["Latitude"], errors="coerce")
    df_filtered["Longitude"] = pd.to_numeric(df_filtered["Longitude"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["Duration"])

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

    required_columns = ["B-Party", "Duration", "Address", "Latitude", "Longitude"]
    if not all(col in df.columns for col in required_columns):
        st.error(
            f"Missing required columns: {', '.join([col for col in required_columns if col not in df.columns])}"
        )
        return

    b_party_stats = analyze_b_party(df)

    if b_party_stats.empty:
        st.warning("No known phone numbers (B-Party) found in the dataset.")
    else:
        st.dataframe(b_party_stats)

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


def display_card(
    title,
    value,
    width="100%",
    padding="15px",
    font_size="18px",
    font_weight="bold",
    title_font_size="16px",
    card_color="#26292a",
):
    """Displays a customizable card."""
    st.markdown(
        f"""
        <div style="background-color:{card_color};padding:{padding};border-radius:8px;text-align:center;color:#c9c8c4;width:{width};box-shadow:0px 2px 5px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;font-size:{title_font_size};">{title}</h3>
            <p style="font-size:{font_size};font-weight:{font_weight};margin-top:5px;color:#70b8ef">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_dataset_highlights(df: pd.DataFrame):
    """
    Displays highlights from the CDR dataset, including key metrics such as last known location,
    A-Party number, most called B-Party, longest call B-Party, CDR date range, unique B-Party numbers,
    longest incoming and outgoing calls, IMEI and IMSI usage, international and Afghan numbers in B-Party,
    and GSM activity statistics.
    """

    st.dataframe(df)
    st.download_button(
        label="Download Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cdr_dataset.csv",
        mime="text/csv",
    )

    last_known_location = (
        df["Address"].dropna().iloc[-1]
        if "Address" in df.columns and not df["Address"].dropna().empty
        else "N/A"
    )

    a_party_number = (
        df["A-Party"].dropna().unique().tolist() if "A-Party" in df.columns else "N/A"
    )

    most_called_b_party = df["B-Party"].mode()[0] if "B-Party" in df.columns else "N/A"

    longest_call_b_party = (
        df.loc[df["Duration"].idxmax()]["B-Party"]
        if "Duration" in df.columns and "B-Party" in df.columns
        else "N/A"
    )

    if "Date & Time" in df.columns:
        df["Date & Time"] = pd.to_datetime(df["Date & Time"])
        cdr_start = df["Date & Time"].min()
        cdr_end = df["Date & Time"].max()
    else:
        cdr_start, cdr_end = "N/A", "N/A"

    unique_b_parties = df["B-Party"].nunique() if "B-Party" in df.columns else "N/A"

    if (
        "Call Type" in df.columns
        and "Duration" in df.columns
        and "B-Party" in df.columns
    ):
        incoming_calls = df[
            df["Call Type"].str.contains("InComing", na=False, case=False)
        ]
        outgoing_calls = df[
            df["Call Type"].str.contains("OutGoing", na=False, case=False)
        ]

        longest_incoming_call = (
            incoming_calls.loc[incoming_calls["Duration"].idxmax()]
            if not incoming_calls.empty
            else None
        )
        longest_outgoing_call = (
            outgoing_calls.loc[outgoing_calls["Duration"].idxmax()]
            if not outgoing_calls.empty
            else None
        )

        longest_incoming_b_party = (
            longest_incoming_call["B-Party"]
            if longest_incoming_call is not None
            else "N/A"
        )
        longest_incoming_duration = (
            longest_incoming_call["Duration"]
            if longest_incoming_call is not None
            else "N/A"
        )

        longest_outgoing_b_party = (
            longest_outgoing_call["B-Party"]
            if longest_outgoing_call is not None
            else "N/A"
        )
        longest_outgoing_duration = (
            longest_outgoing_call["Duration"]
            if longest_outgoing_call is not None
            else "N/A"
        )
    else:
        longest_incoming_b_party, longest_incoming_duration = "N/A", "N/A"
        longest_outgoing_b_party, longest_outgoing_duration = "N/A", "N/A"

    st.title("CDR Dataset Highlights")

    st.subheader("CDR Date Range")
    col1, col2 = st.columns(2)
    with col1:
        display_card("Start Date", cdr_start, padding="15px", font_size="20px")
    with col2:
        display_card("End Date", cdr_end, padding="15px", font_size="20px")

    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_card("Last Known Location", last_known_location)
    with col2:
        display_card("A-Party Number", a_party_number[0])
    with col3:
        display_card("Most Called B-Party", most_called_b_party)
    with col4:
        display_card("Longest Call B-Party", longest_call_b_party)

    st.header("Additional Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_card("Unique B-Party Numbers", unique_b_parties)
    with col2:
        display_card(
            "Longest Incoming Call",
            f"{longest_incoming_b_party} ({longest_incoming_duration} seconds)",
        )
    with col3:
        display_card(
            "Longest Outgoing Call",
            f"{longest_outgoing_b_party} ({longest_outgoing_duration} seconds)",
        )

    st.subheader("IMEI and IMSI Usage")
    imei_used = df["IMEI"].unique().tolist() if "IMEI" in df.columns else ["N/A"]
    imsi_used = df["IMSI"].unique().tolist() if "IMSI" in df.columns else ["N/A"]
    col1, col2 = st.columns(2)
    with col1:
        display_card("IMEI(s) Used", ", ".join(map(str, imei_used)))
    with col2:
        display_card("IMSI(s) Used", ", ".join(map(str, imsi_used)))

    st.subheader("International and Afghan Numbers in B-Party")
    international_numbers = (
        df[df["B-Party"].str.startswith("+")]["B-Party"].unique().tolist()
        if "B-Party" in df.columns
        else "N/A"
    )
    afg_numbers = (
        df[(df["B-Party"].str.startswith("93")) & (df["B-Party"].str.len() > 8)][
            "B-Party"
        ]
        .unique()
        .tolist()
        if "B-Party" in df.columns
        else "N/A"
    )
    col1, col2 = st.columns(2)
    with col1:
        display_card(
            "International Numbers",
            ", ".join(map(str, international_numbers)),
            font_size="16px",
        )
    with col2:
        display_card(
            "Afghan Numbers", ", ".join(map(str, afg_numbers)), font_size="16px"
        )

    st.subheader("GSM Activity")
    total_gsm_activity = len(df)
    outgoing_calls_sms = (
        len(df[df["Call Type"].str.contains("OutGoing", na=False, case=False)])
        if "Call Type" in df.columns
        else "N/A"
    )
    incoming_calls_sms = (
        len(df[df["Call Type"].str.contains("InComing", na=False, case=False)])
        if "Call Type" in df.columns
        else "N/A"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        display_card("Total GSM Activity (Calls/SMS)", total_gsm_activity)
    with col2:
        display_card("Outgoing Calls/SMS", outgoing_calls_sms)
    with col3:
        display_card("Incoming Calls/SMS", incoming_calls_sms)


def two_file_comparative_analysis(df1, df2):
    st.title("Two File Comparative Analysis")

    # Display highlights for both datasets side by side
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.header("Dataset 1 Highlights")
    #     display_dataset_highlights(df1)

    # # Add a divider between the two columns
    # st.markdown("---")  # This creates a horizontal line as a divider

    # with col2:
    #     st.header("Dataset 2 Highlights")
    #     display_dataset_highlights(df2)

    # 1. Check if any A-Party in df1 is in B-Party in df2 and vice versa
    common_a_b_parties = df1[df1["A-Party"].isin(df2["B-Party"])]
    common_b_a_parties = df2[df2["A-Party"].isin(df1["B-Party"])]

    st.subheader("Connections between A-Party in one file and B-Party in the other")
    if not common_a_b_parties.empty or not common_b_a_parties.empty:
        st.write("Common A-Party in `df1` found in B-Party of `df2`:")
        st.write(common_a_b_parties)
        st.write("Common A-Party in `df2` found in B-Party of `df1`:")
        st.write(common_b_a_parties)
    else:
        st.write(
            "No A-Party numbers in one file were found as B-Party numbers in the other."
        )

    # 3. Find common IMEI and IMSI numbers if columns are available
    if "IMEI" in df1.columns and "IMEI" in df2.columns:
        # Ensure both IMEI columns are of the same type
        df1["IMEI"] = df1["IMEI"].astype(str)
        df2["IMEI"] = df2["IMEI"].astype(str)

        common_imei = pd.merge(
            df1[["IMEI"]], df2[["IMEI"]], on="IMEI"
        ).drop_duplicates()

        st.subheader("Common IMEI Numbers")
        if not common_imei.empty:
            st.write(common_imei)
        else:
            st.write("No common IMEI numbers found.")

    if "IMSI" in df1.columns and "IMSI" in df2.columns:
        # Ensure both IMSI columns are of the same type
        df1["IMSI"] = df1["IMSI"].astype(str)
        df2["IMSI"] = df2["IMSI"].astype(str)

        # Find common IMSI values
        common_imsi = pd.merge(
            df1[["IMSI"]], df2[["IMSI"]], on="IMSI"
        ).drop_duplicates()

        st.subheader("Common IMSI Numbers")
        if not common_imsi.empty:
            st.write(common_imsi)
        else:
            st.write("No common IMSI numbers found.")

    # 4. Find common B-Party numbers between df1 and df2
    common_b_parties = pd.merge(
        df1[["B-Party"]], df2[["B-Party"]], on="B-Party"
    ).drop_duplicates()

    st.subheader("Common B-Party Numbers")
    if not common_b_parties.empty:
        st.write("Common B-Party Numbers:")
        st.write(common_b_parties)
    else:
        st.write("No common B-Party numbers found between the two files.")

    st.subheader("Closest Coordinates Between Two Files")

    if (
        "Latitude" in df1.columns
        and "Longitude" in df1.columns
        and "Latitude" in df2.columns
        and "Longitude" in df2.columns
    ):
        closest_coordinates = find_closest_coordinates(df1, df2)

        if not closest_coordinates.empty:
            st.write(closest_coordinates)

            # Import PyDeck for visualization
            closest_coordinates["df1_DateTime"] = pd.to_datetime(closest_coordinates["df1_DateTime"], errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S")
            closest_coordinates["df12_DateTime"] = pd.to_datetime(closest_coordinates["df2_DateTime"], errors='coerce').dt.strftime("%Y-%m-%d %H:%M:%S")

            # Prepare map data
            map_data = pd.DataFrame(
                {
                    "Latitude": closest_coordinates["df1_Latitude"].tolist()
                    + closest_coordinates["df2_Latitude"].tolist(),
                    "Longitude": closest_coordinates["df1_Longitude"].tolist()
                    + closest_coordinates["df2_Longitude"].tolist(),
                    "Color": [[255, 0, 0]] * len(closest_coordinates)
                    + [[0, 0, 255]] * len(closest_coordinates),
                    "Address": closest_coordinates["df1_Address"].tolist()
                    + closest_coordinates["df2_Address"].tolist(),
                    "DateTime": closest_coordinates["df1_DateTime"].tolist()
                    + closest_coordinates["df2_DateTime"].tolist(),
                }
            )
            st.markdown(
                """
            ### Graph Description:
            - **Red Dots**: Points from the first dataset (`df1`).
            - **Blue Dots**: Closest points from the second dataset (`df2`).
            - Addresses from `df1` are displayed in the data table if available.
            """
            )
            # Adjust dot size and map centering
            layer = pdk.Layer(
                "ScatterplotLayer",
                map_data,
                get_position=["Longitude", "Latitude"],
                get_color="Color",
                get_radius=100, 
                pickable=True,
            )

            tooltip = {
                "html": """
                    <b>Latitude:</b> {Latitude}<br>
                    <b>Longitude:</b> {Longitude}<br>
                    <b>Address:</b> {Address}<br>
                    <b>Date & Time:</b> {DateTime}
                """,
                "style": {"backgroundColor": "steelblue", "color": "white"},
            }

            # Center map dynamically based on the points and add the tooltip
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=map_data["Latitude"].mean(),
                        longitude=map_data["Longitude"].mean(),
                        zoom=11,
                        pitch=50,
                    ),
                    layers=[layer],
                    tooltip=tooltip,  # Add tooltip to the map
                )
            )
        else:
            st.write("No closest coordinates found between the two files.")


def find_closest_coordinates(df1, df2):
    """
    Find the closest coordinates between two DataFrames using an approximate distance calculation.

    Parameters:
    - df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.

    Returns:
    - closest_coordinates: DataFrame with the closest coordinates, distances, addresses, and date-time information.
    """
    # Drop rows with missing coordinates
    df1_clean = df1.dropna(subset=["Latitude", "Longitude"])
    df2_clean = df2.dropna(subset=["Latitude", "Longitude"])

    closest_coordinates = []

    # Convert degrees to radians for faster calculations
    df1_clean["Latitude_rad"] = np.radians(df1_clean["Latitude"])
    df1_clean["Longitude_rad"] = np.radians(df1_clean["Longitude"])
    df2_clean["Latitude_rad"] = np.radians(df2_clean["Latitude"])
    df2_clean["Longitude_rad"] = np.radians(df2_clean["Longitude"])

    # Iterate over each coordinate pair in df1
    for _, row1 in df1_clean.iterrows():
        coord1_lat = row1["Latitude_rad"]
        coord1_lon = row1["Longitude_rad"]
        min_distance = float("inf")
        closest_coord = None

        # Compare with all coordinate pairs in df2
        for _, row2 in df2_clean.iterrows():
            coord2_lat = row2["Latitude_rad"]
            coord2_lon = row2["Longitude_rad"]
            
            # Approximate distance (Euclidean distance in radians)
            delta_lat = coord1_lat - coord2_lat
            delta_lon = coord1_lon - coord2_lon
            distance = np.sqrt(delta_lat**2 + delta_lon**2)  # Approximation in radians

            # Convert radians to meters (1 rad ≈ 6,371,000 meters on Earth's surface)
            distance_meters = distance * 6371000

            if distance_meters < min_distance:
                min_distance = distance_meters
                closest_coord = row2

        if closest_coord is not None:
            closest_coordinates.append(
                {
                    "df1_Latitude": row1["Latitude"],
                    "df1_Longitude": row1["Longitude"],
                    "df1_Address": row1.get("Address", "N/A"),  # Include address if available
                    "df2_Address": row2.get("Address", "N/A"),  # Include address if available
                    "df1_DateTime": row1.get("Date & Time", "N/A"),  # Include datetime if available
                    "df2_Latitude": closest_coord["Latitude"],
                    "df2_Longitude": closest_coord["Longitude"],
                    "df2_DateTime": closest_coord.get("Date & Time", "N/A"),  # Include datetime if available
                    "Distance (meters)": min_distance,
                }
            )

    return pd.DataFrame(closest_coordinates)


def create_color_palette(n):
    """Create distinct colors for each dataset"""
    base_colors = [
        [255, 0, 0],    # Red
        [0, 0, 255],    # Blue
        [0, 255, 0],    # Green
        [255, 165, 0],  # Orange
        [128, 0, 128]   # Purple
    ]
    return base_colors[:n]

def multi_file_comparative_analysis(dataframes):
    """
    Perform comparative analysis across multiple dataframes
    """
    st.title("Multi-File Comparative Analysis")
    
    # Analyze common parties across all files
    for i, df1 in enumerate(dataframes):
        for j, df2 in enumerate(dataframes[i+1:], start=i+1):
            st.subheader(f"Analysis between File {i+1} and File {j+1}")
            
            # Check A-Party in one file against B-Party in another
            common_a_b_parties = df1[df1["A-Party"].isin(df2["B-Party"])]
            common_b_a_parties = df2[df2["A-Party"].isin(df1["B-Party"])]
            
            st.write("#### Cross-Party Connections")
            if not common_a_b_parties.empty or not common_b_a_parties.empty:
                st.write(f"Common A-Party in File {i+1} found in B-Party of File {j+1}:")
                st.write(common_a_b_parties)
                st.write(f"Common A-Party in File {j+1} found in B-Party of File {i+1}:")
                st.write(common_b_a_parties)
            else:
                st.write("No cross-party connections found between these files.")

            # Check for common IMEI/IMSI if available
            if "IMEI" in df1.columns and "IMEI" in df2.columns:
                df1["IMEI"] = df1["IMEI"].astype(str)
                df2["IMEI"] = df2["IMEI"].astype(str)
                common_imei = pd.merge(df1[["IMEI"]], df2[["IMEI"]], on="IMEI").drop_duplicates()
                
                st.write("#### Common IMEI Numbers")
                if not common_imei.empty:
                    st.write(common_imei)
                else:
                    st.write("No common IMEI numbers found.")

            if "IMSI" in df1.columns and "IMSI" in df2.columns:
                df1["IMSI"] = df1["IMSI"].astype(str)
                df2["IMSI"] = df2["IMSI"].astype(str)
                common_imsi = pd.merge(df1[["IMSI"]], df2[["IMSI"]], on="IMSI").drop_duplicates()
                
                st.write("#### Common IMSI Numbers")
                if not common_imsi.empty:
                    st.write(common_imsi)
                else:
                    st.write("No common IMSI numbers found.")

            # Check for common B-Party numbers
            common_b_parties = pd.merge(df1[["B-Party"]], df2[["B-Party"]], on="B-Party").drop_duplicates()
            
            st.write("#### Common B-Party Numbers")
            if not common_b_parties.empty:
                st.write(common_b_parties)
            else:
                st.write("No common B-Party numbers found.")
            
            st.markdown("---")  # Add separator between file comparisons

    # Create combined map visualization for all files
    st.subheader("Combined Location Visualization")
    
    # Check if all dataframes have coordinate data
    dfs_with_coords = [df for df in dataframes if "Latitude" in df.columns and "Longitude" in df.columns]

    if dfs_with_coords:
        # Get color palette for the number of dataframes
        colors = create_color_palette(len(dfs_with_coords))
        
        # Create legend using Streamlit columns
        st.write("##### Legend:")
        cols = st.columns(len(dfs_with_coords))
        for idx, col in enumerate(cols):
            color = colors[idx]
            col.color_picker(
                f"File {idx + 1}",
                f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                disabled=True,
            )
        
        # Prepare combined map data
        all_points = []
        for idx, df in enumerate(dfs_with_coords):
            # Drop rows with missing coordinates
            df_clean = df.dropna(subset=["Latitude", "Longitude"])
            df_clean["date_time_str"] = pd.to_datetime(df_clean["Date & Time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_clean["date_time_obj"] = pd.to_datetime(df_clean["Date & Time"], errors="coerce")
            if not df_clean.empty:
                points = pd.DataFrame({
                    "Latitude": df_clean["Latitude"],
                    "Longitude": df_clean["Longitude"],
                    "Color": [colors[idx]] * len(df_clean),
                    "Source": [f"File {idx + 1}"] * len(df_clean),
                    "Call_Type": df_clean.get("Call Type", "N/A"),
                    "Date_Time": df_clean.get("date_time_str", "N/A"),
                    "DateTimeObj": df_clean.get("date_time_obj", pd.NaT),
                    "A_Party": df_clean.get("A-Party", "N/A"),
                    "B_Party": df_clean.get("B-Party", "N/A")
                })
                all_points.append(points)
        
        if all_points:
            combined_points = pd.concat(all_points, ignore_index=True)
            
            # Set up time slider
            st.info("INFO: Adjust the slider to select the time range for displaying locations.")
            min_date = combined_points["DateTimeObj"].min()
            max_date = combined_points["DateTimeObj"].max()
            
            if pd.isnull(min_date) or pd.isnull(max_date):
                st.warning("No valid Date & Time data available for filtering.")
            else:
                selected_time_range = st.slider(
                    "Select Date & Time Range:",
                    min_value=min_date.to_pydatetime(),
                    max_value=max_date.to_pydatetime(),
                    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                    format="YYYY-MM-DD HH:mm:ss",
                )
                
                # Filter data based on the selected time range
                time_filtered_data = combined_points[
                    (combined_points["DateTimeObj"] >= pd.to_datetime(selected_time_range[0]))
                    & (combined_points["DateTimeObj"] <= pd.to_datetime(selected_time_range[1]))
                ]
                
                if time_filtered_data.empty:
                    st.write("No data available for the selected time range.")
                else:
                    # Create PyDeck layer
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        time_filtered_data,
                        get_position=["Longitude", "Latitude"],
                        get_color="Color",
                        get_radius=75,
                        pickable=True,
                    )
                    
                    # Calculate view state based on filtered points
                    center_lat = time_filtered_data["Latitude"].mean()
                    center_lon = time_filtered_data["Longitude"].mean()
                    
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=11,
                        pitch=50,
                    )
                    
                    # Create and display the map with tooltip
                    r = pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={
                            "html": "<b>Source:</b> {Source}<br/>"
                                "<b>Call Type:</b> {Call_Type}<br/>"
                                "<b>Date & Time:</b> {Date_Time}<br/>"
                                "<b>A-Party:</b> {A_Party}<br/>"
                                "<b>B-Party:</b> {B_Party}<br/>"
                                "<b>Latitude:</b> {Latitude}<br/>"
                                "<b>Longitude:</b> {Longitude}"
                        }
                    )
                    
                    st.pydeck_chart(r)
        else:
            st.write("No valid coordinate data found in any of the files.")

    else:
        st.write("No coordinate data available in the uploaded files.")

def handle_multi_file_analysis():
    st.title("Multi-File Analysis")
    
    # Reset state button
    if "file_uploaded" in st.session_state:
        if st.button("Reset Analysis"):
            st.session_state.clear()
            st.rerun()
    
    # File count selector
    file_count = st.slider("Select number of files to analyze", min_value=2, max_value=10, value=2)
    
    # Create file uploaders
    uploaded_files = []
    for i in range(file_count):
        uploaded_file = st.file_uploader(f"Choose file {i+1} :file_folder:", key=f"file{i}")
        if uploaded_file:
            uploaded_files.append(uploaded_file)
    
    # Process files if all are uploaded
    if len(uploaded_files) == file_count:
        dataframes = []
        for file in uploaded_files:
            df = load_data(file)
            df = find_table_start(df)
            df = preprocess_file(df)
            dataframes.append(df)
        
        st.success(f"All {file_count} files uploaded and processed successfully!")
        
        # Perform comparative analysis
        multi_file_comparative_analysis(dataframes)
        
        st.session_state["file_uploaded"] = True
    else:
        st.info(f"Please upload all {file_count} files to begin analysis.")
        
        
# def find_closest_points_multi(dataframes):
#     """
#     Find closest points between multiple dataframes.
#     Returns a DataFrame with closest points and their details for each pair of files.
#     """
#     results = []
    
#     # Convert coordinates to radians for all dataframes
#     dfs_rad = []
#     for idx, df in enumerate(dataframes):
#         df_clean = df.dropna(subset=["Latitude", "Longitude"])
#         df_clean["Latitude_rad"] = np.radians(df_clean["Latitude"])
#         df_clean["Longitude_rad"] = np.radians(df_clean["Longitude"])
#         dfs_rad.append(df_clean)
    
#     # Compare each pair of dataframes
#     for i, df1 in enumerate(dfs_rad):
#         for j, df2 in enumerate(dfs_rad[i+1:], start=i+1):
#             closest_points = []
            
#             # Find closest points
#             for _, row1 in df1.iterrows():
#                 coord1_lat = row1["Latitude_rad"]
#                 coord1_lon = row1["Longitude_rad"]
#                 min_distance = float("inf")
#                 closest_coord = None
                
#                 for _, row2 in df2.iterrows():
#                     coord2_lat = row2["Latitude_rad"]
#                     coord2_lon = row2["Longitude_rad"]
                    
#                     # Approximate distance calculation
#                     delta_lat = coord1_lat - coord2_lat
#                     delta_lon = coord1_lon - coord2_lon
#                     distance = np.sqrt(delta_lat**2 + delta_lon**2)
#                     distance_meters = distance * 6371000  # Convert to meters
                    
#                     if distance_meters < min_distance:
#                         min_distance = distance_meters
#                         closest_coord = row2
                
#                 if closest_coord is not None and min_distance <= 1000:  # Only include points within 1km
#                     closest_points.append({
#                         "File_Pair": f"File {i+1} - File {j+1}",
#                         "Point1_Latitude": row1["Latitude"],
#                         "Point1_Longitude": row1["Longitude"],
#                         "Point1_DateTime": row1.get("Date/Time", "N/A"),
#                         "Point1_Address": row1.get("Address", "N/A"),
#                         "Point2_Latitude": closest_coord["Latitude"],
#                         "Point2_Longitude": closest_coord["Longitude"],
#                         "Point2_DateTime": closest_coord.get("Date/Time", "N/A"),
#                         "Point2_Address": closest_coord.get("Address", "N/A"),
#                         "Distance_Meters": min_distance
#                     })
            
#             if closest_points:
#                 results.extend(closest_points)
    
#     return pd.DataFrame(results) if results else pd.DataFrame()