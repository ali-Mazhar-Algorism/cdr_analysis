from typing import Tuple, List
import pandas as pd
import streamlit as st
from geopy.distance import geodesic
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

            # Prepare map data
            map_data = pd.DataFrame(
                {
                    "Latitude": closest_coordinates["df1_Latitude"].tolist()
                    + closest_coordinates["df2_Latitude"].tolist(),
                    "Longitude": closest_coordinates["df1_Longitude"].tolist()
                    + closest_coordinates["df2_Longitude"].tolist(),
                    "Color": [[255, 0, 0]] * len(closest_coordinates)
                    + [[0, 0, 255]] * len(closest_coordinates),
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
                get_radius=75, 
                pickable=True,
            )

            # Center map dynamically based on the points
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=map_data["Latitude"].mean(),
                        longitude=map_data["Longitude"].mean(),
                        zoom=11,
                        pitch=50,
                    ),
                    layers=[layer],
                )
            )
        else:
            st.write("No closest coordinates found between the two files.")


def find_closest_coordinates(df1, df2):
    """
    Find the closest coordinates between two DataFrames.

    Parameters:
    - df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.

    Returns:
    - closest_coordinates: DataFrame with the closest coordinates and distances.
    """
    # Drop rows with missing coordinates
    df1_clean = df1.dropna(subset=["Latitude", "Longitude"])
    df2_clean = df2.dropna(subset=["Latitude", "Longitude"])

    closest_coordinates = []

    # Iterate over each coordinate pair in df1
    for _, row1 in df1_clean.iterrows():
        coord1 = (row1["Latitude"], row1["Longitude"])
        min_distance = float("inf")
        closest_coord = None

        # Compare with all coordinate pairs in df2
        for _, row2 in df2_clean.iterrows():
            coord2 = (row2["Latitude"], row2["Longitude"])
            try:
                distance = geodesic(coord1, coord2).meters  # Distance in meters
                if distance < min_distance:
                    min_distance = distance
                    closest_coord = row2
            except ValueError as e:
                print(f"Skipping invalid coordinate pair: {coord1} or {coord2} - {e}")

        if closest_coord is not None:
            closest_coordinates.append(
                {
                    "df1_Latitude": coord1[0],
                    "df1_Longitude": coord1[1],
                    "df1_Address": row1.get("Address", "N/A"),  # Include address if available
                    "df2_Latitude": closest_coord["Latitude"],
                    "df2_Longitude": closest_coord["Longitude"],
                    "Distance (meters)": min_distance,
                }
            )

    return pd.DataFrame(closest_coordinates)
