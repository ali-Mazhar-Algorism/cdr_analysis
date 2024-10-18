
sections = ["Introduction", "Geolocation Map", "Analysis"]
required_columns = {"Date & Time", "Latitude", "Longitude"}

# maps
analysis_markdown_1 = '''
## 3.1 Call Type Frequency by Phase of Day (Morning, Afternoon, Evening)
The following sub section plots the Call information (whether is a call or sms) frequency by the phase of day. 
It helps analyze user activity according to the phase of the day.

The phase of the day is calculated by the following rules:

- Morning: 6 AM - 12 PM
- Afternoon: 12 PM - 6 PM
- Evening: 6 PM - 12 AM
'''

scatter_map_guide = '''
This page will have a map of the geolocation of the ```Call/SMS``` data.

#### Features 

1. Zoom in and out of the map.
2. Select the call types to display on the map.
3. Hover over an individual data point and see the details of the CDR data.

    a. The details will include the `B-Party`, the `Call Type`, the `Date & Time` of the type of contact.
4. The color of the data point will be :large_orange_circle: for `OUTGOING SMS` and :large_green_circle: for `INCOMING SMS`, :large_blue_circle: for `INCOMING CALL` and :red_circle: for `OUTGOING CALL`.
5. Toggle the map to view on the whole page by clicking the `View fullscreen` button on the top right of the map.
6. The largest circle is the latest/current location.
'''


line_tracking_map_guide =  """
This chart visualizes the movement of the tracked entity over time. The color gradient represents the transition from the starting point to the endpoint:
- **Starting Point**: :large_green_circle:
- **End Point**: 🔴

The chart displays the path taken, with the color changing from :large_green_circle: to 🔴 as the entity moves from the start to the end of the selected date range.
"""

select_type_info = "INFO: Select the call types from the dropdown to display on the map."




# analysis
day_freq_heading = "Call Frequency by Dates"

phase_freq_heading = "Call Frequency by Phase of Day"
phase_freq_sub_heading = "This analysis will show the frequency of calls by phase of day (morning, afternoon, evening/night)."
phase_freq_sub_info = "INFO: Please select the call types you want to analyze."

calls_plot_heading = f"Longest Talk Times"
calls_plot_sub_heading = "Each bar represents a call with a duration in minutes. The x-axis shows the B-Party number and the y-axis shows the duration of the call. Hover over the bar to see the details."

b_analysis_heading = "B-Party Analysis"
b_analysis_sub_heading = "The following analysis provides a detailed breakdown for selected B-Party numbers and call types. It visualizes the number of calls and the total call duration over time."

# warnings

no_data_warning = "No data available for the selected types."
no_selected_type_warning = "Please select at least one call type to display the map."
no_location_data_warning = "Latitude and Longitude columns are missing in the dataframe."
no_date_selected = "The 'Date & Time' column is missing in the data."