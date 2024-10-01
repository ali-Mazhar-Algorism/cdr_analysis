
sections = ["Introduction", "Geolocation Map", "Analysis"]

geo_markdown = '''
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

analysis_markdown_1 = '''
## 3.1 Call Type Frequency by Phase of Day (Morning, Afternoon, Evening)
The following sub section plots the Call information (whether is a call or sms) frequency by the phase of day. 
It helps analyze user activity according to the phase of the day.

The phase of the day is calculated by the following rules:

- Morning: 6 AM - 12 PM
- Afternoon: 12 PM - 6 PM
- Evening: 6 PM - 12 AM
'''