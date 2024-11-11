import pandas as pd
from datetime import datetime


def find_table_start(df: pd.DataFrame) -> pd.DataFrame:
    # Look for the row where the column names start (e.g., 'CALL_TYPE', 'MSISDN')
    for i, row in df.iterrows():
        if (
            "CALL_TYPE" in row.values
            or "call_type" in row.values
            or "call" in row.values
        ):
            df.columns = df.iloc[i]  # Set this row as header
            df = df.drop(range(i + 1))  # Drop the rows above the header
            break
    return df


def prune_columns(raw_data: pd.DataFrame, threshold: float = 0.25):
    columns_to_remove = raw_data.filter(regex=r"^Unnamed").columns
    data = raw_data.drop(labels=columns_to_remove, axis=1)
    if "Sr #" in data.columns:
        data = data.drop(labels="Sr #", axis=1)

    threshold = len(data) * threshold
    # Drop columns that do not have at least the threshold number of non-NaN values
    df_cleaned = data.dropna(axis=1, thresh=threshold)
    # Display the cleaned DataFrame
    return df_cleaned


def format_phone_number(number: str):
    # Remove any leading '+' if present
    number = number.lstrip("+")

    # Handle international numbers that start with '00'
    if number.startswith("00"):
        return number
    # Check if it already starts with the Pakistani country code '92' and has the correct length
    elif number.startswith("92") and len(number) == 12:
        return number
    # Handle local numbers starting with '03' and convert them to the correct format
    elif len(number) == 10 and number.startswith("03"):
        return "92" + number[1:]
    # Handle local numbers starting with '3' and convert them to the correct format
    elif len(number) == 11 and number.startswith("3"):
        return "92" + number
    # For other numbers, we will check if they seem like valid Pakistani numbers
    elif number.startswith("0") and len(number) == 11:
        return "92" + number[1:]
    elif number.startswith("92") and len(number) > 12:
        return "92" + number[-10:]
    else:
        # Keep the number as it is if it does not fit any of the above criteria
        return number


def parse_site_info(info_string):
    # Handle case where the string is '?'
    if info_string == "?":
        return {
            "Address": "",
            "Latitude": "",
            "Longitude": "",
            "Connection code": "",
            "Internet": "",
        }

    # Split the string by the pipe '|' delimiter
    parts = info_string.split("|")

    # Extract latitude and longitude if available
    latitude = parts[1] if len(parts) > 1 else ""
    longitude = parts[2] if len(parts) > 2 else ""

    # Further split the first part to get connection code and address with internet info
    first_part = parts[0].split(", ")

    # Extract the connection code
    connection_code = first_part[0].strip() if len(first_part) > 0 else ""

    # Extract the internet type and address
    address_internet = (
        first_part[1].split(" (") if len(first_part) > 1 else [first_part[0], ""]
    )
    address = address_internet[0].strip() if len(address_internet) > 0 else ""
    internet = (
        address_internet[1].replace(")", "").strip()
        if len(address_internet) > 1
        else ""
    )

    # Create a dictionary to hold the parsed information
    parsed_info = {
        "Address": address,
        "Latitude": latitude,
        "Longitude": longitude,
        "Connection code": connection_code,
        "Internet": internet,
    }

    return parsed_info

def calculate_duration(df):
    # Check if both "Start Time" and "End Time" columns exist
    if ("Start Time" in df.columns and "End Time" in df.columns) or ("CALL_START_DT_TM" in df.columns and "CALL_END_DT_TM" in df.columns):
        # Convert to datetime format if not already
        if "Start Time" in df.columns:
            df["Start Time"] = pd.to_datetime(df["Start Time"], errors='coerce')
        if "CALL_START_DT_TM" in df.columns:
            df["Start Time"] = pd.to_datetime(df["CALL_START_DT_TM"], errors='coerce')
        
        if "End Time" in df.columns:
            df["End Time"] = pd.to_datetime(df["End Time"], errors='coerce')
        if "CALL_END_DT_TM" in df.columns:
            df["End Time"] = pd.to_datetime(df["CALL_END_DT_TM"], errors='coerce')
        
        # Calculate duration in seconds
        df["Duration"] = (df["End Time"] - df["Start Time"]).dt.total_seconds()
    else:
        print("Columns 'Start Time', 'End Time', 'CALL_START_DT_TM', and 'CALL_END_DT_TM' are not present in the DataFrame.")
    
    return df

# Common column mapping for all data types
COLUMN_MAPPING = {
    "CALL_TYPE": "Call Type",
    "CallType": "Call Type",
    "Type": "Call Type",
    "call_type": "Call Type",
    "Call Type": "Call Type",
    "MSISDN": "A-Party",
    "msisdn": "A-Party",
    "A Number": "A-Party",
    "Aparty": "A-Party",
    "A-Party": "A-Party",
    "CALL_DIALED_NUM": "B-Party",
    "BNUMBER": "B-Party",
    "bnumber": "B-Party",
    "B Number": "B-Party",
    "BParty": "B-Party",
    "STRT_TM": "Date & Time",
    "strt_tm": "Date & Time",
    "Datetime": "Date & Time",
    "Start Time": "Date & Time",
    "Date & Time": "Date & Time",
    "MINS": "Minutes",
    "mins": "Minutes",
    "SECS": "Seconds",
    "secs": "Seconds",
    "location": "Address",
    "SITE_ADDRESS": "Address",
    "site_address": "Address",
    "Site": "Address",
    "site": "Address",
    "SiteLocation": "Address",
    "Location": "Address",
    "Latitude": "Latitude",
    "lat": "Latitude",
    "LAT": "Latitude",
    "longitude": "Longitude",
    "LNG": "Longitude",
    "lng": "Longitude",
    "Longitude and Latitude": "Longitude",
    "imei": "IMEI",
    "Imei": "IMEI",
    "Imsi": "IMSI",
    "imsi": "IMSI",
}


# Standardize column names
def standardize_columns(df):
    df = df.rename(columns=COLUMN_MAPPING)
    return df


def seperate_address(df):
    # Check if the '|' character exists in any row of the 'Address' column
    if df["Address"].str.contains("|", na=False).any():
        address_split = df["Address"].str.split("|", expand=True)

        # If there are 4 parts, assign them to the respective columns
        if address_split.shape[1] == 4:
            df[["Address", "Latitude", "Longitude", "_"]] = address_split
        # If there are 3 parts, assign them to the respective columns
        elif address_split.shape[1] == 3:
            df[["Address", "Latitude", "Longitude"]] = address_split
        else:
            df["Address"] = df["Address"]

        # Convert Latitude and Longitude columns to numeric, coercing errors to NaN
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Return the dataframe (no changes if "|" is not in the Address column)
    return df


# Standardize Call Type values
def standardize_call_type(df):
    df["Call Type"] = df["Call Type"].replace(
        {
            "Call - Incoming": "InComing",
            "Call - Outgoing": "OutGoing",
            "SMS - Incoming": "InComing SMS",
            "SMS - Outgoing": "OutGoing SMS",
            "Incoming": "InComing",
            "Outgoing": "OutGoing",
            "Incoming SMS": "InComing SMS",
            "Outgoing SMS": "OutGoing SMS",
            "INCOMING": "InComing",
            "OUTGOING": "OutGoing",
        }
    )
    return df


# Preprocess each type based on specific requirements
def preprocess_common(df):
    # Extract A-Party number if needed
    # df = standardize_datetime(df, column="Date & Time")
    a_party_number = (
        df.iloc[0, 1]
        if isinstance(df.iloc[0, 1], (int, float)) and df.iloc[0, 1] >= 10
        else None
    )
    a_party_number = df.iloc[0]["A-Party"] if "A-Party" in df.columns else None

    df["A-Party"] = a_party_number

    # Calculate duration if Minutes and Seconds are available
    if "Minutes" in df.columns and "Seconds" in df.columns:
        df["Duration"] = pd.to_numeric(df["Minutes"]) * 60 + pd.to_numeric(
            df["Seconds"]
        )
    elif "Date & Time" in df.columns and "End Time" in df.columns:
        df["Duration"] = (
            pd.to_datetime(df["End Time"]) - pd.to_datetime(df["Date & Time"])
        ).dt.total_seconds()  

    # Standardize latitude and longitude to numeric
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    
    return df


# Function to standardize Date & Time format
def standardize_datetime(df, column="Date & Time"):
    # Define possible date formats
    date_formats = ["%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]

    def parse_date(date_str):
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).strftime("%m/%d/%Y %H:%M:%S")
            except ValueError:
                continue
        # Return NaT if no format matches
        return pd.NaT

    # Apply parse_date to Date & Time column
    df[column] = df[column].astype(str).apply(parse_date)
    return df


# Main preprocessing function
def preprocess_file(df):
    df = calculate_duration(df)
    df = standardize_columns(df)
    df = seperate_address(df)
    df = standardize_call_type(df)
    df = preprocess_common(df)
    df = prune_columns(df)
    
    required_columns = [
        "A-Party",
        "B-Party",
        "Call Type",
        "Date & Time",
        "Duration",
        "Address",
        "IMEI",
        "IMSI",
        "Latitude",
        "Longitude",
    ]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None  # or use np.nan for empty numeric fields

    df["A-Party"] = df["A-Party"].astype(str)
    df["B-Party"] = df["B-Party"].astype(str)

    df["A-Party"] = df["A-Party"].apply(format_phone_number)
    df["B-Party"] = df["B-Party"].apply(format_phone_number)

    return df[
        [
            "A-Party",
            "B-Party",
            "Call Type",
            "Date & Time",
            "Duration",
            "Address",
            "IMEI",
            "IMSI",
            "Latitude",
            "Longitude",
        ]
    ]
