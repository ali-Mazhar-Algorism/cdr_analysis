import pandas as pd

# Mapping column names for renaming
COLUMN_MAP = {
    'CALL_TYPE': 'Call Type',
    'call_type': 'Call Type',
    'call': 'Call Type',
    'MSISDN': 'A-Party',
    'BNUMBER': 'B-Party',
    'STRT_TM': 'Date & Time',
    'MINS': 'Minutes',
    'SECS': 'Seconds',
    'SITE_ADDRESS': 'Address',
    'LAT': 'Latitude',
    'LNG': 'Longitude',
    # Add other mappings as needed
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns using the COLUMN_MAP
    df = df.rename(columns=COLUMN_MAP)
    return df

def find_table_start(df: pd.DataFrame) -> pd.DataFrame:
    # Look for the row where the column names start (e.g., 'CALL_TYPE', 'MSISDN')
    for i, row in df.iterrows():
        if 'CALL_TYPE' in row.values or 'call_type' in row.values or 'call' in row.values:
            df.columns = df.iloc[i]  # Set this row as header
            df = df.drop(range(i + 1))  # Drop the rows above the header
            break
    return df

def prune_columns(raw_data: pd.DataFrame, threshold: float = 0.25):
    columns_to_remove = raw_data.filter(regex=r'^Unnamed').columns
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
    number = number.lstrip('+')
    
    # Handle international numbers that start with '00'
    if number.startswith('00'):
        return number
    # Check if it already starts with the Pakistani country code '92' and has the correct length
    elif number.startswith('92') and len(number) == 12:
        return number
    # Handle local numbers starting with '03' and convert them to the correct format
    elif len(number) == 10 and number.startswith('03'):
        return '92' + number[1:]
    # Handle local numbers starting with '3' and convert them to the correct format
    elif len(number) == 11 and number.startswith('3'):
        return '92' + number
    # For other numbers, we will check if they seem like valid Pakistani numbers
    elif number.startswith('0') and len(number) == 11:
        return '92' + number[1:]
    elif number.startswith('92') and len(number) > 12:
        return '92' + number[-10:]
    else:
        # Keep the number as it is if it does not fit any of the above criteria
        return number

def parse_site_info(info_string):
    # Handle case where the string is '?'
    if info_string == '?':
        return {
            'Address': '',
            'Latitude': '',
            'Longitude': '',
            'Connection code': '',
            'Internet': ''
        }
    
    # Split the string by the pipe '|' delimiter
    parts = info_string.split('|')
    
    # Extract latitude and longitude if available
    latitude = parts[1] if len(parts) > 1 else ''
    longitude = parts[2] if len(parts) > 2 else ''
    
    # Further split the first part to get connection code and address with internet info
    first_part = parts[0].split(', ')
    
    # Extract the connection code
    connection_code = first_part[0].strip() if len(first_part) > 0 else ''
    
    # Extract the internet type and address
    address_internet = first_part[1].split(' (') if len(first_part) > 1 else [first_part[0], '']
    address = address_internet[0].strip() if len(address_internet) > 0 else ''
    internet = address_internet[1].replace(')', '').strip() if len(address_internet) > 1 else ''
    
    # Create a dictionary to hold the parsed information
    parsed_info = {
        'Address': address,
        'Latitude': latitude,
        'Longitude': longitude,
        'Connection code': connection_code,
        'Internet': internet
    }
    
    return parsed_info

# Type-specific preprocessing functions

def preprocess_type1(df):
    # Extract the A-Party number from the first line
    a_party_number = df.iloc[0]['MSISDN'] if 'MSISDN' in df.columns else None
    # Rename columns
    df = df.rename(columns={
        'CALL_TYPE': 'Call Type',
        'MSISDN': 'A-Party',
        'BNUMBER': 'B-Party',
        'STRT_TM': 'Date & Time',
        'MINS': 'Minutes',
        'SECS': 'Seconds',
        'SITE_ADDRESS': 'Address',
        'LAT': 'Latitude',
        'LNG': 'Longitude'
    })
    
    # If A-Party is missing, fill it with the value extracted from the first row
    if 'A-Party' in df.columns and df['A-Party'].isna().all():
        df['A-Party'] = a_party_number
    
    # Calculate duration in seconds
    df['Duration'] = pd.to_numeric(df['Minutes']) * 60 + pd.to_numeric(df['Seconds'])
    
    # Standardize the 'Call Type' values
    df['Call Type'] = df['Call Type'].replace({
        'Call - Incoming': 'InComing',
        'Call - Outgoing': 'OutGoing',
        'SMS - Incoming': 'InComing SMS',
        'SMS - Outgoing': 'OutGoing SMS'
    })
    
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    # Return the standardized dataframe with relevant columns
    
    return df[['A-Party', 'B-Party', 'Call Type', 'Date & Time', 'Duration', 'Address', 'Latitude', 'Longitude']]

def preprocess_type2(df):
    df = df.filter(regex='^(?!.*Unnamed)')
    df = df.rename(columns={
        'Call Type': 'Call Type',
        'A-Party': 'A-Party',
        'B-Party': 'B-Party',
        'Date & Time': 'Date & Time',
        'Duration': 'Duration',
        'Site': 'Address'
    })

    df['Call Type'] = df['Call Type'].replace({
        'Incoming': 'InComing',
        'Outgoing': 'OutGoing',
        'Incoming SMS': 'InComing SMS',
        'Outgoing SMS': 'OutGoing SMS'
    })
    
    df[['Address', 'Latitude', 'Longitude', 'Some Key']] = df['Address'].str.split('|', expand=True)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    print(df.head(3))
    
    return df[['A-Party', 'B-Party', 'Call Type', 'Date & Time', 'Duration', 'Address', 'Latitude', 'Longitude']]

def preprocess_type3(df):
    df = df.rename(columns={
        'CallType': 'Call Type',
        'Aparty': 'A-Party',
        'BParty': 'B-Party',
        'Datetime': 'Date & Time',
        'Duration': 'Duration',
        'SiteLocation': 'Address'
    })
    
    df['Call Type'] = df['Call Type'].replace({
        'Incoming': 'InComing',
        'Outgoing': 'OutGoing',
        'Incoming SMS': 'InComing SMS',
        'Outgoing SMS': 'OutGoing SMS'
    })
    
    df[['Address', 'Latitude', 'Longitude', 'Some Code']] = df['Address'].str.split('|', expand=True)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    return df[['A-Party', 'B-Party', 'Call Type', 'Date & Time', 'Duration', 'Address', 'Latitude', 'Longitude']]

def preprocess_type4(df):
    df = df.rename(columns={
        'A Number': 'A-Party',
        'B Number': 'B-Party',
        'Direction': 'Call Type',
        'Start Time': 'Date & Time',
        'Location': 'Address',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    })
    
    df['Duration'] = pd.to_datetime(df['End Time']) - pd.to_datetime(df['Date & Time'])
    df['Duration'] = df['Duration'].dt.total_seconds()
    
        # Standardize the 'Call Type' values
    df['Call Type'] = df['Call Type'].replace({
        'INCOMING': 'InComing',
        'OUTGOING': 'OutGoing',
    })
    
    return df[['A-Party', 'B-Party', 'Date & Time', 'Call Type', 'Duration', 'Address', 'Latitude', 'Longitude']]

def preprocess_type5(df):
    df = df.rename(columns={
        'Call Type': 'Call Type',
        'A-Party': 'A-Party',
        'B-Party': 'B-Party',
        'Date & Time': 'Date & Time',
        'Duration': 'Duration',
        'Address': 'Address',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    })
    
    df['Call Type'] = df['Call Type'].replace({
        'Incoming': 'InComing',
        'Outgoing': 'OutGoing',
        'Incoming SMS': 'InComing SMS',
        'Outgoing SMS': 'OutGoing SMS'
    })
    
    return df[['A-Party', 'B-Party', 'Call Type', 'Date & Time', 'Duration', 'Address', 'Latitude', 'Longitude']]

# Main preprocessing function

def preprocess_file(df):
    if 'CALL_TYPE' in df.columns:
        print("type1")
        # 3134312323_1040470
        return preprocess_type1(df)
    elif 'Call Type' in df.columns and 'Longitude and Latitude' in df.columns:
        print("type2")
        # 923214104372
        return preprocess_type2(df)
    elif 'CallType' in df.columns:  
        print("type3")
        # 923054686127
        return preprocess_type3(df)
    elif 'A Number' in df.columns:
        print("type4")
        # 923244503235
        return preprocess_type4(df)
    elif 'A-Party' in df.columns and 'Call Type' in df.columns:
        print("type5")
        return preprocess_type5(df)
    else:
        print("Unknown file type.")
        return pd.DataFrame()  # Return empty DataFrame for unknown types
