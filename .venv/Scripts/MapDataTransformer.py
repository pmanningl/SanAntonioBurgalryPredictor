import jsoncomm
import re
from datetime import datetime
import numpy as np
import pandas as pd

def data_transformer(file_path):
    print("-----TRANSFORMING DATA")
    # Load JSON data from the file
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            df = pd.DataFrame(transformer(data))
            df_normalized, lat_min, lat_max, lon_min, lon_max = normalize_lat_long(df)
            df_normalized.to_csv("transformed_data.csv", sep='\t', encoding='utf-8')
            bounds_data = {
                'lat_min': [lat_min],
                'lat_max': [lat_max],
                'lon_min': [lon_min],
                'lon_max': [lon_max]
            }

            df_bounds = pd.DataFrame(bounds_data)

            # Save the bounds data to a CSV file
            df_bounds.to_csv("bounds_data.csv", index=False)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file '{file_path}'.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


def transformer(data):
    burglaryList = []
    jsonShortcut = data['data']['data']['pins']
    for burglary in jsonShortcut:
        burglaryDict = {'ReferenceID': jsonShortcut[burglary]['ReferenceID'],
                        'Latitude': float(jsonShortcut[burglary]['Latitude']),
                        'Longitude': float(jsonShortcut[burglary]['Longitude']),
                        'DateTime': jsonShortcut[burglary]['DateTime'],
                        **date_to_cyclical(jsonShortcut[burglary]['DateTime'])
                        }
        burglaryList.append(burglaryDict)
    return burglaryList

#--------------------------------------------------------------------------------

def normalize_lat_long(df):
    # Ensure the DataFrame contains 'Latitude' and 'Longitude' columns
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise ValueError("DataFrame must contain 'Latitude' and 'Longitude' columns")

    # Normalize Latitude
    lat_min = df['Latitude'].min()
    lat_max = df['Latitude'].max()
    df['Latitude'] = 2 * (df['Latitude'] - lat_min) / (lat_max - lat_min) - 1

    # Normalize Longitude
    lon_min = df['Longitude'].min()
    lon_max = df['Longitude'].max()
    df['Longitude'] = 2 * (df['Longitude'] - lon_min) / (lon_max - lon_min) - 1

    return df, lat_min, lat_max, lon_min, lon_max


def date_to_cyclical(date_str):
    # Define the input string and pattern
    pattern = r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.\d{3}"

    # Extract values
    match = re.match(pattern, date_str)
    if not match:
        raise ValueError("Date string does not match the expected format")

    year, month, day, hour, minute, _ = map(int, match.groups())

    # Create a datetime object
    dt = datetime(year, month, day, hour, minute)

    # Convert time to an angle
    total_minutes_in_day = 24 * 60
    time_angle = ((hour * 60 + minute) / total_minutes_in_day) * 2 * np.pi  # Convert to radians

    # Convert day of the week to an angle
    day_of_week_angle = (dt.weekday() / 7) * 2 * np.pi  # Convert to radians (0=Monday, 6=Sunday)

    # Calculate sine and cosine components
    cyclical_features = {
        'day_of_week_sin': np.sin(day_of_week_angle),
        'day_of_week_cos': np.cos(day_of_week_angle),
        'time_sin': np.sin(time_angle),
        'time_cos': np.cos(time_angle),
    }
    return cyclical_features


data_transformer("map_data.json")
#-----------------------------------------------------------------------------------


