import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import csv

# Load the data from the CSV file
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    names = ['time','open','close','high','low','volume']
    
    print(df)
    return df

# Clean the data - Initial resample_data of the original data set, after using reindex and ffill functions to minimise missing data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  
    """Clean the data."""

    # Complete time range from the start to the end of data at 1-minute intervals
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')

    # Reindex the DataFrame to have a row for each minute in the full time range
    df = df.reindex(full_time_range)

    # Forward fill to handle the missing data after reindexing
    df.ffill(inplace=True)

    # Aggregate the data as before
    df = df.resample('1T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Sort the DataFrame by the 'time' index just in case
    df = df.sort_index()

    print('df.isna().sum()')
        
    return df
    print(df)

# Second stage of resampling through multi_index function to create new data points at 5, 15, 30, 60, 240, and 1440 minutes
# We also remove nan values using ffill and bfill
def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    
    """Resample the data to create new data points."""
    
    print(df)
    
    # Define the time frames and corresponding new column names
    time_frames = ['5T', '15T', '30T', '60T', '240T', '1440T']  # in minutes
    new_columns = ['5_min', '15_min', '30_min', '1_hour', '4_hour', '1_day']
    columns_to_resample = ['open', 'close', 'high', 'low', 'volume']

    # Initialise a multi_Index for the new columns
    tuples = [(col, sub_col) for col in new_columns for sub_col in columns_to_resample]
    multi_index = pd.MultiIndex.from_tuples(tuples)
    resampled_df = pd.DataFrame(index=df.index, columns=multi_index)

    # Assigning resampled values to new columns and sub-columns
    for time_frame, col in zip(time_frames, new_columns):
        df_resampled = df.resample(time_frame).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Forward fill the resampled data to avoid NaN values
        df_resampled.ffill(inplace=True)

        print(df_resampled)
        
        # Assign resampled values to sub-columns
        for sub_col in columns_to_resample:
            resampled_df[(col, sub_col)] = df_resampled[sub_col]

    # Combine the original df with the resampled_df
    df_combined = pd.concat([df, resampled_df], axis=1)

    # Reorder the columns to match the expected order
    column_order = ['open', 'close', 'high', 'low', 'volume'] + list(resampled_df.columns)
    df_combined = df_combined[column_order]

    # Forward fill again to handle any NaN values after reindexing
    df_combined.ffill(inplace=True)
    
    # Back fill to complete the missing data
    df_combined.bfill(inplace=True)

    # Check for NaN values
    nan_values = df_combined.isna().sum()
    print(nan_values)

    # Return the modified DataFrame
    return df_combined

# Load the data
file_path = r'C:\Users\Shadow\.cursor-tutor\projects\Machine Learning Modules\btcusd_ISO8601.csv'
df = load_data(file_path)

# Clean the data
df = clean_data(df)

# Resample the data
df = resample_data(df)

# Check the head of the DataFrame
print(df.head)

# save the file
df.to_csv('BTCUSD_CLEANED.csv', index=True)

# Plot the data
plt.figure(figsize=(15, 5))
plt.plot(df['close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.savefig('bitcoin_close_price.png')
