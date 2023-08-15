"""
Cleaning and wrangling the electricity demand data in order to use for modelling


This file includes the functions necessary to take the input format of the data and transform it into a usable format.


Author: Joshua Berg
"""

# imports
import pandas as pd
from utils import *

def clean_data(data):
    # Add previous day's value
    data['Lag_1day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=1)), 
                                                        'Load (kW)'].values[0] if (x - pd.DateOffset(days=1)) in data['Time'].values else None)
    data['Lag_2day'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=2)), 
                                                        'Load (kW)'].values[0] if (x - pd.DateOffset(days=2)) in data['Time'].values else None)
    data['Lag_1week'] = data['Time'].apply(lambda x: data.loc[data['Time'] == (x - pd.DateOffset(days=7)), 
                                                        'Load (kW)'].values[0] if (x - pd.DateOffset(days=7)) in data['Time'].values else None)
    # Remove first week
    cutoff = data['Time'].dt.date.min() + pd.DateOffset(days=7)
    data = data[data['Time'].dt.date >= cutoff.date()]

    # Clean time values
    data['Day'] = data['Time'].dt.day
    data['Month'] = data['Time'].dt.month
    data['Hour'] = data['Time'].dt.hour
    data['Year'] = data['Time'].dt.year
    data['Weekday'] = data['Time'].dt.weekday
    data.drop("Time", axis="columns", inplace=True)
    
    return data

def merge_data(data1, data2):
    merged = pd.concat([data1, data2], ignore_index=True)
    merged = merged.sort_values('Time')
    merged = merged.drop_duplicates(subset='Time', keep='last')
    return merged

if __name__ == "__main__":
    data1 = pd.read_excel(get_root('data/elec_p4_dataset/Train/Actuals_part1.xlsx'))
    data2 = pd.read_csv(get_root('data/elec_p4_dataset/Train/Actuals_part2.csv'))

    data1['Time'] = pd.to_datetime(data1['Time'])
    data2['Time'] = pd.to_datetime(data2['Time'])

    data = merge_data(data1, data2)
    data= clean_data(data)
    data.to_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'), index=False)
