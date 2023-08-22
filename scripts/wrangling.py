"""
Cleaning and wrangling the electricity demand data in order to use for modelling


This file includes the functions necessary to take the input format of the data and transform it into a usable format.


Author: Joshua Berg
"""

# imports
import pandas as pd
from utils import *

def merge_data(data1, data2):
    """
    Merge the two given datasets
    """
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
    data.to_csv(get_root('data/elec_p4_dataset/Train/merged_actuals.csv'), index=False)
