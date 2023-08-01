"""
Cleaning and wrangling the electricity demand data in order to use for modelling


This file includes the functions necessary to take the input format of the data and transform it into a usable format.


Author: Joshua Berg
"""

# imports
import pandas as pd
from utils import *

def clean_data(data):
    data['Time'] = pd.to_datetime(data['Time'])
    data['Day'] = data['Time'].dt.day
    data['Month'] = data['Time'].dt.month
    data['Hour'] = data['Time'].dt.hour
    data['Year'] = data['Time'].dt.year
    data.drop("Time", axis="columns", inplace=True)
    
    return data

if __name__ == "__main__":
    clean_data(pd.read_excel(get_root('data/Train/Actuals_part1.xlsx')))