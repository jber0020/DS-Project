"""
Cleaning and wrangling the electricity demand data in order to use for modelling


This file includes the functions necessary to take the input format of the data and transform it into a usable format.


Author: Joshua Berg
"""

# imports
import pandas as pd
from utils import *

actuals_part1 = pd.read_excel(get_root('data/Train/Actuals_part1.xlsx'))


