import sys
import warnings
import numpy as np
import pandas as pd

# Configuration File

# City & Date
city = "Barcelona"
interestedMonth = "March"
interestedYear  = "2020"

monthList = { # Dictionary
    "January"  : "31",
    "February" : "28",
    "March"    : "31",
    "April"    : "30",
    "May"      : "31",
    "June"     : "30",
    "July"     : "31",
    "August"   : "31",
    "September": "30",
    "October"  : "31",
    "November" : "30",
    "December" : "31"
}

# Selected Parameters
selection = ['o3', 'no2', 'pm10']

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
np.set_printoptions(threshold=sys.maxsize)

