import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Configuration File

# Filename
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# City & Date
city = "Barcelona"
interestedMonth = "March"
interestedYear  = "2020"

monthList = { # Dictionary
    "January"  : 31,
    "February" : 28,
    "March"    : 31,
    "April"    : 30,
    "May"      : 31,
    "June"     : 30,
    "July"     : 31,
    "August"   : 31,
    "September": 30,
    "October"  : 31,
    "November" : 30,
    "December" : 31
}

# Selected Parameters
selection = ["o3", "no2", "pm10"]

# Parameter to analyze
paramAnalyzed = "no2"

# Avoid warnings
warnings.simplefilter(action="ignore")

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.precision = 10

# Suppress=True allows to not use scientific notation when reading from CSV.
np.set_printoptions(threshold=np.inf, suppress=True)

# Display number options
decimals = 4
decimalsSparse = 3