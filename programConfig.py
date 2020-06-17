import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Configuration File

# Filename
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

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

############################### Parameters to be modified ###############################

# City & Date       # If city = BCN, month = March, the small dataset test will be used instead.
city = "Barcelona"  # Right now, Barcelona is the only one possible.
month = "February"   # January, February and March possible.
year  = "2020"      # Only 2020 possible.

# Parameter to analyze
contaminant = "pm10" # any of the selection list.

# Minimum amount of data for a station to be kept 0 < x < 1
minPercentatge = 0.85

# Maximum distance between stations
maxDistanceBetweenStations = 20 # km 

# Target Station Selector:
# [0]: Station to reconstruct, [1]: Faulty Station
# Two scenarios, Faulty station impacts (adjacent) or it doesn't.
targetStation = [1, 3]

# Parameters of faulty station
mu, sigma = 0, 255

#########################################################################################
 
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

# Value of K to reconstruct
VALUEK = 8

# Error Percentage
levelError = 0.20

