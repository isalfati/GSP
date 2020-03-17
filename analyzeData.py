import os
import sys
import math
import gmplot
import openaq
import warnings
import geopandas
import numpy as np
import pandas as pd
import geopy.distance

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# Parameter to be analyzed
analyze = 'o3'

# Global Variables
city = "Barcelona"
interested_day = "2020-03-08"
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# Configuration Parameters
## Avoid warnings
warnings.simplefilter(action='ignore')

## Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.precision = 4
cartoplot = 0.01

#plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 10)

# Suppress=True allows to not use scientific notation when reading from CSV.
np.set_printoptions(threshold=np.inf, suppress=True)

## Cleaning
columnsClean = {"country", "date", "value", "parameter"}

# Data Analysis
dataParam = pd.read_csv(filename + interested_day + "_per_hours_" + analyze + "_" + city + ".csv", sep=',', index_col=0)

locationStation = dataParam.drop_duplicates(subset="location")
locationStation = locationStation.drop(columns=columnsClean, axis=1)
locationStation.reset_index(drop=True, inplace=True)
#print(locationStation['location'].unique())

# Distances between stations that measure the parameter to analyze
points = [locationStation['longitude'].tolist(), locationStation['latitude'].tolist()]
size   = len(locationStation)
print(locationStation.dtypes)
weightMatrix = np.zeros((size, size))

for i in range(0, len(points[0])):
    pointTmp = [points[0][i], points[1][i]]
    for j in range(i, len(points[0])):
        nextPointTmp = [[points[0][i], points[1][j]]]
        weightMatrix[i][j] = round(geopy.distance.vincenty(pointTmp, nextPointTmp).km, 3)


# Symmetric Matrix
weightMatrix = weightMatrix + weightMatrix.T

# VIP.
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in weightMatrix]))
#np.savetxt(filename + "_oe.txt", weightMatrix, delimiter='\t', fmt='%1.3f')

print(weightMatrix[5][6])