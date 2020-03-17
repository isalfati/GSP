import os
import sys
import math
import gmplot
import openaq
import warnings
import geopandas
import numpy as np
import pandas as pd
import seaborn as sns
import geopy.distance

import mplleaflet as mpll
import folium

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting


# Parameter to be analyzed
analyze = 'no2'

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
pd.options.display.precision = 13
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
points = [locationStation['latitude'].tolist(), locationStation['longitude'].tolist()]
print(points)
size   = len(locationStation)
print(locationStation)
weightMatrix = np.zeros((size, size))

for i in range(0, len(points[0])):
    pointTmp = [points[0][i], points[1][i]]
    #print("From {}.".format(pointTmp))
    for j in range(i, len(points[0])):
        nextPointTmp = [[points[0][j], points[1][j]]]
        #print(" |-> To {}.".format(nextPointTmp))
        distance = round(geopy.distance.VincentyDistance(pointTmp, nextPointTmp).km, 4) # LAT1, LON1, LAT2, LON2
        weightMatrix[i][j] = distance if distance <= 15.0 else 0.0 #example
        #weightMatrix[i][j] = distance

# Symmetric Matrix
weightMatrix = weightMatrix + weightMatrix.T

# VIP.
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in weightMatrix]))
# np.savetxt(filename + "_oe.txt", weightMatrix, delimiter='\t', fmt='%1.3f')

weightGraph = graphs.Graph(weightMatrix)
#sns.set(style='darkgrid', color_codes=True)
#sns.jointplot(x="longitude", y="latitude", data=locationStation)
#plt.show()

#plot:


line_plot_fig, line_plot_ax = plt.subplots(figsize=(12,9))

for i in range(0, len(weightMatrix[0])):
    pointTmp = [weightMatrix[0][i], weightMatrix[1][i]]
    for j in range(i, len(weightMatrix[0])):
        if weightMatrix[i][j] > 0.0:
            #print("Distance between {} and {}: {}.".format(locationStation.loc[i].location, locationStation.loc[j].location, weightMatrix[i][j]))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[i].latitude, locationStation.loc[i].longitude))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[j].latitude, locationStation.loc[j].longitude))
            df2 = pd.DataFrame(np.array([[locationStation.loc[i].latitude, locationStation.loc[i].longitude], [locationStation.loc[j].latitude, locationStation.loc[j].longitude]]), columns=["longitude", "latitude"])
            line_plot_ax.plot(df2.latitude, df2.longitude, 'black')

#line_plot_ax.plot(locationStation.longitude, locationStation.latitude, 'black')
line_plot_ax.plot(locationStation.longitude, locationStation.latitude, 'bo', markersize=8) # Black Circles
#for i, point in locationStation.iterrows():
#    line_plot_ax.text(point['longitude']+0.01, point['latitude'], str(point['location']))

# Show and save
plt.show()
#mpll.show(fig=line_plot_fig, path=filename + interested_day + "_location_station_" + analyze + "_map.html")
mpll.save_html(fig=line_plot_fig, fileobj=filename + interested_day + "_location_station_" + analyze + "_map.html")

