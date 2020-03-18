import os
import sys
import math
import gmplot
import openaq
import folium #
import warnings
import geopandas
import numpy as np
import pandas as pd
import seaborn as sns
import geopy.distance
import matplotlib as mpl
import mplleaflet as mpll
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
#plt.rcParams['figure.figsize'] = (10, 10)
figx, figy = (10, 10)
markersize = 8

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
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in weightMatrix]))
# np.savetxt(filename + "_oe.txt", weightMatrix, delimiter='\t', fmt='%1.3f')

# Graph Creation
weightGraph = graphs.Graph(weightMatrix)


# Set Figure
fig, ax = plt.subplots(1, 2, figsize=(figx*2, figy))

# Style
sns.set(style='whitegrid', color_codes=True)

# Define Plot Type
sns.scatterplot(x="longitude", y="latitude", data=locationStation, ax=ax[0])
# Save Figure
extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(filename + interested_day + "_Location_Stations_" + analyze + ".png", bbox_inches=extent.expanded(1.2, 1.2))

# Plot Edges + Map
line_plot_ax = ax[1]
line_plot_ax.set_xlabel("longitude")
line_plot_ax.set_ylabel("latitude")

# Plot Edges.
for i in range(0, len(weightMatrix[0])):
    pointTmp = [weightMatrix[0][i], weightMatrix[1][i]]
    for j in range(i, len(weightMatrix[0])):
        if weightMatrix[i][j] > 0.0:
            #print("Distance between {} and {}: {}.".format(locationStation.loc[i].location, locationStation.loc[j].location, weightMatrix[i][j]))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[i].latitude, locationStation.loc[i].longitude))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[j].latitude, locationStation.loc[j].longitude))
            df2 = pd.DataFrame(np.array([[locationStation.loc[i].latitude, locationStation.loc[i].longitude], [locationStation.loc[j].latitude, locationStation.loc[j].longitude]]), columns=["longitude", "latitude"])
            line_plot_ax.plot(df2.latitude, df2.longitude, 'black')

line_plot_ax.plot(locationStation.longitude, locationStation.latitude, 'bo') # Black Circles
#for i, point in locationStation.iterrows():
#    line_plot_ax.text(point['longitude']+0.01, point['latitude'], str(point['location']))


extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(filename + interested_day + "_Location_Stations_Graph_" + analyze + ".png", bbox_inches=extent.expanded(1.2, 1.2))
#plt.show()

min_lat = locationStation["latitude"].min()
max_lat = locationStation["latitude"].max()
min_lon = locationStation["longitude"].min()
max_lon = locationStation["longitude"].max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

m = folium.Map(location=[center_lat, center_lon], tiles='openstreetmap', zoom_start=10, control_scale=True)

for i in range(0, len(locationStation)):
    folium.Marker([locationStation.iloc[i]['latitude'], locationStation.iloc[i]['longitude']], popup=locationStation.iloc[i]['location']).add_to(m)

m.save(filename + interested_day + "_location_station_" + analyze + "_map.html")
