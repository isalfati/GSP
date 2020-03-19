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

#@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ GLOBAL VARIABLES @@@
#@@@@@@@@@@@@@@@@@@@@@@@@

# Parameter to be analyzed
analyze = 'pm10'

# City, Day, Output location, maximum distance between stations
city = "Barcelona"
interested_day = "2020-03-08"
interestedDayTimestamp = "2020_03_08"
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"
maxDistance = 15.0

# Figures
figx, figy = (10, 10)

# Cleaning columns
columnsClean = {"country", "date", "value", "parameter"}

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Configuration Parameters @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.precision = 10
decimals = 4
decimalsSparse = 3

# Suppress=True allows to not use scientific notation when reading from CSV.
np.set_printoptions(threshold=np.inf, suppress=True)

#@@@@@@@@@@@@@@@@
#@@@ Analysis @@@
#@@@@@@@@@@@@@@@@

# Import data
dataParam = pd.read_csv(filename + interested_day + "_per_hours_" + analyze + "_" + city + ".csv", sep=',', index_col=0)

# locationStation contains a Dataframe with [location(identifier), city, latitude, longitude]
locationStation = dataParam.drop_duplicates(subset="location")
locationStation = locationStation.drop(columns=columnsClean, axis=1)
locationStation.reset_index(drop=True, inplace=True)
#print("List of stations that measures {}: {}".format(analyze, locationStation['location'].unique()))

# List of timestamps of a day.
listTimeStamps = []
for i in range(0, 24):
    listTimeStamps.append(interestedDayTimestamp + "-0" + str(i) + ":00:00") if i < 10 else listTimeStamps.append(interestedDayTimestamp + '-' + str(i) + ":00:00")
    
# Check if any station has missing measurement time, this is useful for reconstructing signals of missing data.
for i in locationStation['location'].unique():
    iterList = dataParam[dataParam.location.str.contains(i)]
    iterTimeList = iterList.date.tolist()
    missingMeasurements = [elem for elem in listTimeStamps if elem not in iterTimeList]
    if missingMeasurements:
        print("The station with identifier {} misses the following time measurements {}.".format(i, missingMeasurements))

# Building Weight Matrix a.k.a. adjacency matrix with weights
# Each element of this matrix measures the distance between station i and station j.
# If they are separated more than a certain distance, there will be no connection between i and j.
points = [locationStation['latitude'].tolist(), locationStation['longitude'].tolist()]
size   = len(locationStation)
weightMatrix = np.zeros((size, size))

for i in range(0, len(points[0])):
    pointTmp = [points[0][i], points[1][i]]
    #print("From {}.".format(pointTmp))
    for j in range(i, len(points[0])):
        nextPointTmp = [[points[0][j], points[1][j]]]
        #print(" |-> To {}.".format(nextPointTmp))
        distance = round(geopy.distance.VincentyDistance(pointTmp, nextPointTmp).km, 4) # LAT1, LON1, LAT2, LON2
        weightMatrix[i][j] = distance if distance <= maxDistance else 0.0 #example
        #weightMatrix[i][j] = distance

# Because we want undirected graphs, we add the transposed matrix
weightMatrix = weightMatrix + weightMatrix.T
print("\nWeighted Matrix:")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in weightMatrix]))

# Graph Creation
Graph = graphs.Graph(weightMatrix)

# Access Laplacian Matrix.
LaplacianMatrix = Graph.L
print("\nLaplacian Matrix:")
print('\n'.join(['\t'.join([str(round(cell, decimalsSparse)) for cell in row]) for row in LaplacianMatrix.toarray()]))

"""
# Set Figure
fig, ax = plt.subplots(1, 2, figsize=(figx*2, figy))

# Style
sns.set(style='whitegrid', color_codes=True)

# Define Plot Type
sns.scatterplot(x="longitude", y="latitude", data=locationStation, ax=ax[0])
# Save Figure
extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(filename + interested_day + "_Location_Stations_" + analyze + "_" + city+ ".png", bbox_inches=extent.expanded(1.2, 1.2))

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
plt.savefig(filename + interested_day + "_Location_Stations_Graph_" + analyze + "_" + city + ".png", bbox_inches=extent.expanded(1.2, 1.2))
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

m.save(filename + interested_day + "_location_station_" + analyze + "_" + city + "_map.html")

plt.show()
"""