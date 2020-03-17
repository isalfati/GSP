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

# Global Variables
city = "Barcelona"
selection = ['o3', 'no2', 'pm10'] #PM25 does not exist in Barcelona

# 24 hour period
interested_day = "2020-03-08"
date_from      = "2020-03-07T23:00:00"
date_to        = "2020-03-08T22:59:00"

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
cartoplot = 0.01

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 10)

np.set_printoptions(threshold=sys.maxsize)

# Check versions
print("Versions of the libraries:\n")
print("OpenAQ  v{}.".format(openaq.__version__))
print("Pandas  v{}.".format(pd.__version__))
print("Geopy   v{}.".format(geopy.__version__))
print("Cartopy v{}.".format(cartopy.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis program gathers information of some of the following parameters:\n")
param = api.parameters(df=True)
print(param)
print(selection)
print()

# yy/mm/dd

filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# Obtaining measurements of the sensors in a City
resCityDay = api.measurements(city=city, parameter=selection, date_from=date_from, date_to=date_to, order_by="date", limit=10000, df=True)

# Delete unwanted columns
columnsToDelete = ["unit", "date.utc"]
columnsToRename = {"coordinates.latitude" : "latitude", "coordinates.longitude" : "longitude"}
resCityDay = resCityDay.drop(columnsToDelete, axis=1)
resCityDay = resCityDay.rename(columns=columnsToRename)
resCityDay["date"] = resCityDay.index
resCityDay["date"] = resCityDay["date"].dt.strftime("%Y_%m_%d-%H:%M:%S")
resCityDay.reset_index(drop = True, inplace = True)

resCityDay.to_csv(filename + interested_day + "_per_hours_" + city + ".csv", sep=',')


dataO3Day = resCityDay[resCityDay.parameter.str.contains('o3')]
dataO3Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataO3Day.reset_index(drop = True, inplace = True)
dataO3Day.to_csv(filename + interested_day + "_per_hours_o3_" + city + ".csv", sep=',')

dataNO2Day  = resCityDay[resCityDay.parameter.str.contains('no2')]
dataNO2Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataNO2Day.reset_index(drop = True, inplace = True)
dataNO2Day.to_csv(filename + interested_day + "_per_hours_no2_" + city + ".csv", sep=',')

dataPM10Day = resCityDay[resCityDay.parameter.str.contains('pm10')]
dataPM10Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataPM10Day.reset_index(drop = True, inplace = True)
dataPM10Day.to_csv(filename + interested_day + "_per_hours_pm10_" + city + ".csv", sep=',')

#################################################################################################################################################################
"""
columnsClean = {"country", "date", "value", "parameter"}

#####################
# Graph Creation O3 #
#####################

## Cleaning Location O3 Data
locationO3 = dataO3Day.drop_duplicates(subset="location")
locationO3 = locationO3.drop(columns=columnsClean, axis=1)
print(locationO3)

## Distances between O3 stations
pointsO3 = [locationO3['longitude'].tolist(), locationO3['latitude'].tolist()]
print(locationO3.dtypes)
sizeO3 = len(locationO3)
WO3 = np.zeros((sizeO3, sizeO3))

for i in range(0, len(pointsO3[0])):
    pointTmp = [pointsO3[0][i], pointsO3[1][i]]
    for j in range(i, len(pointsO3[0])):
        nextPointTmp = [[pointsO3[0][j], pointsO3[1][j]]]
        WO3[i][j] = round(geopy.distance.vincenty(pointTmp, nextPointTmp).km, 3)
        print(WO3[i][j])

## Symmetric Matrix
WO3 = WO3 + WO3.T
print(np.matrix(WO3))

## Graph Construction
GO3 = graphs.Graph(WO3)
#print('{} nodes, {} edges'.format(GO3.N, GO3.Ne))

print("\nBasic Weighted Graph info:")
print("Is connected? Answer: {}".format(GO3.is_connected()))
print("Is directed? Anser: {}".format(GO3.is_directed()))

min_lat = locationO3["latitude"].min()
max_lat = locationO3["latitude"].max()
min_lon = locationO3["longitude"].min()
max_lon = locationO3["longitude"].max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

print("\nBoundaries GeoMap O3 in {}:".format(city))
print("Minimum Latitude: {}, Maximum Latitude: {}.".format(min_lat, max_lat))
print("Minimum Longitude: {}, Maximum Longitude: {}.".format(min_lon, max_lon))
print("Medium Latitude: {}, Medium Longitude: {}.".format(center_lat, center_lon))

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.set_extent([min_lon-cartoplot, max_lon+cartoplot, min_lat-cartoplot, max_lat+cartoplot])
ax.set_title("Location of O3 Stations", pad=25)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

# Plotting by COORDINATES
gdfO3 = geopandas.GeoDataFrame(locationO3, geometry=geopandas.points_from_xy(locationO3.longitude, locationO3.latitude))
gdfO3.plot(ax=ax, color='red')
plt.show()

dataO3_00 = dataO3Day[dataO3Day.date.str.contains('2020_03_08-00:00:00')]
dataO3_00.reset_index(drop = True, inplace = True)
print(dataO3_00)

fig2 = plt.figure(figsize=(10,8))
x = dataO3_00['location']
plt.xticks(rotation=45)
values = dataO3_00['value']

plt.stem(x, values)
plt.show()

######################
# Graph Creation NO2 #
######################

## Cleaning Location NO2 Data
locationNO2 = dataNO2Day.drop_duplicates(subset="location")
locationNO2 = locationNO2.drop(columns=columnsClean, axis=1)

## Distances between NO2 stations
pointsNO2 = [locationNO2['longitude'].tolist(), locationNO2['latitude'].tolist()]
sizeNO2 = len(locationNO2)
WNO2 = np.zeros((sizeNO2, sizeNO2))

for i in range(0, len(pointsNO2[0])):
    pointTmp = [pointsNO2[0][i], pointsNO2[1][i]]
    for j in range(i, len(pointsNO2[0])):
        nextPointTmp = [[pointsNO2[0][j], pointsNO2[1][j]]]
        WNO2[i][j] = round(geopy.distance.vincenty(pointTmp, nextPointTmp).km, 3)

## Symmetric Matrix
WNO2 = WNO2 + WNO2.T
#print(np.matrix(WNO2))

## Graph Construction
GNO2 = graphs.Graph(WNO2)

print("\nBasic Weighted Graph info:")
print("Is connected? Answer: {}".format(GNO2.is_connected()))
print("Is directed? Anser: {}".format(GNO2.is_directed()))

min_lat = locationNO2["latitude"].min()
max_lat = locationNO2["latitude"].max()
min_lon = locationNO2["longitude"].min()
max_lon = locationNO2["longitude"].max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

print("\nBoundaries GeoMap NO2 in {}:".format(city))
print("Minimum Latitude: {}, Maximum Latitude: {}.".format(min_lat, max_lat))
print("Minimum Longitude: {}, Maximum Longitude: {}.".format(min_lon, max_lon))
print("Medium Latitude: {}, Medium Longitude: {}.".format(center_lat, center_lon))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.set_extent([min_lon-cartoplot, max_lon+cartoplot, min_lat-cartoplot, max_lat+cartoplot])
ax.set_title("Location of NO2 Stations", pad=25)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

# Plotting by COORDINATES
gdfO3 = geopandas.GeoDataFrame(locationNO2, geometry=geopandas.points_from_xy(locationNO2.longitude, locationNO2.latitude))
gdfO3.plot(ax=ax, color='red')

plt.show()

####################### 
# Graph Creation PM10 #
#######################

## Cleaning Location PM10 Data
locationPM10 = dataPM10Day.drop_duplicates(subset="location")
locationPM10 = locationPM10.drop(columns=columnsClean, axis=1)

## Distances between NO2 stations
pointsPM10 = [locationPM10['longitude'].tolist(), locationPM10['latitude'].tolist()]
sizePM10 = len(locationPM10)
WPM10 = np.zeros((sizePM10, sizePM10))

for i in range(0, len(pointsPM10[0])):
    pointTmp = [pointsPM10[0][i], pointsPM10[1][i]]
    for j in range(i, len(pointsPM10[0])):
        nextPointTmp = [[pointsPM10[0][j], pointsPM10[1][j]]]
        WPM10[i][j] = round(geopy.distance.vincenty(pointTmp, nextPointTmp).km, 3)

## Symmetric Matrix
WPM10 = WPM10 + WPM10.T
print(np.matrix(WPM10))

## Graph Construction
GPM10 = graphs.Graph(WPM10)

print("\nBasic Weighted Graph info:")
print("Is connected? Answer: {}".format(GPM10.is_connected()))
print("Is directed? Anser: {}".format(GPM10.is_directed()))

min_lat = locationPM10["latitude"].min()
max_lat = locationPM10["latitude"].max()
min_lon = locationPM10["longitude"].min()
max_lon = locationPM10["longitude"].max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

print("\nBoundaries GeoMap PM10 in {}:".format(city))
print("Minimum Latitude: {}, Maximum Latitude: {}.".format(min_lat, max_lat))
print("Minimum Longitude: {}, Maximum Longitude: {}.".format(min_lon, max_lon))
print("Medium Latitude: {}, Medium Longitude: {}.".format(center_lat, center_lon))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.set_extent([min_lon-cartoplot, max_lon+cartoplot, min_lat-cartoplot, max_lat+cartoplot])
ax.set_title("Location of PM10 Stations", pad=25)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

# Plotting by COORDINATES
gdfO3 = geopandas.GeoDataFrame(locationPM10, geometry=geopandas.points_from_xy(locationPM10.longitude, locationPM10.latitude))
gdfO3.plot(ax=ax, color='red')

plt.show()
"""