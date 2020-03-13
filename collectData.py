import os
import sys
import math
import openaq
import warnings
import numpy as np
import pandas as pd
import geopy.distance
# import seaborn as sns
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

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 5)

np.set_printoptions(threshold=sys.maxsize)

# Check versions
print("Versions of the libraries:\n")
print("OpenAQ v{}".format(openaq.__version__))
print("Pandas v{}".format(pd.__version__))
# print("Seaborn v{}".format(sns.__version__))
# print("Matplot v{}".format(mpl.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis program gathers information of some of the following parameters:\n")
param = api.parameters(df=True)
print(param)
print(selection)
print()

# yy/mm/dd
date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
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
dataO3Day.reset_index(drop = True, inplace = True)
dataO3Day.to_csv(filename + interested_day + "_per_hours_O3_" + city + ".csv", sep=',')

dataNO2Day  = resCityDay[resCityDay.parameter.str.contains('no2')]
dataNO2Day.reset_index(drop = True, inplace = True)
dataNO2Day.to_csv(filename + interested_day + "_per_hours_NO2_" + city + ".csv", sep=',')

dataPM10Day = resCityDay[resCityDay.parameter.str.contains('pm10')]
dataPM10Day.reset_index(drop = True, inplace = True)
dataPM10Day.to_csv(filename + interested_day + "_per_hours_PM10_" + city + ".csv", sep=',')

columnsClean = {"country", "date"}

#####################
# Graph Creation O3 #
#####################

## Cleaning Location O3 Data
locationO3 = dataO3Day.drop_duplicates(subset="location")
locationO3 = locationO3.drop(columns=columnsClean, axis=1)

## Distances between O3 stations
pointsO3 = [locationO3['longitude'].tolist(), locationO3['latitude'].tolist()]
sizeO3 = len(locationO3)
WO3 = np.zeros((sizeO3, sizeO3))

for i in range(0, len(pointsO3[0])):
    pointTmp = [pointsO3[0][i], pointsO3[1][i]]
    for j in range(i, len(pointsO3[0])):
        nextPointTmp = [[pointsO3[0][j], pointsO3[1][j]]]
        WO3[i][j] = round(geopy.distance.vincenty(pointTmp, nextPointTmp).km, 3)

## Symmetric Matrix
WO3 = WO3 + WO3.T

## Graph Construction
GO3 = graphs.Graph(WO3)

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

## Graph Construction
GNO2 = graphs.Graph(WNO2)

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

## Graph Construction
GPM10 = graphs.Graph(WPM10)
