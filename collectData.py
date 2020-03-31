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
date_to        = "2020-03-08T22:00:00"

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
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

# Filename
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# Obtaining measurements of the sensors in a City
resCityDay = api.measurements(city=city, parameter=selection, date_from=date_from, date_to=date_to, order_by="date", limit=10000, df=True)

# Delete unwanted columns
columnsToDelete = ["unit", "date.utc"]
columnsToRename = {"coordinates.latitude" : "latitude", "coordinates.longitude" : "longitude"}

columnsSensorsToDelete = ["id", "country", "city", "cities", "sourceName", "sourceNames", "sourceType", "sourceTypes", 
                          "firstUpdated", "lastUpdated", "countsByMeasurement", "coordinates.longitude", "coordinates.latitude", "parameters", "count"]
columnsSensorsToRename = {"locations" : "spot"}
infoSensor = api.locations(city=city, df=True)
infoSensor = infoSensor.drop(columnsSensorsToDelete, axis=1)
infoSensor = infoSensor.rename(columns=columnsSensorsToRename)

for index in range(0, len(infoSensor)):
    infoSensor["spot"][index] = infoSensor["spot"][index][0]

print(infoSensor)

resCityDay = resCityDay.drop(columnsToDelete, axis=1)
resCityDay = resCityDay.rename(columns=columnsToRename)
resCityDay["date"] = resCityDay.index
resCityDay["date"] = resCityDay["date"].dt.strftime("%Y_%m_%d-%H:%M:%S")
resCityDay.reset_index(drop = True, inplace = True)

# Merge with spots
resCityDay = resCityDay.merge(infoSensor, on="location", how="left")
# Maybe resCityDay should be sorted by location and date?
resCityDay.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
resCityDay.reset_index(drop=True, inplace=True)
resCityDay.to_csv(filename + interested_day + "_per_hours_" + city + ".csv", sep=',')

dataO3Day = resCityDay[resCityDay.parameter.str.contains('o3')]
dataO3Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataO3Day.reset_index(drop=True, inplace=True)
dataO3Day.to_csv(filename + interested_day + "_per_hours_o3_" + city + ".csv", sep=',')

dataNO2Day  = resCityDay[resCityDay.parameter.str.contains('no2')]
dataNO2Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataNO2Day.reset_index(drop=True, inplace=True)
dataNO2Day.to_csv(filename + interested_day + "_per_hours_no2_" + city + ".csv", sep=',')

dataPM10Day = resCityDay[resCityDay.parameter.str.contains('pm10')]
dataPM10Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataPM10Day.reset_index(drop=True, inplace=True)
dataPM10Day.to_csv(filename + interested_day + "_per_hours_pm10_" + city + ".csv", sep=',')

print("Data collected sucessfully.")