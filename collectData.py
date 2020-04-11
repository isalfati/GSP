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
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# Global Variables
city = "Barcelona"
selection = ['o3', 'no2', 'pm10'] #PM25 does not exist in Barcelona

interestedYear  = "2020"
interestedMonth = "March"

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

api = openaq.OpenAQ()

# List of Parameters
print("\nThis program gathers information of some of the following parameters:\n")
param = api.parameters(df=True)
print(param)
print(selection)
print()

# Filename
filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# Obtaining measurements of the sensors in a City during a Certain Month
dataParam = pd.DataFrame()
monthIndex = list(monthList.keys()).index(interestedMonth) + 1
monthIndex = "0" + str(monthIndex) if monthIndex < 10 else str(monthIndex)
partialDate = interestedYear + "-" + monthIndex + "-"

for day in range(1, int(monthList[interestedMonth]) + 1):
    print("Wait a moment, please...")
    partialDay = "0" + str(day) if day < 10 else str(day)
    auxDateFrom = partialDate + partialDay
    auxDateTo   = partialDate + partialDay + "T23:00:00"
    tmpDF = api.measurements(city=city, parameter=selection, date_from=auxDateFrom, date_to=auxDateTo, order_by="date", limit=10000, df=True)
    dataParam = dataParam.append(tmpDF)

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

dataParam = dataParam.drop(columnsToDelete, axis=1)
dataParam = dataParam.rename(columns=columnsToRename)
dataParam["date"] = dataParam.index
dataParam["date"] = dataParam["date"].dt.strftime("%Y_%m_%d-%H:%M:%S")
dataParam.reset_index(drop = True, inplace = True)

# Merge with spots
dataParam = dataParam.merge(infoSensor, on="location", how="left")
# Maybe dataParam should be sorted by location and date?
dataParam.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataParam.reset_index(drop=True, inplace=True)
dataParam.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours.csv", sep=',')

dataO3Day = dataParam[dataParam.parameter.str.contains('o3')]
dataO3Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataO3Day.reset_index(drop=True, inplace=True)
dataO3Day.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours_o3.csv", sep=',')

dataNO2Day  = dataParam[dataParam.parameter.str.contains('no2')]
dataNO2Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataNO2Day.reset_index(drop=True, inplace=True)
dataNO2Day.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours_no2.csv", sep=',')

dataPM10Day = dataParam[dataParam.parameter.str.contains('pm10')]
dataPM10Day.sort_values(['location', 'date'], ascending=[True, True], inplace=True)
dataPM10Day.reset_index(drop=True, inplace=True)
dataPM10Day.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours_pm10.csv", sep=',')

print("Data collected sucessfully.")