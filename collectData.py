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

# Obtaining the latest values of the sensors in a City
resCityDay = api.measurements(city=city, parameter=selection, date_from=date_from, date_to=date_to, order_by="date", limit=10000, df=True)




# Delete unwanted columns
columnsToDelete = ["unit", "date.utc"]
columnsToRename = {"coordinates.latitude" : "latitude", "coordinates.longitude" : "longitude"}
resCityDay = resCityDay.drop(columnsToDelete, axis=1)
resCityDay = resCityDay.rename(columns=columnsToRename)
resCityDay["date"] = resCityDay.index
#resCityDay["date"] = pd.to_datetime(resCityDay["date"])
resCityDay["date"] = resCityDay["date"].dt.strftime("%Y_%m_%d-%H:%M:%S")
resCityDay.reset_index(drop = True, inplace = True)

#print(resCityDay)
resCityDay.to_csv(filename + interested_day + "_per_hours_" + city + ".csv", sep=',')

dataO3Day   = resCityDay[resCityDay.parameter.str.contains('o3')]
dataO3Day.reset_index(drop = True, inplace = True)
dataO3Day.to_csv(filename + interested_day + "_per_hours_O3_" + city + ".csv", sep=',')

dataNO2Day  = resCityDay[resCityDay.parameter.str.contains('no2')]
dataNO2Day.reset_index(drop = True, inplace = True)
dataNO2Day.to_csv(filename + interested_day + "_per_hours_NO2_" + city + ".csv", sep=',')

dataPM10Day = resCityDay[resCityDay.parameter.str.contains('pm10')]
dataPM10Day.reset_index(drop = True, inplace = True)
dataPM10Day.to_csv(filename + interested_day + "_per_hours_PM10_" + city + ".csv", sep=',')


"""
basicData = pd.DataFrame(resCity[['location', 'parameter', 'value']])
refinedData = basicData[basicData.parameter.str.contains('|'.join(selection))]
#print(refinedData)

# This returns the city locations of all the sensors
cityLocations = api.locations(city=city, df=True)  
#cityLocations.to_csv(filename + "5_all_info_" + city + ".csv", sep=',')

cityLocationsLeftJoin = cityLocations[['location', 'coordinates.longitude', 'coordinates.latitude']]
#print(cityLocationsLeftJoin)

mergedDataSet = pd.merge(left=refinedData, right=cityLocationsLeftJoin, how='left', left_on='location', right_on='location')
#mergedDataSet.to_csv(filename + "0_" + city + "_DataSet.csv", sep=',')
#print(mergedDataSet)

dataO3 = mergedDataSet[mergedDataSet.parameter.str.contains('o3')]
dataO3.reset_index(drop = True, inplace = True)
#dataO3.to_csv(filename + "1_O3_" + city + ".csv", sep=',')
#print("\n============================== O3 DATA ==============================")
#print(dataO3)

dataNO2 = mergedDataSet[mergedDataSet.parameter.str.contains('no2')]
dataNO2.reset_index(drop = True, inplace = True)
#dataNO2.to_csv(filename + "2_NO2_" + city + ".csv", sep=',')
#print("\n============================== NO2 DATA ==============================")
#print(dataNO2)

dataPM10 = mergedDataSet[mergedDataSet.parameter.str.contains('pm10')]
dataPM10.reset_index(drop = True, inplace = True)
#dataPM10.to_csv(filename + "3_PM10_" + city + ".csv", sep=',')
#print("\n============================== PM25 DATA ==============================")
#print(dataPM10)

# Graph Creation O3
pointsO3 = [dataO3['coordinates.longitude'].tolist(), dataO3['coordinates.latitude'].tolist()]
#print("Coordinates O3")
#print(pointsO3)

sizeO3 = len(dataO3)
WO3 = np.zeros((sizeO3, sizeO3))

for i in range(0, len(pointsO3[0])):
    pointTmp = [pointsO3[0][i], pointsO3[1][i]]
    #print(*pointTmp, sep=", ")
    for j in range(i, len(pointsO3[0])):
        nextPointTmp = [[pointsO3[0][j], pointsO3[1][j]]]
        #print(*nextPointTmp, sep=", ")
        WO3[i][j] = geopy.distance.vincenty(pointTmp, nextPointTmp).km

# Symmetric Matrix
WO3 = WO3 + WO3.T
#print(WO3)
#np.savetxt("/home/zap2x/Desktop/WO3.txt", WO3, fmt='%.2f')
# Graph O3
GO3 = graphs.Graph(WO3)

# Graph Creation NO2
pointsNO2 = [dataNO2['coordinates.longitude'].tolist(), dataNO2['coordinates.latitude'].tolist()]
#print("\nCoordinates NO2")
#print(pointsNO2)

sizeNO2 = len(dataNO2)
WNO2 = np.zeros((sizeNO2, sizeNO2))

for i in range(0, len(pointsNO2[0])):
    pointTmp = [pointsNO2[0][i], pointsNO2[1][i]]
    #print(*pointTmp, sep=", ")
    for j in range(i, len(pointsNO2[0])):
        nextPointTmp = [[pointsNO2[0][j], pointsNO2[1][j]]]
        #print(*nextPointTmp, sep=", ")
        WNO2[i][j] = geopy.distance.vincenty(pointTmp, nextPointTmp).km

# Symmetric Matrix
WNO2 = WNO2 + WNO2.T
#print(WNO2)
# Graph NO2
GNO2 = graphs.Graph(WNO2)

# Graph Creation PM10
pointsPM10 = [dataPM10['coordinates.longitude'].tolist(), dataPM10['coordinates.latitude'].tolist()]
#print("\nCoordinates PM10")
#print(pointsPM10)

sizePM10 = len(dataPM10)
WPM10 = np.zeros((sizePM10, sizePM10))

for i in range(0, len(pointsPM10[0])):
    pointTmp = [pointsPM10[0][i], pointsPM10[1][i]]
    #print(*pointTmp, sep=", ")
    for j in range(i, len(pointsPM10[0])):
        nextPointTmp = [[pointsPM10[0][j], pointsPM10[1][j]]]
        #print(*nextPointTmp, sep=", ")
        WPM10[i][j] = geopy.distance.vincenty(pointTmp, nextPointTmp).km

# Symmetric Matrix
WPM10 = WPM10 + WPM10.T
#print(WPM10)
# Graph PM10
GPM10 = graphs.Graph(WPM10)
"""