import openaq
import warnings
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib as mpl
from datetime import datetime
# import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

# Global Variables
city = "Madrid"
selection = ['o3', 'no2', 'pm25']

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

# Check versions
print("Versions of the libraries:\n")
print("OpenAQ v{}".format(openaq.__version__))
print("Pandas v{}".format(pd.__version__))
# print("Seaborn v{}".format(sns.__version__))
# print("Matplot v{}".format(mpl.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis program gathers information of the following parameters:\n")
#param = api.parameters(df=True)
print(selection)
print()

# yy/mm/dd
date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
filename = "/home/zap2x/Desktop/GSP/Datasets/" + date

# Obtaining the latest values of the sensors in a City
resCity = api.latest(city=city, parameters=['o3', 'no2', 'pm25'], df=True)
#resCityParams.to_csv(filename + "_all_params_" + city + ".csv", sep=',')

basicData = pd.DataFrame(resCity[['location', 'parameter', 'value']])
refinedData = basicData[basicData.parameter.str.contains('|'.join(selection))]
#print(refinedData)

# This returns the city locations of all the sensors
cityLocations = api.locations(city=city, df=True)  
#cityLocations.to_csv(filename + "_all_in_" + city + "_city.csv", sep=',')

cityLocationsLeftJoin = cityLocations[['location', 'coordinates.longitude', 'coordinates.latitude']]
#print(cityLocationsLeftJoin)

mergedDataSet = pd.merge(left=refinedData, right=cityLocationsLeftJoin, how='left', left_on='location', right_on='location')
#mergedDataSet.to_csv(filename + "_" + city + "_DataSet.csv", sep=',')
print(mergedDataSet)

dataO3 = mergedDataSet[mergedDataSet.parameter.str.contains('o3')]
dataO3.reset_index(drop = True, inplace = True)
#dataO3.to_csv(filename + "_O3_" + city + ".csv", sep=',')
print("\n============================== O3 DATA ==============================")
print(dataO3)

dataNO2 = mergedDataSet[mergedDataSet.parameter.str.contains('no2')]
dataNO2.reset_index(drop = True, inplace = True)
#dataNO2.to_csv(filename + "_NO2_" + city + ".csv", sep=',')
print("\n============================== NO2 DATA ==============================")
print(dataNO2)

dataPM25 = mergedDataSet[mergedDataSet.parameter.str.contains('pm25')]
dataPM25.reset_index(drop = True, inplace = True)
#dataPM25.to_csv(filename + "_PM25_" + city + ".csv", sep=',')
print("\n============================== PM25 DATA ==============================")
print(dataPM25)
