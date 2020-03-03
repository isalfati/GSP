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

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.width = None

# Check versions
print("Versions of the libraries:\n")
print("OpenAQ v{}".format(openaq.__version__))
print("Pandas v{}".format(pd.__version__))
# print("Seaborn v{}".format(sns.__version__))
# print("Matplot v{}".format(mpl.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis API gathers information of the following parameters:\n")
param = api.parameters(df=True)
print(param)

# yy/mm/dd
date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
filename = "/home/zap2x/Desktop/GSP/Datasets/" + date

resCity = api.locations(city=city, df=True)  # This returns the city of Madrid
# print(resCity.iloc[0])  # replaces the old .ix[<index>]
print(resCity)

resO3 = api.latest(city=city, parameter='o3', df=True)
# resO3.to_csv(filename + "_o3.csv", sep=',')

resNO2 = api.latest(city=city, parameter='no2', df=True)
# resNO2.to_csv(filename + "_no2.csv", sep=',')

resPM25 = api.latest(city=city, parameter='pm25', df=True)
# resPM25.to_csv(filename + "_pm25.csv", sep=',')

# TODO: Check if there is any way to obtain the id of the stations that have all
# three records (o2, no2, pm25), if not, I should search all three data sets and find
# those stations who have all three records.