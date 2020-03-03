import openaq
import warnings
import pandas as pd
#import seaborn as sns
#import matplotlib as mpl
from datetime import datetime
#import matplotlib.pyplot as plt

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
#print("Seaborn v{}".format(sns.__version__))
#print("Matplot v{}".format(mpl.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis API gathers information of the following parameters:\n")
param = api.parameters(df=True)
print(param)

# yy/mm/dd
date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
filename = "/home/zap2x/Desktop/GSP/Datasets/" + date


res = api.locations(city=city, df=True) # This returns the city of Madrid
print(res.iloc[0]) # replaces the old .ix[<index>] 

res = api.latest(city=city, parameter='o3', df=True)
res.to_csv(filename + "_o3.csv", sep=',')

res = api.latest(city=city, parameter='no2', df=True)
res.to_csv(filename + "_no2.csv", sep=',')

res = api.latest(city=city, parameter='pm25', df=True)
res.to_csv(filename + "_pm25.csv", sep=',')