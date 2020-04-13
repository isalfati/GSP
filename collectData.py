import openaq
from programConfig import * # 

# Check versions
print("Versions of the libraries:\n")
print("OpenAQ  v{}.".format(openaq.__version__))
print("Pandas  v{}.".format(pd.__version__))

api = openaq.OpenAQ()

# List of Parameters
print("\nThis program gathers information of some of the following parameters:\n")
param = api.parameters(df=True)
print(param[["name", "description"]])
print("\nIn particular, the following selection: {}.\n".format(selection))

# Obtaining measurements of the sensors in a City during a Certain Month
dataParam = pd.DataFrame()
monthIndex = list(monthList.keys()).index(interestedMonth) + 1
monthIndex = "0" + str(monthIndex) if monthIndex < 10 else str(monthIndex)
partialDate = interestedYear + "-" + monthIndex + "-"

for day in range(1, monthList[interestedMonth] + 1):
    print("({}/{}): Wait a moment, please...".format(day, monthList[interestedMonth]))
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
dataParam.sort_values(["location", "date"], ascending=[True, True], inplace=True)
dataParam.reset_index(drop=True, inplace=True)
dataParam.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours.csv", sep=",")

for elem in selection:
    dataAux = dataParam[dataParam.parameter.str.contains(elem)]
    dataAux.sort_values(["location", "date"], ascending=[True, True], inplace=True)
    dataAux.reset_index(drop=True, inplace=True)
    dataAux.to_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours_" + elem + ".csv", sep=",")

print("\nAll the data has been collected sucessfully.")