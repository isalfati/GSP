import time # Lord
#import folium
#import seaborn as sns
import geopy.distance
from programConfig import *
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from pygsp import graphs, filters, plotting, utils

missingDataIndicator = False

# Import data
dataParam = pd.read_csv(filename + month + "_" + city + "_" + year + "_" + contaminant + ".csv", sep=",")
dataParam["DATA"] = pd.to_datetime(dataParam["DATA"], format=("%d/%m/%Y")).dt.strftime("%d-%m-%Y")
dataParam.sort_values(["CODI EOI", "DATA"], ascending=[True, True], inplace=True)
dataParam.reset_index(drop=True, inplace=True)

# Clean data, rename columns and obtain info from stations.
columnsToClean = ["CODI MESURAMENT", "CODI MUNICIPI", "PROVINCIA", "CODI ESTACIÓ", 
                  "ÀREA URBANA", "MAGNITUD", "CONTAMINANT", "UNITATS", "PUNT MOSTREIG", "ANY", "MES", "DIA", "Georeferència"]

for i in range(1, 25):
    columnsToClean.append("V0" + str(i) if i < 10 else "V" + str(i))

# Pure and clean data.
cleanDataParam = dataParam
cleanDataParam.drop(columns=columnsToClean, inplace=True)
cleanDataParam = cleanDataParam.where(pd.notnull(cleanDataParam), None)

# Info about the stations (TODO: Ensure that all fields are not NONE/Null/NaN)
infoStations = cleanDataParam.drop_duplicates(subset="CODI EOI")
infoStations = infoStations.iloc[:,0:7]
infoStations.reset_index(drop=True, inplace=True)
print(infoStations)

print("\nThese are the stations that measure {}:\n".format(contaminant))
print(*infoStations["NOM ESTACIÓ"], sep="\n")

# Fill simple gaps of information that can be extracted from infoStations
for colname in infoStations.columns.values:
    for i, val in cleanDataParam[colname].iteritems():
        if val is None:
            #print("Column Missing: {}, row index: {}, Actual Value: {}, Codi EOI: {}.".format(colname, i, val, cleanDataParam.iloc[i]["CODI EOI"]))
            eoi = cleanDataParam.iloc[i]["CODI EOI"]
            element = infoStations[infoStations["CODI EOI"] == eoi][colname]
            cleanDataParam.at[i, colname] = element.iat[0]

#print(cleanDataParam)

#TODO: Missing data and TIMESTAMPS
listMissingData = []
for elem in infoStations["NOM ESTACIÓ"]:
    listMissingData.append([elem, False, []])
print()
print(*listMissingData, sep="\n")

# Adjacency, Distance and Weighted Matrices
points = [infoStations["LATITUD"].tolist(), infoStations["LONGITUD"].tolist()]
size = len(infoStations)

adjacencyMatrix = np.zeros((size, size))        # 1 or 0
distancesMatrix = np.zeros((size, size))        # [i][j] = distance between [i] and [j]
weightMatrix    = np.zeros((size, size))        # [i][j] = e^(-distance(i,j)²) / (tau²)
maxDistanceBetweenStations = 70.0

sumDistances = 0

# Distances Matrix
for i in range(0, size):
    origin = [points[0][i], points[1][i]]
    #print("From {}.".format(origin))
    for j in range(i, size):
        destination = [points[0][j], points[1][j]]
        #print(" |-> To {}.".format(destination))
        distance = round(geopy.distance.VincentyDistance(origin, destination).km, decimals) # LAT1 / LONG1 @ LAT2 / LONG2
        sumDistances += distance
        distancesMatrix[i][j] = distance

totalDistance = round(sumDistances*2, decimals)
tau = round(totalDistance / (size*size), decimals)

print("\nThe constant Tau will be {}.".format(tau))
print("Threshold is {}.\n".format(maxDistanceBetweenStations)) #TODO: what to do with long distance stations?

# Weighted Matrix + Adjacencies
for i in range(0, size):
    for j in range(i, size):
        dst = distancesMatrix[i][j]
        aux1 = dst*dst
        aux2 = tau*tau

        if dst <= maxDistanceBetweenStations and i != j:
            weightMatrix[i][j] = round(math.exp(-aux1/aux2), decimals)
            adjacencyMatrix[i][j] = 1
        else:
            weightMatrix[i][j] = adjacencyMatrix[i][j] = 0

# Because we want undirected graphs, we add the tranposed matrix.
distancesMatrix = distancesMatrix + distancesMatrix.T
print("Distances Matrix:")
print("\n".join(["\t".join([str(cell) for cell in row]) for row in distancesMatrix]))

adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T
print("\nAdjacency Matrix:")
print("\n".join(["\t".join([str(cell) for cell in row]) for row in adjacencyMatrix]))

weightMatrix = weightMatrix + weightMatrix.T
print("\nWeighted Matrix:")
print("\n".join(["\t".join([str(cell) for cell in row]) for row in weightMatrix]))

#@@@@@@@@@@@@@@@@@@@@@@
#@@@ Graph Creation @@@
#@@@@@@@@@@@@@@@@@@@@@@

# For now, I"m using the weighted matrix, calculating the distance between latitudes and longitudes
Graph = graphs.Graph(weightMatrix)
Graph.compute_laplacian("normalized")

# Access Laplacian Matrix.
LaplacianSparseMatrix = Graph.L
LaplacianMatrix = LaplacianSparseMatrix.toarray()
print("\nLaplacian Matrix (Normalized):")
print("\n".join(["\t".join([str(round(cell, decimalsSparse)) for cell in row]) for row in LaplacianMatrix]))

# Compute a full eigendecomposition of the graph Laplacian such that: L = UAU*
#   Where A is the diagonal matrix of eigenvalues
#   Where the columns of U are the eigenvectors
Graph.compute_fourier_basis()

# Spectral decomposition
Eigenvalues = Graph.e
print("\nEigenvalues, a.k.a. Graph Adjacency Spectrum:\n {}.".format(Graph.e))

#plt.stem(Graph.e)
#plt.show()

Eigenvectors = Graph.U
print("\nEigenvectors:")
print("\n".join(["\t".join([str(round(cell, decimalsSparse)) for cell in row]) for row in Eigenvectors]))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Creating Matrix: Timestamp / S1, ..., SN @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

dates = cleanDataParam["DATA"].unique().tolist()
columnNames = list(cleanDataParam.columns.values)
originalHours = columnNames[cleanDataParam.columns.get_loc("H01"):]
hours = [elem.replace("H", "") for elem in originalHours]

timestamps = []
for date in dates:
    for h in hours:
        timestamps.append(date + "_" + h + ":00:00")

generalMatrix = []            
missingDataPerColumn = np.zeros((len(infoStations)))

for day in range(1, monthList[month]+1):
    iMonth = list(monthList.keys()).index(month) + 1
    d = "0" + str(day) if day < 10 else str(day)
    m = "0" + str(iMonth) if iMonth < 10 else str(iMonth)
    date = d + "-" + m + "-" + year

    for hour in originalHours:
        auxList = []
        for index, station in enumerate(infoStations["CODI EOI"]):
            subDF = cleanDataParam[cleanDataParam["DATA"].str.contains(date)]
            auxDF = subDF[subDF["CODI EOI"] == station][hour]
            value = None
            if not auxDF.empty:        
                value = subDF[subDF["CODI EOI"] == station][hour].iat[0]
            
            if value is None:
                missingDataPerColumn[index] += 1
            #print("Station: {}.".format(infoStations[infoStations["CODI EOI"] == station]["NOM ESTACIÓ"].iat[0]))
            auxList.append(value)
            # Since we can check None values it may be worth to add which fields are missing data.
        auxList.append(date + "_" + hour.replace("H", "") + ":00:00")
        #print(auxList)
        generalMatrix.append(auxList)

print("\nAmount of data missing per station:")
print(*missingDataPerColumn, sep=", ")

pollutionColumns = []
for name in infoStations["NOM ESTACIÓ"]:
    pollutionColumns.append(name)

pollutionColumns.append("timestamp")

pollutionDF = pd.DataFrame(generalMatrix, columns=pollutionColumns)

# Drop station that do not have at least 85% of the data, they are only a few.
columnsToDrop = []
columnsIndexToDrop = []
listStationNames = infoStations["NOM ESTACIÓ"].tolist()

for index in range(0, len(missingDataPerColumn)):
    if missingDataPerColumn[index] > (monthList[month]*24*minPercentatge):
        columnsToDrop.append(listStationNames[index])
        columnsIndexToDrop.append(index)

print("\nOriginal Stations:")
print(listStationNames, sep="\t")

print("\nThese are the stations that do not fullfill the {} constraint:".format(minPercentatge))
print(columnsToDrop, sep="\t")
print("\nIndex of those columns:")
print(columnsIndexToDrop, sep=", ")

pollutionDF.drop(columns=columnsToDrop, axis=1, inplace=True)
pollutionDF.dropna(axis=0, inplace=True)
print(pollutionDF)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End of Matrix Creation and Cleaning @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Split Matrix for analysis purpouses @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

matrixPollution = pollutionDF.to_numpy()
cleanPollutionColumns = [item for item in pollutionColumns if item not in columnsToDrop]
print("\nClean Columns:")
print(cleanPollutionColumns)
print()

sizeMatrixPollution = len(matrixPollution)

split60 = int(sizeMatrixPollution*0.6)
split40 = sizeMatrixPollution - split60

training60 = matrixPollution[0:split60]
training60DF = pd.DataFrame(training60, columns=pollutionDF.columns.values)

test40 = matrixPollution[split60:]
test40DF = pd.DataFrame(test40, columns=pollutionDF.columns.values)
#print(test40DF)
#print(training60DF)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End Matrix for analysis purpouses @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# METHOD A)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Linear Combination Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

tmpRecon = np.zeros(len(test40))
# Station To Be Reconstructed
reconStation = 0

adjacencyCols = adjacencyMatrix[reconStation]
interestedAdjacencyCols = [] 
laplacianCols = LaplacianMatrix[reconStation]
interestedLaplacianCols = []

for index in range(0, len(adjacencyCols)):
    if index not in columnsIndexToDrop:
        interestedAdjacencyCols.append(adjacencyCols[index])
        interestedLaplacianCols.append(laplacianCols[index])

print("Adjacency of interested columns: ")
print(interestedAdjacencyCols, sep=", ")

for i in range(0, len(test40)):
    for j in range(0, len(interestedAdjacencyCols)):
        adj = interestedAdjacencyCols[j]
        lpc = interestedLaplacianCols[j]
        tst = test40[i][j]
        #print("ADJ: {}, LPC: {}, TST: {}.".format(adj, lpc, tst))
        tmpRecon[i] += (adj*lpc*tst)
    #print("@@@@ Result: {}.".format(tmpRecon[i]))
    #print()
    tmpRecon[i] *= -1

originalValuesStation = test40[:, reconStation]
predictedValuesStation = tmpRecon

#for i in range(0, len(originalValuesStation)):
#        print("Original vs Reconstructed: {} % {}.".format(originalValuesStation[i], predictedValuesStation[i]))

MSE = mean_squared_error(test40[:, reconStation], tmpRecon)
RMSE = math.sqrt(MSE)

print("\nRMSE: {}\n.".format(RMSE))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End Linear Combination Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# METHOD B)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Machine Learning - Linear Regression @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

adjColNames = []

for index in range(0, len(cleanPollutionColumns)-1):
    if adjacencyCols[index] > 0:
        adjColNames.append(cleanPollutionColumns[index])
    
print(adjColNames)

########### Training
adjacencyTraining60 = training60DF[adjColNames]
input60X  = adjacencyTraining60.to_numpy()
output60Y = training60DF[cleanPollutionColumns[reconStation]].to_numpy()

model = LinearRegression().fit(input60X, output60Y)

r_sq = model.score(input60X, output60Y)
print("Coefficient of Determination: {}.".format(r_sq))
print("Intercept: {}.".format(model.intercept_))
print("Slope: {}.".format(model.coef_))

########### Test

adjacencyTest40 = test40DF[adjColNames]
input40X  = adjacencyTest40.to_numpy()
output40Y = test40DF[cleanPollutionColumns[reconStation]].to_numpy()

y_pred = model.predict(input40X)
#print("Predicted Response: ", y_pred, sep="\n")

MSE = mean_squared_error(output40Y, y_pred)
RMSE = math.sqrt(MSE)

print("\nRMSE: {}.\n".format(RMSE))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@#@@@@@@@@@@@@@@@@@@@
#@@@ End Machine Learning - Linear Regression @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@#@@@@@@@@@@@@@@@@@@@

# METHOD C)

#@@@@@@@@@@@@@@@@@@@@@
#@@@ GSP Stankovic @@@
#@@@@@@@@@@@@@@@@@@@@@

# Smoothness of Eigenvectors
eigenvec = []
eigenvecTransposed = []
smoothness = []

for index in range(0, len(Eigenvectors[0])):
    eigenvec = np.array([Eigenvectors[:, index]])
    eigenvecTransposed = eigenvec.T

print("Eigenvector: \n{}.".format(eigenvec))
print("\nEigenvector Transposed:\n{}.".format(eigenvecTransposed))
print("\nSmoothness of the eigenvector:\n")
print("\n\n".join(["\n".join(["\t".join([str(round(item, decimalsSparse)) for item in row]) for row in elem]) for elem in smoothness]))

N = len(infoStations["NOM ESTACIÓ"]) # N Nodes of our graph
vertexSignal = [[]]

# Is any data missing?
#if missingDataIndicator: # <- Aplicación #1
#print("\nThere is some data missing, reconstructing data.")

auxPollutionDF = pollutionDF
auxPollutionDF.drop(columns=listStationNames[0], inplace=True)

interestedLength = pollutionDF.shape[1]-1

N = len(infoStations["NOM ESTACIÓ"]) # N Nodes of our graph

#REPASAR 

for elemTime in auxPollutionDF["timestamp"].unique().tolist():
    #print("Timestamp: {}.".format(elemTime))
    values = auxPollutionDF[auxPollutionDF["timestamp"].str.contains(elemTime)].values.tolist()[0]
 
    if not len(values):
        print("we cannot recover the data from the following timeslot: {}.".format(elemTime))
        #add to unrecoverable
    else:
        # initializing recovery
        valuesSignal = values[0:interestedLength]
        valuesStations = auxPollutionDF.columns.values.tolist()[0:interestedLength]
        stations = infoStations["NOM ESTACIÓ"].tolist()

        vertexSignal = []
        for name in valuesStations:
            vertexSignal.append(stations.index(name))

        print("Vertex List: {}.".format(vertexSignal))
        print("Signal List: {}.\n".format(valuesSignal))        
    
        K = M = len(vertexSignal)

        for val in range(1, M+1):
            K = val
            print("N: {}, M: {}, K = {}.".format(N, len(vertexSignal), K))

            measurementMatrix = np.array(Eigenvectors[vertexSignal, 0:K]) # Extract all eigenvector rows of the indexes corresponding to station of the data not missing and the K first columns
            pInv = np.linalg.pinv(measurementMatrix)

            coeficientsX = np.matmul(pInv, valuesSignal).tolist()
            
            for i in range(0, N-K):
                coeficientsX.append(0)

            coeficientsX = np.array(coeficientsX)

            # Recover Signal
            signalRecovered = np.matmul(Eigenvectors, coeficientsX.T)
            signalRecovered = [round(x, decimals) for x in signalRecovered]

            print("Recovered L: {}.\n".format(signalRecovered))
    

    






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End of GSP Stankovic @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Variation 1)

# Variation 2)

# Variation 3)

# Future work Variation 4)