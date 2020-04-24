import time # Lord
import folium 
import seaborn as sns
import geopy.distance
import matplotlib as mpl
from programConfig import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pygsp import graphs, filters, plotting, utils

# Necessary
missingDataIndicator = False
# TBD, but we will try different thresholds 5km, 10km, 15km, 20km.
maxDistance = 20.14
# Cleaning columns
columnsClean = ["parameter", "value", "country", "city", "date"]


# Figures
figx, figy = (10, 10)

#################################################

# Import data from a specific parameter <paramAnalyzed>
dataParam = pd.read_csv(filename + city + "_" + interestedMonth + "_" + interestedYear + "_per_hours_" + paramAnalyzed + ".csv", sep=",", index_col=0)

# Clean data and obtain info about the stations
locationStation = dataParam.drop_duplicates(subset="location")
locationStation.drop(columns=columnsClean, inplace=True)
locationStation.reset_index(drop=True, inplace=True)

# GeoLocation
spotLocationStation = locationStation["spot"].unique().tolist()
# Identifier
idLocationStation   = locationStation["location"].unique().tolist()

print("List of stations that measures {}:\n".format(paramAnalyzed))
print(*spotLocationStation, sep="\n")
print()

#List of all possible timestamps of the month
listTimestamps = []
for day in range(1, monthList[interestedMonth]+1):
    indexMonth = list(monthList.keys()).index(interestedMonth) + 1
    strIndexMonth = "0" + str(indexMonth) if indexMonth < 10 else str(indexMonth)
    partialYM = interestedYear + "_" + strIndexMonth + "_"
    partialYMD = partialYM + "0" + str(day) + "-" if day < 10 else partialYM + str(day) + "-"
    for hour in range(0, 24):
        strTime = partialYMD + "0" + str(hour) + ":00:00" if hour < 10 else partialYMD + str(hour) + ":00:00"
        listTimestamps.append(strTime)
        
# Check which stations have missing data
listMissingData = []
for elem in spotLocationStation:
    listMissingData.append([elem, False, []])

for index, val in enumerate(locationStation["location"]):
    iterList = dataParam[dataParam["location"].str.contains(val)]
    iterTimeList = iterList.date.tolist()
    missingMeasurements = [elem for elem in listTimestamps if elem not in iterTimeList]
    if missingMeasurements:
        #print("The station with identifier {} located at {} misses the foloowing time measurements {}.".format(val, spotLocationStation[index], missingMeasurements))
        listMissingData[index][1] = True
        listMissingData[index][2] = missingMeasurements

        missingDataIndicator = True

#print(*listMissingData, sep="\n")
#print()

# Building weight Matrix aka Adjacency Matrix with Weights
# Each element of this matrix measures the distance between station i and station j.
# If they are separated more than a certain distance, there will be no connection between i and j.
points = [locationStation["latitude"].tolist(), locationStation["longitude"].tolist()]
size   = len(locationStation)

adjacencyMatrix = np.zeros((size, size))    # 1 or 0
distancesMatrix = np.zeros((size, size))    # [i][j] = distance(i, j)
weightMatrix = np.zeros((size, size))       # [i][j] = e^(-distance(i,j)^2) / (Tau^2)

sumdist = 0

# Distances Matrix
for i in range(0, size):
    pointTmp = [points[0][i], points[1][i]]
    #print("From {}.".format(pointTmp))
    for j in range(i, size):
        nextPointTmp = [[points[0][j], points[1][j]]]
        #print(" |-> To {}.".format(nextPointTmp))
        distance = round(geopy.distance.VincentyDistance(pointTmp, nextPointTmp).km, decimals) # LAT1, LON1, LAT2, LON2
        sumdist += distance
        distancesMatrix[i][j] = distance

totalDist = round(sumdist * 2, decimals)
tau = round(totalDist / (size*size), decimals) # Mean of all distances

print("The constant Tau will be {}.".format(tau))
print("Threshold is: {}.".format(maxDistance))  # <----------------- MAX DISTANCE

# Weighted Matrix + Adjacencies
for i in range(0, size):
    for j in range(i, size):
        dst = distancesMatrix[i][j]
        aux1 = dst*dst
        aux2 = tau*tau

        if dst <= maxDistance and i != j:       # <----------------- MAX DISTANCE
            weightMatrix[i][j] = round(math.exp(-aux1/aux2), decimals)
            adjacencyMatrix[i][j] = 1
        else:
            weightMatrix[i][j] = 0
            adjacencyMatrix[i][j] = 0

# Because we want undirected graphs, we add the transposed matrix
distancesMatrix = distancesMatrix + distancesMatrix.T
print("\nDistances Matrix:")
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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Signal Reconstruction using Stankovic Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
# Smoothness of Eigenvectors
eigenvec   = []
eigenvecTransposed = []
smoothness = []
for index in range(0, len(Eigenvectors[0])):
    eigenvec = np.array([Eigenvectors[:, index]])
    eigenvecTransposed = eigenvec.T

#print("Eigenvector:\n{}.".format(eigenvec))
#print("Eigenvector Transposed:\n{}.".format(eigenvecTransposed))
#print("Smoothness of the eigenvector:\n")
#print("\n\n".join(["\n".join(["\t".join([str(round(item, decimalsSparse)) for item in row]) for row in elem]) for elem in smoothness]))

N = len(locationStation["location"]) # N Nodes of our graph
vertexSignal = [[]]

# Is any data missing?
if missingDataIndicator:
    #TODO: Encapsulate in a function.
    print("\nThere is some data missing, reconstructing data.")

    for index, elem in enumerate(listMissingData):
        spot       = elem[0]
        missing    = elem[1]
        listTimes  = elem[2]

        unrecoverableData = []
        print("\nIndex {} is the station located at {}.".format(index, spot))
        print("({}/{}): Reconstructing data, please wait.\n".format(index+1, len(listMissingData)+1))

        for timeslot in listTimes:
            valuesDF = dataParam[dataParam.date.str.contains(timeslot)]
            if not len(valuesDF.index):
                #print("We cannot recover the data from the following timeslot: {}.".format(timeslot))
                unrecoverableData.append(timeslot)
            else:
                print("->@@@@@ Initializing signal recovery for timeslot {}. @@@@@<-\n".format(timeslot))
                valuesSignal = valuesDF["value"].tolist()       # Values <----------------|
                valuesStations = valuesDF["location"].tolist()  #                         |
                stations = locationStation["location"].tolist() # Vertex who has values ->|

                vertexSignal = [] #
                for i in valuesStations:
                    vertexSignal.append(stations.index(i))

                M = len(vertexSignal)
                # Definition of K
                K = 6
                
                vertexSignal = [] #
                for i in valuesStations:
                    vertexSignal.append(stations.index(i))

                M = len(vertexSignal)
                # Definition of K
                
                print("Vertex List: {}.".format(vertexSignal))
                print("Signal List: {}.\n".format(valuesSignal))

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

        listTimes = elem[2] = [item for item in elem[2] if item in unrecoverableData]

        if not elem[2]:
            elem[1] = False
        
        print("Using this method, it is impossible to recover ({}/{}) of the data for the station located at {}:\n".format(len(listTimes), 24*monthList[interestedMonth], spot))
        print(listTimes)
        print()

        #TODO: missingDataIndicator to False if all the data was recovered.
    
else:
    print("\nNo missing data in the dataset.")
"""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End of Stankovic Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Cleaning & Splitting Matrices @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

uniqueTimestamps = dataParam["date"].unique().tolist()
uniqueTimestamps.sort()
#print(uniqueTimestamps)

generalMatrix = []
print("\nGenerating Matrix, wait please..\n")

for timestamp in uniqueTimestamps:
    auxList = []

    for index, value in enumerate(idLocationStation):
        auxSubSet = dataParam[dataParam.location.str.contains(value)]
        auxElement = auxSubSet[auxSubSet.date.str.contains(timestamp)]
        result = auxElement.value.to_list()
        if not result:
            auxList.append(-1)
        else:
            auxList.append(result[0])

    #print(timestamp)
    #print(auxList)

    #if not -1 in auxList:
    generalMatrix.append(auxList)
    auxList.append(timestamp)


#print("\n".join(["\t".join([str(cell) for cell in row]) for row in generalMatrix]))

print("Matrix Generated.")

pollutionColumns = []
for i in range(0, len(idLocationStation)):
    #pollutionColumns.append(spotLocationStation[i])
    pollutionColumns.append(paramAnalyzed + "_" + str(i))

pollutionColumns.append("timestamp")

pollutionDF = pd.DataFrame(generalMatrix, columns=pollutionColumns)
matrixPollution = pollutionDF.to_numpy()
# Split Matrix

sizeMatrixPollution = len(matrixPollution)

split60 = int(sizeMatrixPollution*0.6)
split40 = sizeMatrixPollution-split60

training60 = matrixPollution[0:split60]
training60DF = pd.DataFrame(training60, columns=pollutionColumns)
#print(training60DF)

test40     = matrixPollution[split60:]
print(test40)
test40DF = pd.DataFrame(training60, columns=pollutionColumns)
#print(test40DF)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End Cleaning & Splitting Matrices @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Linear Combination Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

tmpRecon = np.zeros(len(test40))
reconStation = 0

for i in range(0, len(test40)):
    for j in range(0, len(idLocationStation)):
        #print("tmpRecon: {}, adjacency: {}, laplacian: {}, test: {}.".format(tmpRecon[i], adjacencyMatrix[reconStation][j], LaplacianMatrix[reconStation][j], test40[i][j]))
        adj = adjacencyMatrix[reconStation][j]
        lpc = LaplacianMatrix[reconStation][j]
        tst = test40[i][j]

        if tst < 0:
            tst = 0
        
        tmpRecon[i] += adj*lpc*tst
    tmpRecon[i] *= -1
        
#TODO:  RMSE

originals = test40[:, 0]
predicted = tmpRecon

print("size originals: {}, size predicted: {}.".format(len(originals), len(predicted)))

for i in range(0, len(originals)):
    print("{} : {}.".format(originals[i], predicted[i]))


MSE = mean_squared_error(test40[:, 0], tmpRecon)
RMSE = math.sqrt(MSE)

print("RMSE: {}.".format(RMSE))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End Linear Combination Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

"""            
for item in list:
    if conditional:
        expression
            
Equivalent to:

[ expresion for item in list if conditional ]
"""