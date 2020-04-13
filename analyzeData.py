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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Signal Reconstruction @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

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
    print("\nThere is some data missing, reconstructing data.")
       
    for index, elem in enumerate(listMissingData):
        identifier = elem[0]
        missing    = elem[1]
        listTimes  = elem[2]

        print("Index {} is element {}.".format(index, elem))

        for timeslot in listTimes:
            print("\n############### New Timeslot ###############")
            print("Checking time slot: {}.".format(timeslot))            
            valuesDF = dataParam[dataParam.date.str.contains(timeslot)]
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
            
            for val in range(1, M+1):
                K = val
                print("\n========== New It ==========")
                print("N: {}, M: {}, K = {}.".format(N, len(vertexSignal), K))
                print("Vertex List: {}.".format(vertexSignal))
                print("Signal List: {}.".format(valuesSignal))

                # Start Time
                start = time.process_time()
                
                measurementMatrix = np.array(Eigenvectors[vertexSignal, 0:K]) # Extract all eigenvector rows of the indexes corresponding to station of the data not missing and the K first columns
                pInv = np.linalg.pinv(measurementMatrix)

                coeficientsX = np.matmul(pInv, valuesSignal).tolist()
                
                for i in range(0, N-K):
                    coeficientsX.append(0)

                coeficientsX = np.array(coeficientsX)
                            
                # Recover Signal
                signalRecovered = np.matmul(Eigenvectors, coeficientsX.T)
                signalRecovered = [round(x, decimals) for x in signalRecovered]

                print("Recovered L: {}.".format(signalRecovered))

                
                # End Time
                elapsedTime = (time.process_time() - start)*100
                print("\nElapsed time(ms): {}.".format(elapsedTime))

                #probably  list missing data [1] should be false after reconstruction 
else:
    print("\nNo missing data in the dataset.")

"""            
for item in list:
    if conditional:
        expression
            
Equivalent to:

[ expresion for item in list if conditional ]
"""