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
                  "ÀREA URBANA", "MAGNITUD", "PUNT MOSTREIG", "ANY", "MES", "DIA", "Georeferència"]

for i in range(1, 25):
    columnsToClean.append("V0" + str(i) if i < 10 else "V" + str(i))

# Pure and clean data.
cleanDataParam = dataParam
cleanDataParam.drop(columns=columnsToClean, inplace=True)

# Info about the stations
infoStations = cleanDataParam.drop_duplicates(subset="CODI EOI")
infoStations = infoStations.iloc[:,0:6]
infoStations.reset_index(drop=True, inplace=True)
print(infoStations)

print("These are the stations that measure {}:\n".format(contaminant))
print(*infoStations["NOM ESTACIÓ"], sep="\n")

#TODO: Missing data and TIMESTAMPS

# Adjacency, Distance and Weighted Matrices
points = [infoStations["LATITUD"].tolist(), infoStations["LONGITUD"].tolist()]
size = len(infoStations)

adjacencyMatrix = np.zeros((size, size))        # 1 or 0
distancesMatrix = np.zeros((size, size))        # [i][j] = distance between [i] and [j]
weightMatrix    = np.zeros((size, size))        # [i][j] = e^(-distance(i,j)²) / (tau²)
maxDistanceBetweenStations = 20.0

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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ End of Stankovic Method @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@






