import time # Lord
#import folium
#import seaborn as sns
import geopy.distance
from programConfig import *
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from pygsp import graphs, filters, plotting, utils

def importData():
    print("Importing data...\n")
    
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

    # Clean data
    dataParam.drop(columns=columnsToClean, inplace=True)
    dataParam = dataParam.where(pd.notnull(dataParam), None)    
    
    return dataParam

def obtainStationInfo(dataSet):
    print("Obtaining info about stations...\n")

    info = dataSet.drop_duplicates(subset="CODI EOI")
    info = info.iloc[:, 0:7]
    info.reset_index(drop=True, inplace=True)
    
    print("Filling gaps of information...\n")
    for colname in info.columns.values:
        for i, val in dataSet[colname].iteritems():
            if val is None:
                eoi = dataSet.iloc[i]["CODI EOI"]
                element = info[info["CODI EOI"] == eoi][colname]
                dataSet.at[i, colname] = element.iat[0]

    print("Summary of the stations: ")
    print(info, "\n")

    return info

def obtainMissingData(infoStations):
    print("Finding missing data...\n")

    listMissing = []
    for elem in infoStations["NOM ESTACIÓ"]:
        listMissing.append([elem, False, []])
    
    print("Summary of missing data: \n")
    print(*listMissing, sep="\n")

    return listMissing

def obtainDistancesMatrix(points, size):
    print("Obtaining matrix of distances...\n")
    
    distancesMatrix = np.zeros((size, size))
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

    print("The constant Tau is: {}.\n".format(tau))
    
    return distancesMatrix, tau
    
def obtainWeightAdjacencyMatrix(distancesMatrix, maxDistance, size, tau):
    print("Obtaining matrix of weights and adjacencies...\n")

    weightMatrix    = np.zeros((size, size))
    adjacencyMatrix = np.zeros((size, size))

    for i in range(0, size):
        for j in range(i, size):
            dst = distancesMatrix[i][j]
            aux1 = dst*dst
            aux2 = tau*tau

            if dst <= maxDistance and i != j:
                weightMatrix[i][j] = round(math.exp(-aux1/aux2), decimals)
                adjacencyMatrix[i][j] = 1
            else:
                weightMatrix[i][j] = adjacencyMatrix[i][j] = 0

    return weightMatrix, adjacencyMatrix

def obtainPollutionDataMatrix(infoStations, cleanDataParam):
    print("\nObtaining pollution data matrix, please wait...\n")

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

    return generalMatrix, missingDataPerColumn, timestamps 

def findColumnsToDrop(infoStations, missingDataPerColumn):
    # Drop stations that do not have at least 85% of the data (a few have 0 data).
    columnsToDrop = []
    columnIndexToDrop = []
    listStationNames = infoStations["NOM ESTACIÓ"].tolist()

    for index in range(0, len(missingDataPerColumn)):
        if missingDataPerColumn[index] > (monthList[month]*24*minPercentatge):
            columnsToDrop.append(listStationNames[index])
            columnsIndexToDrop.append(index)

    return columnsToDrop, columnIndexToDrop

def main():
    print("Welcome. This program analyzes the data of different contaminants in the air of Catalonia.")
    print("Right now is configured to analyze the following contaminant: {}.\n".format(contaminant))

    # Import Data
    cleanDataParam = importData()

    # Obtain Info about Stations
    infoStations = obtainStationInfo(cleanDataParam)

    # Find which timestamps have missing data
    listMissingData = obtainMissingData(infoStations)

    # Calculate Distance Matrix, each element contains the distance between elem [i] and elem [j]
    points = [infoStations["LATITUD"].tolist(), infoStations["LONGITUD"].tolist()]
    maxDistanceBetweenStations = 80 # km 

    distancesMatrix, tau = obtainDistancesMatrix(points, len(infoStations))

    # Calculate Weighted & Adjacency Matrix
    weightMatrix, adjacencyMatrix = obtainWeightAdjacencyMatrix(distancesMatrix, maxDistanceBetweenStations, len(infoStations), tau)

    # Because we want undirected graphs, we add the tranposed matrix.
    print("Summary of the matrices:")
    distancesMatrix = distancesMatrix + distancesMatrix.T
    print("Distances Matrix:")
    print("\n".join(["\t".join([str(cell) for cell in row]) for row in distancesMatrix]))

    adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T
    print("\nAdjacency Matrix:")
    print("\n".join(["\t".join([str(cell) for cell in row]) for row in adjacencyMatrix]))

    weightMatrix = weightMatrix + weightMatrix.T
    print("\nWeight Matrix:")
    print("\n".join(["\t".join([str(cell) for cell in row]) for row in weightMatrix]))

    # Graph Creation
    Graph = graphs.Graph(weightMatrix)
    Graph.compute_laplacian("normalized")

    # Access Laplacian Matrix
    LaplacianSparseMatrix = Graph.L
    LaplacianMatrix = LaplacianSparseMatrix.toarray()
    print("\nLaplacian Matrix (Normalized):")
    print("\n".join(["\t".join([str(round(cell, decimalsSparse)) for cell in row]) for row in LaplacianMatrix]))

    # Compute a full eigendecomposition of the graph Laplacian such that L = UAU*,
    # |-> Where A is the diagonal matrix of eigenvalues
    # |-> Where the columns of U are the eigenvectors.
    Graph.compute_fourier_basis()

    # Spectral decomposition
    Eigenvalues = Graph.e
    print("\nEigenvalues or \"Graph Adjacency Spectrum\":")
    #print(" ".join(["".join(str(e)) for e in Eigenvalues]))
    print("\n", Eigenvalues)
    
    Eigenvectors = Graph.U
    print("\nEigenvectors:")
    print("\n".join(["\t".join([str(round(cell, decimalsSparse)) for cell in row]) for row in Eigenvectors]))

    #plt.stem(Eigenvalues)
    #plt.show()

    # Create Table  [S0, S1, ..., Sn, Timestamp]
    generalMatrix, missingDataPerColumn, timestamps = obtainPollutionDataMatrix(infoStations, cleanDataParam)

    print("Amount of data missing per station:\n")
    print(*missingDataPerColumn, sep=", ")
    print()

    # DataFrame
    pollutionColumns = infoStations["NOM ESTACIÓ"].tolist()
    pollutionColumns.append("timestamp")

    pollutionAux = pd.DataFrame(generalMatrix, columns=pollutionColumns)
    pollutionDF = pollutionAux.where(pd.notnull(pollutionAux), None)

    # 
    columnsToDrop, columnsIndexToDrop = findColumnsToDrop(infoStations, missingDataPerColumn)

    print("\nThese are the stations that do not fullfill the {} constraint:".format(minPercentatge))
    print(columnsToDrop, sep="\t")
    print("\nIndex of those columns:")
    print(columnsIndexToDrop, sep=", ")

if __name__ == "__main__":
    main()