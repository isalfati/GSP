import time # Lord
import folium
import seaborn as sns
import geopy.distance
from copy import deepcopy
from programConfig import *
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from pygsp import graphs, filters, plotting, utils

# Program created by: Ivan Salfati 
# Mail: ivansalfatift@gmail.com

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
    print()

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
            #print(distance)
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

def coordinatePlots(infoStations, weightMatrix):
    print("\nGenerating Folium Map...")

    min_lat = infoStations["LATITUD"].min()
    max_lat = infoStations["LATITUD"].max()
    min_lon = infoStations["LONGITUD"].min()
    max_lon = infoStations["LONGITUD"].max()
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(location=[center_lat, center_lon], tiles='openstreetmap', zoom_start=10, control_scale=True)

    for i in range(0, len(infoStations)):
        folium.Marker([infoStations.iloc[i]['LATITUD'], infoStations.iloc[i]['LONGITUD']], popup=infoStations.iloc[i]["NOM ESTACIÓ"]).add_to(m)

    m.save(filename + "Plots/" + month + "_" + city + "_" + year + "_" + contaminant + "_" + str(maxDistanceBetweenStations) + "Km_map.html")

    print("\nGenerating Adjacency Plot...")

    fig, ax = plt.subplots(figsize=(10,10), dpi=700)
    
    # Style
    sns.set(style='whitegrid', color_codes=True)

    sns.scatterplot(x="LONGITUD", y="LATITUD", data=infoStations, ax=ax)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    #Plot Edges
    
    for i in range(0, len(weightMatrix[0])):
        pointTmp = [weightMatrix[0][i], weightMatrix[1][i]]
        for j in range(i, len(weightMatrix[0])):
            if weightMatrix[i][j] > 0.0:
                df2 = pd.DataFrame(np.array([[infoStations.loc[i]["LATITUD"], infoStations.loc[i]["LONGITUD"]], [infoStations.loc[j]["LATITUD"], infoStations.loc[j]["LONGITUD"]]]), columns=["LONGITUD", "LATITUD"])
                ax.plot(df2["LATITUD"], df2["LONGITUD"], 'black')
                

    ax.plot(infoStations["LONGITUD"], infoStations["LATITUD"], 'bo') # Black-Blue Circles
    plt.savefig((filename + "Plots/" + month + "_" + city + "_" + year + "_" + contaminant + "_" + str(maxDistanceBetweenStations) + "Km_adjacencies.png"))

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
    print("Finding columns to be dropped...")
    
    # Drop stations that do not have at least 85% of the data (a few have 0 data).
    columnsToDrop = []
    columnsIndexToDrop = []
    listStationNames = infoStations["NOM ESTACIÓ"].tolist()

    for index in range(0, len(missingDataPerColumn)):
        if missingDataPerColumn[index] > (monthList[month]*24*minPercentatge):
            columnsToDrop.append(listStationNames[index])
            columnsIndexToDrop.append(index)

    return columnsToDrop, columnsIndexToDrop

def splitDataSet(pollutionColumns, columnsToDrop, pollutionDF):
    print("Splitting data set...\n")

    matrixPollution = pollutionDF.to_numpy()
    cleanPollutionColumns = [item for item in pollutionColumns if item not in columnsToDrop]
    sizeMatrixPollution = len(matrixPollution)

    split60 = int(sizeMatrixPollution*0.6)
    split40 = sizeMatrixPollution - split60

    set60 = matrixPollution[0:split60]
    set60DF = pd.DataFrame(set60, columns=pollutionDF.columns.values)

    set40 = matrixPollution[split60:]
    set40DF = pd.DataFrame(set40, columns=pollutionDF.columns.values)

    return set40, set40DF, set60, set60DF, cleanPollutionColumns

def linearCombination(adjacencyCols, laplacianCols, sizeSet40, columnsIndexToDrop, set40):
        print("Linear Combination Reconstruction...")

        recon = np.zeros(sizeSet40)
        interestedAdjacencyCols = []
        interestedLaplacianCols = []

        for index in range(0, len(adjacencyCols)):
            if index not in columnsIndexToDrop:
                interestedAdjacencyCols.append(adjacencyCols[index])
                interestedLaplacianCols.append(laplacianCols[index])

        #print("Adjacency of interested columns: ")
        #print(interestedAdjacencyCols, sep=", ")

        #print("Laplacian of interested columns: ")
        #print(interestedLaplacianCols, sep=", ")

        #print("First element of the set:" )
        #print(set40[0])

        for i in range(0, sizeSet40):
            for j in range(0, len(interestedAdjacencyCols)):
                adj = interestedAdjacencyCols[j]
                lpc = -interestedLaplacianCols[j]
                tst = set40[i][j]
                #print("ADJ: {}, LPC: {}, TST: {}.".format(adj, lpc, tst))
                recon[i] += (adj*lpc*tst)
                #print(recon[i])
            #recon[i] *= -1
            #print("@@@@ Result: {}.\n".format(recon[i]))
    
        return recon
    
def linearRegressionML(sizeCleanPollutionColumns, adjacencyCols, cleanPollutionColumns, set60DF, set40DF, station):
    print("Linear Regression - ML...")

    adjColNames = []
    for index in range(0, sizeCleanPollutionColumns):
        if adjacencyCols[index] > 0:
            adjColNames.append(cleanPollutionColumns[index])

    adjacencyTraining60 = set60DF[adjColNames]
    input60X  = adjacencyTraining60.to_numpy()
    output60Y = set60DF[cleanPollutionColumns[station]].to_numpy()

    # Training
    model = LinearRegression().fit(input60X, output60Y)
    r_sq = model.score(input60X, output60Y)
    
    print("Summary of Training:\n")
    print("Coefficient of Determination: {}.".format(r_sq))
    print("Intercept: {}.".format(model.intercept_))
    print("Slope: {}.".format(model.coef_))

    # Prediction
    adjacencyTest40 = set40DF[adjColNames]
    input40X  = adjacencyTest40.to_numpy()

    y_pred = model.predict(input40X)
    #print("Predicted Response: ", y_pred, sep="\n")

    return y_pred

def stankovicMethod(infoStations, Eigenvectors, EigenvectorsOrig, dataSet):
    print("Stankovic Method...")

    N = len(infoStations["NOM ESTACIÓ"])
    recoveredDataSet = []
    currentPositive = []
    maxK = 0

    for elemTime in dataSet["timestamp"].unique().tolist():
        #print("Timestamp: {}.".format(elemTime))
        values = dataSet[dataSet["timestamp"].str.contains(elemTime)].values.tolist()[0]
        #print("Values: {}.".format(values))
        
        valuesSignal = values[0:len(values)-1]        
        #print("Values Signal: {}.".format(valuesSignal))

        valuesStations = dataSet.columns.values.tolist()[0:len(values)-1]        
        #print("Stations: {}.".format(valuesStations))
        
        listStations = infoStations["NOM ESTACIÓ"].tolist()
        
        vertexSignal = []
        for index, name in enumerate(valuesStations):
            if valuesSignal[index] is not None:
                vertexSignal.append(listStations.index(name))

        valuesSignalArray = np.array(valuesSignal)
        valuesSignalArray
        #valuesSignal = [item for item in valuesSignal if item is not None]
        M = len(vertexSignal)
        #print("M: {}.".format(M))
        #print("VertexList: {}.".format(vertexSignal))
        #print("SignalList: {}.\n".format(valuesSignalArray))

        #for val in range(0, M):
        signalRecovered = []
        K = VALUEK
        #print("N: {}, M: {}, K = {}.".format(N, len(vertexSignal), K))

        # Extract all eigenvector rows of the indexes corresponding to station of the data not missing and the K first columns
        measurementMatrix = np.array(Eigenvectors[:, 0:K])
        #print(measurementMatrix.shape)

        #print("MM: {}.".format(measurementMatrix))

        pInv = np.linalg.pinv(measurementMatrix)

        #print(pInv.shape)
        #print(valuesSignalArray)
        #print(valuesSignalArray.shape)
        coeficientsX = np.matmul(pInv, valuesSignalArray)

        concat = np.zeros(N-K)

        #for i in range(0, N-K):
        #    coeficientsX.append(0)
        
        

        #coeficientsX = np.array(coeficientsX)
        coeficientsX = np.concatenate((coeficientsX, concat))
        #print("coef X: {}.".format(coeficientsX))

        #print("coeficientsX: {}.".format(coeficientsX))
        #print(coeficientsX.shape)

        #Recover Signal.
        signalRecovered = np.matmul(EigenvectorsOrig, coeficientsX)
        #signalRecovered = [round(x, decimals) for x in signalRecovered]
        signalRecovered = [x for x in signalRecovered]
        #print("Recovered L: {}.\n".format(signalRecovered))

        #print(signalRecovered)
        
        """
        positive = all(item >= 0 for item in signalRecovered)

        if positive:
            maxK = K
            currentPositive = signalRecovered
            currentPositive.append(elemTime)
            print(currentPositive)

        if K == M:
            #print("MAXK: {}.".format(maxK))
            #print("MaxK: {} -> Appending: {}.".format(maxK, currentPositive))
            recoveredDataSet.append(currentPositive)
            currentPositive = []

        tmp = signalRecovered
        tmp.append(elemTime)

        if K == M-1:
            print(tmp)
            print()
        """
        
        signalRecovered.append(elemTime)
        recoveredDataSet.append(signalRecovered)

    return recoveredDataSet

def mean_bias_error(y_true, y_pred):
    ytrue = np.array(y_true)
    ypred = np.array(y_pred)

    ytrue = ytrue.reshape(len(ytrue), 1)
    ypred = ypred.reshape(len(ypred), 1)

    diff = (ypred - ytrue)
    mbe = diff.mean()

    return mbe


def main():
    print("Welcome. This program analyzes the data of different contaminants in the air of Catalonia.")
    print("Right now is configured to analyze the following contaminant: {}.\n".format(contaminant))

    # Import Data
    cleanDataParam = importData()

    #print(cleanDataParam)

    # Obtain Info about Stations
    infoStations = obtainStationInfo(cleanDataParam)

    # Find which timestamps have missing data
    #listMissingData = obtainMissingData(infoStations)

    # Calculate Distance Matrix, each element contains the distance between elem [i] and elem [j]
    points = [infoStations["LATITUD"].tolist(), infoStations["LONGITUD"].tolist()]

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

    # Uncomment for plots
    #coordinatePlots(infoStations, weightMatrix)
    
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

    # Find Columns that do not fulfill the % of data.
    columnsToDrop, columnsIndexToDrop = findColumnsToDrop(infoStations, missingDataPerColumn)

    print("\nThese are the stations that do not fullfill the {} constraint:".format(minPercentatge))
    print(columnsToDrop, sep="\t")
    print("\nIndex of those columns:")
    print(columnsIndexToDrop, sep=", ")
    print()

    pollutionStankovicDF = pollutionDF.copy()
    #print("VARIATION STANKOVIC")
    #print(pollutionStankovicDF)

    print("Size Data Set pre-Drop NA: {}.".format(len(pollutionDF)))
    pollutionDF.drop(columns=columnsToDrop, axis=1, inplace=True)
    pollutionDF.dropna(axis=0, inplace=True)
    pollutionDF.reset_index(drop=True, inplace=True)
    print("Size Data Set post-Drop NA: {}.\n".format(len(pollutionDF)))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #@@@ Variation #1: Adding Noise to a Random Station @@@
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    variationPollutionDF = pollutionDF.copy()

    mean = pollutionDF[pollutionColumns[targetStation[1]]].sum() / len(pollutionDF)
    print("MEAN: {}.".format(mean))

    # Creation of noise with the same dimensions as a column (station)
    #noiseValues = np.random.normal(mu, sigma, [len(variationPollutionDF), 1])
    noiseValues = np.random.normal(mu, mean*levelError, [len(variationPollutionDF), 1])

    originalVal = variationPollutionDF[pollutionColumns[targetStation[1]]].values

    newValues = np.hstack([noiseValues[i] + originalVal[i] for i in range(0, len(noiseValues))])
    #newValues = [x + mean*levelError for x in noiseValues]
    variationPollutionDF[pollutionColumns[targetStation[1]]] = newValues

    
    #print("Original vs Corrputed")
    #for j in range(0, len(originalVal)):
    #    print("{}, {}.".format(originalVal[j], newValues[j]))
    

    # Apply reconstruction over all created data sets.
    listDataSets = []
    listDataSets.append(pollutionDF)
    listDataSets.append(variationPollutionDF)

    print("Reconstructing data on station: {}. For the second round, the faulty station will be: {}.\n".format(infoStations.iloc[targetStation[0]]['NOM ESTACIÓ'], infoStations.iloc[targetStation[1]]['NOM ESTACIÓ']))

    for index, elemDF in enumerate(listDataSets):
        print("@@@@@@@@@@@@@@@@@@@@")
        print("@@@ Data Set: #{} @@@".format(index))
        print("@@@@@@@@@@@@@@@@@@@@\n")

        stationToReconstruct = targetStation[0]

        # Split
        set40, set40DF, set60, set60DF, cleanPollutionColumns = splitDataSet(pollutionColumns, columnsToDrop, elemDF)

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #@@@ Linear Combination Method @@@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        print("@@@@@@@@@@ LC @@@@@@@@@@")
        originalValues = set40[:, stationToReconstruct]
        predictedValues = linearCombination(adjacencyMatrix[stationToReconstruct], 
                                            LaplacianMatrix[stationToReconstruct], 
                                            len(set40), columnsIndexToDrop, set40)

        A = set40DF["timestamp"].tolist()
        B = set40[:, stationToReconstruct]
        #C = predictedValues

        #print("[Timestamp, Original, Predicted]")
        #for i in range(0, len(A)):
        #    print("{}, \t {}, \t {}.".format(A[i], B[i], C[i]))

        MSE = mean_squared_error(set40[:, stationToReconstruct], predictedValues)
        RMSE = math.sqrt(MSE)
        print("\nLinear combination RMSE: {}.".format(RMSE))

        MBE = mean_bias_error(set40[:, stationToReconstruct], predictedValues)
        print("                    MBE: {}.\n".format(MBE))

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #@@@ Machine Learning - Linear Regression @@@
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        print("\n@@@@@@@@@@ MLR @@@@@@@@@@")

        predictedValuesLinearRegression = linearRegressionML(len(cleanPollutionColumns)-1, adjacencyMatrix[stationToReconstruct], cleanPollutionColumns, set60DF, set40DF, stationToReconstruct)
        
        #C = predictedValuesLinearRegression
        #print("[Timestamp, Original, Predicted]")
        #for i in range(0, len(A)):
        #    print("{}, \t {}, \t {}.".format(A[i], B[i], C[i]))

        MSE = mean_squared_error(set40[:, stationToReconstruct], predictedValuesLinearRegression)
        RMSE = math.sqrt(MSE)
        print("\nLinear Regression RMSE: {}.".format(RMSE))
        
        MBE = mean_bias_error(set40[:, stationToReconstruct], predictedValuesLinearRegression)
        print("                   MBE: {}.\n".format(MBE))

        #@@@@@@@@@@@@@@@@@@@@@
        #@@@ GSP Stankovic @@@
        #@@@@@@@@@@@@@@@@@@@@@

        print("\n@@@@@@@@@@ Stankovic @@@@@@@@@@")

        # Number of nodes in our graph
        N = len(infoStations["NOM ESTACIÓ"]) 

        EigenCustom = Eigenvectors.copy()
        EigenCustom = np.delete(EigenCustom, (targetStation[0]), 0)
        
        #print("\nEigenCustom:")
        #print("\n".join(["\t".join([str(round(cell, decimalsSparse)) for cell in row]) for row in EigenCustom]))

        #predictedStankovic = stankovicMethod(infoStations, Eigenvectors, set40DF)

        customSet40DF = set40DF.copy()
        #st = infoStations.iloc[targetStation[0]
        customSet40DF.drop(columns=infoStations.iloc[targetStation[0]]['NOM ESTACIÓ'], inplace=True)
        #print(customSet40DF)


        predictedStankovic = stankovicMethod(infoStations, EigenCustom, Eigenvectors, customSet40DF)
        predictedStankovicDF = pd.DataFrame(predictedStankovic, columns=pollutionColumns)

        predictedStankovicStation = predictedStankovicDF[pollutionColumns[stationToReconstruct]].values.tolist()

        C = predictedStankovicStation
        
        #print("[Timestamp, Original, Predicted]")
        #for i in range(0, len(A)):
        #    print("{}, \t {}, \t {}.".format(A[i], B[i], C[i]))
        
        MSE = mean_squared_error(set40[:, stationToReconstruct], predictedStankovicStation)
        RMSE = math.sqrt(MSE)
        print("\nStankovic RMSE: {}.".format(RMSE))

        MBE = mean_bias_error(set40[:, stationToReconstruct], predictedStankovicStation)
        print("           MBE: {}.\n".format(MBE))


if __name__ == "__main__":
    main()

    