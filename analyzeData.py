import os
import sys
import math
import time # Lord
import gmplot
import openaq
import folium 
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import geopy.distance
import matplotlib as mpl
import mplleaflet as mpll
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pygsp import graphs, filters, plotting, utils

#@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ GLOBAL VARIABLES @@@
#@@@@@@@@@@@@@@@@@@@@@@@@

# Parameter to be analyzed
analyze = 'o3'

# City, Day, Output location, maximum distance between stations
city = "Barcelona"
interested_day         = "2020-03-08" # <- |
interestedDayTimestamp = "2020_03_08" # <- | Careful, they serve different purposes

filename = os.environ["HOME"] + "/Desktop/GSP/Datasets/"

# TBD, but we will try different thresholds 5km, 10km, 15km, 20km.
maxDistance = 20.14

meanRatio = 8 # in hours (min 1h, max 24h), should always be even and > 1

missingDataIndicator = False

# Figures
figx, figy = (10, 10)

# Cleaning columns
columnsClean = ["parameter", "value", "country", "city", "date"]

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Configuration Parameters @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Avoid warnings
warnings.simplefilter(action='ignore')

# Set Display Options
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None
pd.options.display.precision = 10
decimals = 4
decimalsSparse = 3

# Suppress=True allows to not use scientific notation when reading from CSV.
np.set_printoptions(threshold=np.inf, suppress=True)

#@@@@@@@@@@@@@@@@
#@@@ Analysis @@@
#@@@@@@@@@@@@@@@@

# Import data
dataParam = pd.read_csv(filename + interested_day + "_per_hours_" + analyze + "_" + city + ".csv", sep=',', index_col=0)

# locationStation contains a Dataframe with [location(identifier), city, latitude, longitude]
locationStation = dataParam.drop_duplicates(subset="location")
locationStation.drop(columns=columnsClean, inplace=True)
locationStation.reset_index(drop=True, inplace=True)
print("List of stations that measures {}: {}".format(analyze, locationStation['location'].unique()))

# List of timestamps of a day.
listTimeStamps = []
for i in range(0, 24):
    listTimeStamps.append(interestedDayTimestamp + "-0" + str(i) + ":00:00") if i < 10 else listTimeStamps.append(interestedDayTimestamp + '-' + str(i) + ":00:00")
    
# Check if any station has missing measurement time, this is useful for reconstructing signals of missing data.
listMissingData = []
for elem in locationStation['location']:
    listMissingData.append([elem, False, []])
    
#####
# TODO: Inducing first row missing.
#####

for index, val in enumerate(locationStation['location']):
    iterList = dataParam[dataParam.location.str.contains(val)]
    iterTimeList = iterList.date.tolist()
    missingMeasurements = [elem for elem in listTimeStamps if elem not in iterTimeList]
    if missingMeasurements:
        print("The station with identifier {} misses the following time measurements {}.".format(val, missingMeasurements))
        listMissingData[index][1] = True
        listMissingData[index][2] = missingMeasurements
        missingDataIndicator = True

"""
# Now we want to agroupate data by a certain hours, every 2, 4, 8, 12, 24h, but some stations have missing data.
"""            

# Building Weight Matrix a.k.a. adjacency matrix with weights
# Each element of this matrix measures the distance between station i and station j.
# If they are separated more than a certain distance, there will be no connection between i and j.
points = [locationStation['latitude'].tolist(), locationStation['longitude'].tolist()]
size   = len(locationStation)

adjacencyMatrix = np.zeros((size, size)) # 1 or 0
distancesMatrix = np.zeros((size, size)) # [i][j] = distance(i, j)
weightMatrix = np.zeros((size, size)) # [i][j] = e^(-distance(i,j)^2) / (Tau^2)

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
print("Threshold is: {}.".format(maxDistance))

# Weighted Matrix + Adjacencies
for i in range(0, size):
    for j in range(i, size):
        dst = distancesMatrix[i][j]
        aux1 = dst*dst
        aux2 = tau*tau

        if dst <= maxDistance and i != j:
            weightMatrix[i][j] = round(math.exp(-aux1/aux2), decimals)
            adjacencyMatrix[i][j] = 1
        else:
            weightMatrix[i][j] = 0
            adjacencyMatrix[i][j] = 0

# Because we want undirected graphs, we add the transposed matrix
distancesMatrix = distancesMatrix + distancesMatrix.T
print("\nDistances Matrix:")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in distancesMatrix]))

adjacencyMatrix = adjacencyMatrix + adjacencyMatrix.T
print("\nAdjacency Matrix:")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in adjacencyMatrix]))

weightMatrix = weightMatrix + weightMatrix.T
print("\nWeighted Matrix:")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in weightMatrix]))

#@@@@@@@@@@@@@@@@@@@@@@
#@@@ Graph Creation @@@
#@@@@@@@@@@@@@@@@@@@@@@

# For now, I'm using the weighted matrix, calculating the distance between latitudes and longitudes
Graph = graphs.Graph(weightMatrix)
Graph.compute_laplacian('normalized')

# Access Laplacian Matrix.
LaplacianSparseMatrix = Graph.L
LaplacianMatrix = LaplacianSparseMatrix.toarray()
print("\nLaplacian Matrix (Normalized):")
print('\n'.join(['\t'.join([str(round(cell, decimalsSparse)) for cell in row]) for row in LaplacianMatrix]))

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
print('\n'.join(['\t'.join([str(round(cell, decimalsSparse)) for cell in row]) for row in Eigenvectors]))


"""
for i in range(0, size):
    plt.subplot(size, 1, i+1)
    plt.yticks([-1, -0.5,  0, 0.5, +1])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.stem(Graph.U[:, i], basefmt='black', markerfmt='or', linefmt='red')
plt.subplots_adjust(hspace=0.5)
plt.show()
"""

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Saving Coordinates Plots @@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Set Figure
fig, ax = plt.subplots(1, 2, figsize=(figx*2, figy))

# Style
sns.set(style='whitegrid', color_codes=True)

# Define Plot Type
sns.scatterplot(x="longitude", y="latitude", data=locationStation, ax=ax[0])
# Save Figure
extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(filename + interested_day + "_location_stations_" + analyze + "_" + city+ ".png", bbox_inches=extent.expanded(1.2, 1.2))

# Plot Edges + Map
line_plot_ax = ax[1]
line_plot_ax.set_xlabel("longitude")
line_plot_ax.set_ylabel("latitude")

# Plot Edges.
for i in range(0, len(weightMatrix[0])):
    pointTmp = [weightMatrix[0][i], weightMatrix[1][i]]
    for j in range(i, len(weightMatrix[0])):
        if weightMatrix[i][j] > 0.0:
            #print("Distance between {} and {}: {}.".format(locationStation.loc[i].location, locationStation.loc[j].location, weightMatrix[i][j]))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[i].latitude, locationStation.loc[i].longitude))
            #print("                   |-> Long {}, Lat {}.".format(locationStation.loc[j].latitude, locationStation.loc[j].longitude))
            df2 = pd.DataFrame(np.array([[locationStation.loc[i].latitude, locationStation.loc[i].longitude], [locationStation.loc[j].latitude, locationStation.loc[j].longitude]]), columns=["longitude", "latitude"])
            line_plot_ax.plot(df2.latitude, df2.longitude, 'black')
            # Add distance in the middle of the edge

line_plot_ax.plot(locationStation.longitude, locationStation.latitude, 'bo') # Black-Blue Circles

# Plot index of location, who's index is assigned by sort(locations)
for i, point in locationStation.iterrows():
    #line_plot_ax.text(point['longitude']+0.01, point['latitude'], str(locationStation.location[i] + ", " + str(i))) 
    line_plot_ax.text(point['longitude']+0.005, point['latitude'], str("ID: ") + str(i) + "-> " + str(locationStation.spot[i]), fontsize=8)

extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(filename + interested_day + "_location_stations_graph_" + analyze + "_" + city + ".png", bbox_inches=extent.expanded(1.2, 1.2))
#plt.show()

#@@@@@@@@@@@@@@@@@@@@@@@@
#@@@ Folium Map (Geo) @@@
#@@@@@@@@@@@@@@@@@@@@@@@@

min_lat = locationStation["latitude"].min()
max_lat = locationStation["latitude"].max()
min_lon = locationStation["longitude"].min()
max_lon = locationStation["longitude"].max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

m = folium.Map(location=[center_lat, center_lon], tiles='openstreetmap', zoom_start=10, control_scale=True)

for i in range(0, len(locationStation)):
    folium.Marker([locationStation.iloc[i]['latitude'], locationStation.iloc[i]['longitude']], popup=locationStation.iloc[i]['spot']).add_to(m)

m.save(filename + interested_day + "_location_stations_" + analyze + "_" + city + "_map.html")
#plt.show()

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
#print('\n\n'.join(['\n'.join(['\t'.join([str(round(item, decimalsSparse)) for item in row]) for row in elem]) for elem in smoothness]))
# TODO: To do.





N = len(locationStation['location']) # N Nodes of our graph
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
            valuesSignal = valuesDF['value'].tolist()       # Values <----------------|
            valuesStations = valuesDF['location'].tolist()  #                         |
            stations = locationStation['location'].tolist() # Vertex who has values ->|

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

           

            
else:
    print("\nNo missing data in the dataset.")


"""            
for item in list:
    if conditional:
        expression
            
Equivalent to:

[ expresion for item in list if conditional ]
"""