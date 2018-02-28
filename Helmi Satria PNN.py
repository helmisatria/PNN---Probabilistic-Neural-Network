#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:17:07 2018

@author: helmisatria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.cross_validation import train_test_split

# ax = plt.axes(projection='3d')

dataSet = np.genfromtxt('data_train_PNN.txt', skip_header=1)
dataTest= np.genfromtxt('data_test_PNN.txt', skip_header=1)

dataX = dataSet[:,0]
dataY = dataSet[:,1]
dataZ = dataSet[:,2]
dataClass = dataSet[:,3]

uniqueClass = list(set(dataClass))

# =============================================================================
# ax = plt.axes(projection='3d')
# ax.scatter(dataX, dataY, dataZ, c=dataClass, cmap='viridis', linewidth=0.5)
# =============================================================================

# =============================================================================
# Euclidean Distance
# =============================================================================

def euclidean(data1, data2):
    decX = (data1[0]-data2[0])**2
    decY = (data1[1]-data2[1])**2
    decZ = (data1[2]-data2[2])**2
    return np.sqrt(decX + decY + decZ)

def separateCol(data, dataSet, col):
    separatedClass = []
    for i, Class in enumerate(data):
        classes = []
        for y, rowData in enumerate(dataSet):
            if (rowData[col] == Class):
                classes.append(rowData)
        separatedClass.append(classes)
    return separatedClass

def neighborDistance(separatedClass):
    dataDistances = []
    for i, aClass in enumerate(separatedClass):
        dataClassDistance = []
        for y, row in enumerate(aClass):
            distances = []
            for z, insideRow in enumerate(aClass):
                if (y != z):
                    distances.append(euclidean(row, insideRow))
            dataClassDistance.append(min(distances))
        dataDistances.append(dataClassDistance)
    return dataDistances

def sumCol(data, col):
    dataSumDistances = []
    for i, val in enumerate(data):
        # sum all item in an array in column = 1
        dataSumDistances.append(sum(row[col] for row in val))
    return dataSumDistances

def cariF(g, dataSumDistances, separatedDataTrain):
    dataF = []
    for i, val in enumerate(dataSumDistances):
        #print(g)
        #print(dataSumDistances[i])
        #print(len(separatedDataTrain[i]))
        dataF.append(float(g * dataSumDistances[i])/len(separatedDataTrain[i]))
    return(dataF)
    
def cariG(test, dataTrain, dataF):
    dataG = []
    for y, rowTest in enumerate(test):
        for i, rowTrain in enumerate(dataTrain):
            typeClass = rowTrain[3]
            calc = np.exp(-1 * (((rowTest[0] - rowTrain[0])**2) + ((rowTest[1] - rowTrain[1]) ** 2)) / 2 * dataF[int(typeClass)])
            tmp = np.append(rowTrain, calc)
            dataG.append(tmp)
    return dataG

def ValidationTest(dataTest, resultTest, g):
    count = 0
    countDataTest = len(dataTest)
    for i, val in enumerate(dataTest):
        if (val[3] == resultTest[i]):
            count += 1
    return count/countDataTest, g

def main(dataTrain, dataTest, dataF):
    Prediction = []
    x = []
    for i, rowTest in enumerate(dataTest):
        dataG = cariG([rowTest], dataTrain, dataF)
        # 3 = Class. separateCol 3 = separate an array to many based on col 3 (class)
        separateG = separateCol(uniqueClass, dataG, 3)
        # 5 = G(x) per row
        sumSeparateG = sumCol(separateG, 5)
        Prediction.append(sumSeparateG.index(max(sumSeparateG)))
        tmp = np.append(rowTest, sumSeparateG.index(max(sumSeparateG)))
        x.append(tmp)
    return Prediction, x
    
separatedClass = separateCol(uniqueClass, dataSet, 3)

dataDistances = neighborDistance(separatedClass)

dataDistances = np.concatenate((dataDistances[0], dataDistances[1], dataDistances[2]), axis=0)
mergedSeparatedClass = np.concatenate((separatedClass[0], separatedClass[1], separatedClass[2]), axis=0)


# Merging array distances with all Data Set
dataWithDistanceMerged = []

for i, val in enumerate(mergedSeparatedClass):
    dataWithDistanceMerged.append(np.append(val, dataDistances[i]))
# ----
    
# =============================================================================
# After Distance Calculated
# =============================================================================

Data_train, Data_test = train_test_split(dataWithDistanceMerged, test_size = 0.2)

separatedDataTrainClasses = separateCol(uniqueClass, Data_train, 3)

dataTrainDistClass = np.array(Data_train)[:, (3, 4)]

separatedDataTrain = separateCol(uniqueClass, dataTrainDistClass, 0)

dataSumDistances = sumCol(separatedDataTrain, 1)

# =============================================================================
# Mulai butuh data set (sebelumnya belum butuh), 
# sebelumnya masih olah data train buat dapetin sum distance buat cari F
# =============================================================================

# =============================================================================
# Single (G) Validation
# =============================================================================
dataF = cariF(1.8, dataSumDistances, separatedDataTrain)

Prediction, x = main(Data_train, Data_test, dataF)

print(ValidationTest(Data_test, Prediction, 1.8))

dataX = np.array(x)[:,0]
dataY = np.array(x)[:,1]
dataZ = np.array(x)[:,2]
dataClass = np.array(x)[:, 5]

# ax.scatter(dataX, dataY, dataZ, c=dataClass, cmap='viridis', linewidth=0.5)

# =============================================================================
# End of Single (G) Validation
# =============================================================================

# =============================================================================
# Test with multiple G - Find the most optimal G
# =============================================================================

index = 0
Result = []
while (np.floor(index) != 10):
    count = 0
    
    dataF = cariF(index, dataSumDistances, separatedDataTrain)
    Prediction, x = main(Data_train, Data_test, dataF)
    Result.append(ValidationTest(Data_test, Prediction, index))
    
    index += .01

df = pd.DataFrame(Result)
df.to_csv('result5.csv', header=None)