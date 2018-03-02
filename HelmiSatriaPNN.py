#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:17:07 2018

@author: helmisatria - 1301154325
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.model_selection import train_test_split

dataSet = np.genfromtxt('data_train_PNN.txt', skip_header=1)

dataX = dataSet[:,0]
dataY = dataSet[:,1]
dataZ = dataSet[:,2]
dataClass = dataSet[:,3]

uniqueClass = list(set(dataClass))

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
                    euc = euclidean(row, insideRow)
                    distances.append(euc)
            tmp = np.append(row, min(distances))
            dataClassDistance.append(tmp)
        dataDistances.append(dataClassDistance)
    dataDistances = np.concatenate((dataDistances))
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
        dataF.append(float(g * dataSumDistances[i])/len(separatedDataTrain[i]))
    return(dataF)

def cariG(titik, dataTrain, dataF):
    dataG = []
    for i, rowTrain in enumerate(dataTrain):
        typeClass = rowTrain[3]
        #print('typeClass ', typeClass)
        calc = np.exp(-1 * (((titik[0] - rowTrain[0])**2) + ((titik[1] - rowTrain[1]) ** 2) + ((titik[2] - rowTrain[2]) ** 2)) / 2 * (dataF[int(typeClass)]) ** 2)
        tmp = np.append(rowTrain, calc)
        dataG.append(tmp)
    return dataG

def validationTest(resultTest):
    count = 0
    countDataTest = len(resultTest)
    for i, val in enumerate(resultTest):
        if (val[3] == val[5]):
            count += 1
    return count/countDataTest

def dataPreparation(dataSet):
    separatedClass = separateCol(uniqueClass, dataSet, 3)
    dataDistances = neighborDistance(separatedClass)
    return dataDistances

def main(dataTrain, titikDicari, dataF):
    dataG = cariG(titikDicari, dataTrain, dataF)
    # 3 = Class. separateCol 3 = separate an array to many based on col 3 (class)
    separateG = separateCol(uniqueClass, dataG, 3)
    # 5 = G(x) (column) per row
    sumSeparateG = sumCol(separateG, 5)
    x = np.append(titikDicari, sumSeparateG.index(max(sumSeparateG)))
    return x

def mainAllTest(dataTrain, DataTest, g):
    results = []
    
    dataDistances = dataPreparation(dataTrain)
    
    dataTrainDistClass = np.array(dataDistances)[:, (3, 4)]
    
    separatedDataTrain = separateCol(uniqueClass, dataTrainDistClass, 0)
    
    dataSumDistances = sumCol(separatedDataTrain, 1)

    for i, rowTest in enumerate(DataTest):
        dataF = cariF(g, dataSumDistances, separatedDataTrain)
        x = main(Data_train, rowTest, dataF)
        results.append(x)
    return results

# =============================================================================
# =============================================================================
# # Training purposes
# =============================================================================
# =============================================================================
z2dataDistances = dataPreparation(dataSet)
# Split Data train
Data_train, Data_test = train_test_split(z2dataDistances, test_size = 0.2)
# End of Split Data train
z3separatedDataTrainClasses = separateCol(uniqueClass, Data_train, 3)
z4dataTrainDistClass = np.array(Data_train)[:, (3, 4)]
z5separatedDataTrain = separateCol(uniqueClass, z4dataTrainDistClass, 0)
z6dataSumDistances = sumCol(z5separatedDataTrain, 1)

# =============================================================================
# Mulai butuh data tes set (sebelumnya belum butuh), 
# sebelumnya masih olah data train buat dapetin sum distance buat cari F
# =============================================================================

#Single (G) Validation

dataF = cariF(0.9, z6dataSumDistances, z5separatedDataTrain)
x = main(Data_train, Data_test[0], dataF)
# =============================================================================

#Train all data set (Training)

zzResults = mainAllTest(Data_train, Data_test, 0.9)
print(validationTest(zzResults), '%')

# =============================================================================
# Find the most optimal for G
# =============================================================================
def searchOptimumG():
    index = 0
    Result = []
    while (np.floor(index) != 2):
        zzResults = mainAllTest(Data_train, Data_test, index)
        Result.append([validationTest(zzResults), index])
        
        print(Result)
        
        index += .01
        
    df = pd.DataFrame(Result)
    df.to_csv('z12.csv', header=None, index=False)

#searchOptimumG()
# =============================================================================
# End of Find the most optimal for G
# =============================================================================

# =============================================================================
# Testing with real data test
# =============================================================================

dataTest= np.genfromtxt('data_test_PNN.txt', skip_header=1)

zzResults = mainAllTest(dataSet, dataTest, 0.9)

dataX = np.array(zzResults)[:,0]
dataY = np.array(zzResults)[:,1]
dataZ = np.array(zzResults)[:,2]
#dataClass = np.array(zzResults)[:, 5]

dataClass = np.array(zzResults)[:, 3]

try1 = plt.axes(projection='3d')
try1.scatter(dataX, dataY, dataZ, c=dataClass, cmap='viridis', linewidth=0.5)

df = pd.DataFrame(dataClass)
df.to_csv('prediksi.txt', header=None, index=False)

# =============================================================================
# try1 = plt.axes(projection='3d')
# try1.scatter(dataX, dataY, dataZ, cmap='viridis', linewidth=0.5)
# =============================================================================
