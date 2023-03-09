from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import pickle 

def calculateDistances(x_test, x_train, y_train):
	for i in range(10):
		s = i * 1000
		e = (i * 1000) + 1000
		distnaces = pairwise_distances(x_test[s:e], x_train)

		distMatrix = []
		for distVec in distnaces:
			distDF = pd.DataFrame({"Distance":distVec, "Label":y_train})
			distDF = distDF.sort_values("Distance").iloc[:60]
			distMatrix.append(distDF)

		#Pickle Distance Vector##
		pickle.dump(distMatrix, open('distMatrix'+str(i)+'.sav','wb'))

	# Load Distance Vector ##
	distTable = []
	for i in range(10):
		distMatrix = pickle.load(open('distMatrix'+str(i)+'.sav', 'rb'))
		distTable.append(distMatrix)

	return distTable

def knn(k, distVect):
	kLabels = distVect["Label"].values[:k]

	uLabel = np.unique(kLabels)

	labelDict = {}

	for label in uLabel:
		labelDict[label] = 0

	for label in kLabels:
		labelDict[label] += 1

	return max(labelDict, key=labelDict.get)

def knnAnalysis(k, distTable, y_test):

	i = 0
	correct = 0
	for distMatrix in distTable:
		for distVect in distMatrix:
			pred = knn(k, distVect)
			if pred == y_test[i]:
				correct += 1
			i += 1

	accuracy = (float(correct)/float(len(x_test))) * 100

	print("Total Accuracy: " + str(accuracy))

	return accuracy

## MAIN ##

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train  = np.asarray([x.flatten() for x in x_train])
x_test  = np.asarray([x.flatten() for x in x_test])

distTable = calculateDistances(x_test, x_train, y_train)

kVals = [1, 3, 5, 10, 20, 30, 40, 50, 60]
accVector = []
for k in kVals:
	print("K: " + str(k))
	accVector.append(knnAnalysis(k, distTable, y_test))
	print('\n')

x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(accVector)
plt.xticks(x, kVals)
plt.ylabel("Accuracy")
plt.xlabel("K Value")
plt.title("k-NN Observations")
plt.show()
