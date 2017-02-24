import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math


VALIDATION_BLOCK_SIZE = 200
k_vals = [1, 5, 10, 50, 100]
df_train = pd.read_csv("data/train.csv", nrows = 1000)
training_data = df_train.drop("label", axis = 1).values
training_results = df_train["label"].values
df_test = pd.read_csv("data/test.csv", nrows = 500)


# @param: point1, point2 - arrays of pixel data for two points
# @return: euclidean distance between the two points
# Note: features are unweighted
def getDistance(point1, point2):
    assert(len(point1) == len(point2))
    distance = math.sqrt(
        np.sum(np.square([point1[i] - point2[i] for i in range(len(point1))])))
    return distance
        
# @param: point - array of pixel data from MNIST dataset relating to query point image
# @param: data - the dataset to finding the k closest neighbors from
# @param: k - number of numbers to find
# @return: an array holding the k nearest neighbors of the query point, an array holding the respective distances
def getKNN(point, data, k):
    neighbors = []  # array holding indexes of the k closest neighbors
    distances = [] # array holding distances of respective neighbors
    for i in range(len(data)):
        neighbors.append(i)
        distances.append(getDistance(point, data[i]))
        # Limit the number of neighbors to just k
        if len(neighbors) > k:
            # Invariant: neighbors, distances size is k + 1
            sortedZip = sorted(zip(distances, neighbors))
            neighbors = [neighbor for (distance, neighbor) in sortedZip]
            distances = [distance for (distance, neighbor) in sortedZip]
            neighbors.pop()
            distances.pop()
            #Invariant: neighbors, distances size is k
    return neighbors, distances

# @param: neighbors - array of indices of points
# @param: results - array of output classifications
# @param: distances - array of distances of each respective point in neighbors from a particular point
# @return: a prediction of classification based off the classification with the lowest average distance from a point
def makePrediction(neighbors, results, distances):
    predictionMap = {}
    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        distance = distances[i]
        neighborPrediction = results[neighbor]
        # Create a dictionary relating each possible prediction to a list of distances from query point
        if neighborPrediction not in predictionMap:
            predictionMap[neighborPrediction] = []
        predictionMap[neighborPrediction].append(distance)
    # Relate each prediction to an average distance (among the neighbors)
    for prediction in predictionMap:
        predictionMap[prediction] = np.sum(predictionMap[prediction])/len(predictionMap[prediction])
    return min(predictionMap, key = predictionMap.get)

# @param: validationBlock - array of arrays holding point data for the validation set
# @param: validationResults - array of integers corresponding to the respective classifications of the validation block
# @param: nonValidationBlock - array of arrays holding point data for the training set
# @param: nonValidationResults - array of integers corresponding to the respective classifications of the training block
# @return: the ratio of incorrectly classified numbers in the validation set
def getValidationError(validationBlock, validationResults, nonValidationBlock, nonValidationResults, k):
    misclassified = 0
    for i in range(len(validationBlock)):
        point = validationBlock[i]
        neighbors, distances = getKNN(point, nonValidationBlock, k)
        prediction = makePrediction(neighbors, nonValidationResults, distances)
        if prediction != validationResults[i]:
            misclassified += 1
    return misclassified/len(validationBlock)

##### Run cross validation
validationErrors = []
for i in range(len(k_vals)):
    k = k_vals[i]
    # Build the validation set
    validationStart = VALIDATION_BLOCK_SIZE * i
    validationEnd = VALIDATION_BLOCK_SIZE * (i + 1)
    validationBlock = training_data[validationStart : validationEnd]
    validationResults = training_results[validationStart : validationEnd]
    # Build the nonvalidation set
    nonValidationBlock = []
    nonValidationResults = []
    for j in range(len(training_data)):
        if j < validationStart or j >= validationEnd:
            nonValidationBlock.append(training_data[j])
            nonValidationResults.append(training_results[j])
    # Calculate validation error
    validationError = getValidationError(validationBlock, validationResults, nonValidationBlock, nonValidationResults, k)
    validationErrors.append(validationError)

plt.plot(k_vals, validationErrors)
plt.title("k-Choice and validation error on k-NN digit classification")
plt.xlabel("k value")
plt.xscale('log')
plt.ylabel("validation error")
plt.show()
