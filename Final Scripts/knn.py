
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import math


K_VALS = [1, 5, 25, 125, 625]
PERCENT_VALIDATION = 0.2
BANDWIDTHS = [10000, 50000, 100000, 1000000]


def main():
    df_train = pd.read_csv("train.csv").sample(frac=0.1)
    training_data = df_train.drop("label", axis=1).values
    training_classifications = df_train["label"].values
    validation_set_size = int(training_data.shape[0] * PERCENT_VALIDATION)
    print(str(validation_set_size))
    print(str(training_data.shape))

    validation_errors = []
    for i in range(len(K_VALS)):
        print('k = ' + str(K_VALS[i]))
        k = K_VALS[i]

        # build the validation set
        start_index = validation_set_size * i
        end_index = validation_set_size * (i + 1)

        validation_data = training_data[start_index:end_index]
        validation_classifications = training_classifications[start_index:end_index]

        # build the model
        model = np.concatenate((training_data[:start_index], training_data[end_index:]), axis=0)
        model_classifications = np.concatenate((training_classifications[:start_index], training_classifications[end_index:]), axis=0)

        validation_error = get_validation_error(validation_data, validation_classifications, model, model_classifications, k)
        validation_errors.append(validation_error)

    plt.plot(K_VALS, validation_errors)
    plt.title("K vs. Classification Error")
    plt.xlabel("k value")
    plt.xscale('log')
    plt.ylabel("classification error")
    plt.savefig('k-nn.png')
    plt.clf()
    
    validation_errors = []
    # Kernalize with the Gaussian kernel over all the points
    for i in range(len(BANDWIDTHS)):
        print("bandwidth:", str(BANDWIDTHS[i]))
        bandwidth = BANDWIDTHS[i]
        
        # build the validation set
        start_index = validation_set_size * i
        end_index = validation_set_size * (i + 1)

        validation_data = training_data[start_index:end_index]
        validation_classifications = training_classifications[start_index:end_index]

        # build the model
        model = np.concatenate((training_data[:start_index], training_data[end_index:]), axis=0)
        model_classifications = np.concatenate((training_classifications[:start_index], training_classifications[end_index:]), axis=0)
        
        validation_error = get_validation_error_gaussian(validation_data, validation_classifications, model, model_classifications, bandwidth)
        validation_errors.append(validation_error)
        
    plt.plot(BANDWIDTHS, validation_errors)
    plt.title("Bandwidth vs. Classification Error")
    plt.xlabel("k value")
    plt.xscale('log')
    plt.ylabel("classification error")
    plt.savefig('k-nn-gaussian.png')
    plt.clf()
    
def test():
    df_train = pd.read_csv("../data/mnist_train.csv").sample(frac=0.25)
    df_train['intercept'] = 1
    trainingData = df_train.drop("label", axis = 1).values
    trainingResults = df_train["label"].values
    df_test = pd.read_csv("../data/mnist_test.csv")
    df_test['intercept'] = 1
    testData = df_test.drop("label", axis=1).values
    testResults = df_test["label"].values

    print("Nearest neighbor error is", get_validation_error(testData, testResults, trainingData, trainingResults, 1))
    print("Gaussian kernel regression error is", get_validation_error_gaussian(testData, testResults, trainingData, trainingResults, 10000))

# @param: point1, point2 - arrays of pixel data for two points
# @return: euclidean distance between the two points
# Note: features are unweighted
def get_distance(point1, point2):
    assert(len(point1) == len(point2))
    distance = math.sqrt(
        np.sum(np.square([point1[i] - point2[i] for i in range(len(point1))])))
    return distance

def get_validation_error_gaussian(validation_block, validation_results, model, model_classifications, bandwidth):
    misclassified = 0
    for i in range(len(validation_block)):
        point = validation_block[i]
        prediction = make_prediction_gaussian(point, model, model_classifications, bandwidth)
        if prediction != validation_results[i]:
            misclassified += 1
    return misclassified / len(validation_block)

def make_prediction_gaussian(point, model, model_classifications, bandwidth):
    classToKernelizedDistances = {}
    denominator = 0
    numerator = 0
    for i in range(len(model)):
        distance = get_distance(point, model[i])
        #distance /= 1000 # Scale distance to avoid overflow
        # Use the gaussian kernel
        kernelizedDistance = math.exp(-1 * distance * distance/bandwidth)
        denominator += kernelizedDistance
        classification = model_classifications[i]
        numerator += kernelizedDistance * classification   
    return int(round(numerator/denominator))        

# @param: point - array of pixel data from MNIST dataset relating to query point image
# @param: data - the dataset to finding the k closest neighbors from
# @param: k - number of numbers to find
# @return: an array holding the k nearest neighbors of the query point, an array holding the respective distances
def get_knn(point, data, k):
    neighbors_and_dists = []  # array holding indexes of the k closest neighbors

    for i in range(k):
        neighbors_and_dists.append((i, np.linalg.norm(point - data[i])))
    neighbors_and_dists.sort(key=lambda tup: tup[1])
    #print(str(neighbors_and_dists))
    for i in range(k + 1, len(data)):
        dist = np.linalg.norm(point - data[i])
        if dist < neighbors_and_dists[k - 1][1]:
            search_index = k - 2
            while search_index >= 0 and neighbors_and_dists[search_index][1] > dist:
                search_index -= 1

            neighbors_and_dists.insert(search_index + 1, (i, dist))

    return list(neighbors_and_dists[:k])


# @param: neighbors - array of indices of points
# @param: results - array of output classifications
# @param: distances - array of distances of each respective point in neighbors from a particular point
# @return: a prediction of classification based off the classification with the lowest average distance from a point
def make_prediction(neighbors_and_dists, classifications):
    prediction_map = defaultdict(list)
    for neighbor, distance in neighbors_and_dists:
        neighbor_prediction = classifications[neighbor]

        # Create a dictionary relating each possible prediction to a list of distances from query point
        prediction_map[neighbor_prediction].append(distance)

    # Relate each prediction to an average distance (among the neighbors)
    for prediction in prediction_map:
        prediction_map[prediction] = sum(prediction_map[prediction]) / float(len(prediction_map[prediction]))

    return min(prediction_map, key=prediction_map.get)


# @param: validationBlock - array of arrays holding point data for the validation set
# @param: validationResults - array of integers corresponding to the respective classifications of the validation block
# @param: nonValidationBlock - array of arrays holding point data for the training set
# @param: nonValidationResults - array of integers corresponding to the respective classifications of the training block
# @return: the ratio of incorrectly classified numbers in the validation set
def get_validation_error(validation_block, validation_results, non_validation_block, non_validation_classifications, k):
    misclassified = 0.0
    for i in range(len(validation_block)):
        point = validation_block[i]
        neighbors_and_dists = get_knn(point, non_validation_block, k)
        prediction = make_prediction(neighbors_and_dists, non_validation_classifications)
        if prediction != validation_results[i]:
            misclassified += 1
    return misclassified / len(validation_block)


if __name__ == '__main__':
    main()
    """Like the baseline we found, through cross validation, that a k of 1 to be best for our k-NN regression. 
    For kernel regression, we found a bandwidth of 10^4 was best for our Nadaraya-Watson Gaussian kernel regression. 
    Let's find out our test error with a model that is 1/4 of the training set for both algorithms."""
    test()

