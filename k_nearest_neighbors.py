import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import math


VALIDATION_BLOCK_SIZE = 60
k_vals = [1, 5, 10, 50, 100]
df_train = pd.read_csv("data/train.csv", nrows=300)
training_data = df_train.drop("label", axis=1).values
training_classifications = df_train["label"].values
df_test = pd.read_csv("data/test.csv", nrows=300)


def main():
    ##### Run cross validation
    validation_errors = []
    for i in range(len(k_vals)):
        print('i: ' + str(i))
        k = k_vals[i]
        # Build the validation set
        validation_start = VALIDATION_BLOCK_SIZE * i
        validation_end = VALIDATION_BLOCK_SIZE * (i + 1)
        validation_data = training_data[validation_start:validation_end]
        validation_classifications = training_classifications[validation_start:validation_end]
        # Build the trainig set
        model = np.concatenate((training_data[:validation_start], training_data[validation_end:]), axis=0)
        model_classifications = np.concatenate((training_classifications[:validation_start], training_classifications[validation_end:]), axis=0)
        validation_error = get_validation_error(validation_data, validation_classifications, model, model_classifications, k)

        validation_errors.append(validation_error)

    plt.plot(k_vals, validation_errors)
    plt.title("k-Choice and validation error on k-NN digit classification")
    plt.xlabel("k value")
    #plt.xscale('log')
    plt.ylabel("validation error")
    #plt.show()
    plt.savefig('lol.png')


# @param: point1, point2 - arrays of pixel data for two points
# @return: euclidean distance between the two points
# Note: features are unweighted
def get_distance(point1, point2):
    assert(len(point1) == len(point2))
    distance = math.sqrt(
        np.sum(np.square([point1[i] - point2[i] for i in range(len(point1))])))
    return distance


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
