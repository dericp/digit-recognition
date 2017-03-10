
# coding: utf-8

# In[24]:

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

K_VALS = [1, 5, 25, 125, 625]
df_train = pd.read_csv("train.csv", nrows = 1000)
df_train['intercept'] = 1
trainingData = df_train.drop("label", axis = 1).values
trainingResults = df_train["label"].values

errors = []
for i in range(len(K_VALS)):
    k = K_VALS[i]
    # build the validation set
    start_index = i * len(trainingData)//len(K_VALS)
    end_index = len(trainingData)//len(K_VALS) * (i + 1)

    validation_data = trainingData[start_index:end_index]
    validation_classifications = trainingResults[start_index:end_index]

    # build the model
    model = np.concatenate((trainingData[:start_index], trainingData[end_index:]), axis=0)
    model_classifications = np.concatenate((trainingResults[:start_index], trainingResults[end_index:]), axis=0)
    
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
    classifier.fit(model, model_classifications)
    score = classifier.score(validation_data, validation_classifications)
    errors.append(1 - score)

plt.plot(K_VALS, errors)
plt.title("K vs. Classification Error")
plt.xlabel("k value")
plt.xscale('log')
plt.ylabel("classification error")
plt.savefig('k-nn-libraryimpl.png')
plt.show()

