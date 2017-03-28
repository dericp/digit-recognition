import numpy as np
import pandas as pd
import math
from sklearn import svm
from matplotlib import pyplot as plt

# Build training set
df_train = pd.read_csv("../data/mnist_train.csv").sample(n=10000)
df_train['intercept'] = 1
trainingData = df_train.drop("label", axis = 1).values
trainingResults = df_train["label"].values

# Build test set
df_test = pd.read_csv("../data/mnist_test.csv").sample(n=2000)
df_test['intercept'] = 1
testData = df_test.drop("label", axis=1).values
testResults = df_test["label"].values


Cvals = [1, 10, 100, 1000, 5000, 10000]

rbfSVMErrors = []
linearSVMErrors = []

for i in range(len(Cvals)):
    Cval = Cvals[i]
    # build the validation set
    # build the validation set
    start_index = i * len(trainingData)//len(Cvals)
    end_index = len(trainingData)//len(Cvals) * (i + 1)
    
    validation_data = trainingData[start_index:end_index]
    validation_classifications = trainingResults[start_index:end_index]

    # build the model
    model = np.concatenate((trainingData[:start_index], trainingData[end_index:]), axis=0)
    model_classifications = np.concatenate((trainingResults[:start_index], trainingResults[end_index:]), axis=0)
    
    # Calculate RBF SVM error
    svm1 = svm.SVC(C=Cval)
    svm1.fit(model, model_classifications)
    rbfScore = svm1.score(validation_data, validation_classifications)
    rbfSVMErrors.append(1 - rbfScore)

    # Calculate Linear SVM error
    svm3 = svm.LinearSVC(C=Cval)
    svm3.fit(model, model_classifications)
    linearScore = svm3.score(validation_data, validation_classifications)
    linearSVMErrors.append(1 - linearScore)

plt.plot(Cvals, rbfSVMErrors)
plt.title("C vs. Validation Error on RBF SVMs")
plt.xscale('log')
plt.xlabel("C")
plt.ylabel("error")
plt.savefig('rbf_svm_CvsError.png')
plt.show()

plt.plot(Cvals, linearSVMErrors)
plt.title("C vs. Validation Error on Linear SVMs")
plt.xscale('log')
plt.xlabel("C")
plt.ylabel("error")
plt.savefig('linear_svm_CvsError.png')
plt.show()

"""We found that the Linear SVM had a markedly lower validation error than the RBF SVM. We were surprised by this. 
The best C value, according to our cross-validation, was C = 1, but we're skeptical that variations in validation error are 
due to variations in validation blocks rather than any impact our C value might have because the differences in error are so 
small. Thus, we'll build a Linear SVM model with C = 1 with our training set against the test set to get our test error."""

df_train = pd.read_csv("../data/mnist_train.csv").sample(frac=0.25)
df_train['intercept'] = 1
trainingData = df_train.drop("label", axis = 1).values
trainingResults = df_train["label"].values
df_test = pd.read_csv("../data/mnist_test.csv")
df_test['intercept'] = 1
testData = df_test.drop("label", axis=1).values
testResults = df_test["label"].values

# Find test error for RBF SVM with "optimal" C
classifier = svm.SVC(C=100)
classifier.fit(trainingData, trainingResults)
print("The test error of the RBF SVM is", 1 - classifier.score(testData, testResults))

# Find test error for Linear SVM with "optimal" C
classifier = svm.LinearSVC(C=1)
classifier.fit(trainingData, trainingResults)
print("The test error of the Linear SVM is", 1 - classifier.score(testData, testResults))