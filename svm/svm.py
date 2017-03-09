import numpy as np
import pandas as pd
import random

step_size = 1e-12 # best step size we found
C = 5000 # best c value we found
step_sizes = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
C_vals = [500, 1000, 5000, 10000, 20000]
EPOCHS = 10
lambda_val = 1e-6
sigma = 10
m = 3000
dev = False
PEGASOS = True


def main():
    df_train = pd.read_csv('../data/mnist_train.csv')
    df_train['bias'] = 1

    if not dev:
        X = df_train.drop('label', axis=1).values
        y = df_train['label'].values

        df_test = pd.read_csv('../data/mnist_test.csv')
        df_test['bias'] = 1
        X_test = df_test.drop('label', axis=1).values
        y_test = df_test['label'].values

        w_s = {}  # a map from classification {0, ..., 9} to weight vector
        for class_val in range(10):
            if PEGASOS:
                w = pegasos(X, y, class_val, EPOCHS)
            else:
                w = sgd(X, y, class_val, step_size, C, EPOCHS)
            w_s[class_val] = w

        test_error = calculate_test_error(w_s, X_test, y_test)
        print('test error: ', test_error)
    else:
        for step_size in step_sizes:
            for C in C_vals:
                print('step size: ', step_size, ' c: ', C)

                df_train_dev, df_dev = np.split(df_train.sample(frac=1), [int(.8 * len(df_train))])
                X = transform(df_train_dev.drop('label', axis=1).values)
                y = df_train_dev['label'].values
                X_dev = transform(df_dev.drop("label", axis=1).values)
                y_dev = df_dev['label'].values

                # Training the SVM classifiers
                w_s = {} # a map from classification {0, ..., 9} to weight vector
                for class_val in range(10):
                    if PEGASOS:
                        w = pegasos(X, y, class_val, EPOCHS)
                    else:
                        w = sgd(X, y, class_val, step_size, C, EPOCHS)
                    w_s[class_val] = w

                test_error = calculate_test_error(w_s, X_dev, y_dev)
                print('test error: ', test_error)


# transforms X into m-dimensional feature vectors using RFF and RBF kernel
# Make sure this function works for both 1D and 2D NumPy arrays.
def transform(X):
    np.random.seed(0)
    b = np.random.rand(m) * 2 * np.pi
    w = np.random.multivariate_normal(np.zeros(X.shape[1]), sigma**2 * np.identity(X.shape[1]), m)
    transformed = (2.0 / m)**0.5 * np.cos(np.dot(X, w.T) + b)
    # feature normalization
    transformed = (transformed - np.mean(transformed, 0)) / np.std(transformed, 0)

    return transformed


def sgd(X, y, class_val, step_size, C, epochs):
    w = np.zeros(len(X[0]))
    for epoch in range(epochs):
        points = list(range(len(X)))
        random.shuffle(points)
        for point in points:
            # is this point in the class we are looking for?
            if y[point] == class_val:
                y_i = 1
            else:
                y_i = -1
            # update step
            partial = C * max(0, 1 - y_i * np.dot(X[point], w)) * (y_i * X[point])
            partial = partial - np.multiply(2, w)
            partial = step_size * partial
            w += partial
    return w


def pegasos(X, y, class_val, epochs):
    w = np.zeros(X.shape[1])
    t = 1
    for epoch in range(epochs):
        points = list(range(X.shape[0]))
        random.shuffle(points)
        for point in points:
            if y[point] == class_val:
                y_i = 1
            else:
                y_i = -1

            learning_rate = 1 / (lambda_val * t)

            w = (1 - learning_rate * lambda_val) * w + learning_rate * hinge_loss_gradient(w, X[point], y_i)
            # optional projection step
            w = min(1, ((1 / lambda_val**0.5) / np.linalg.norm(w))) * w

            t += 1

    return w


# calculate the gradient of the hinge loss function
def hinge_loss_gradient(w, x, y):
    if np.dot(w, x) * y >= 1:
        return 0
    else:
        return y * x


def calculate_test_error(w_s, X_test, y_test):
    misclassified = 0
    for idx in range(X_test.shape[0]):
        x = X_test[idx]

        # Find class with maximum margin and classify x as that class
        # Fencepost: initialize max with margin denoted by 0
        max_margin = np.dot(w_s[0], x)
        max_class = 0
        for class_val in range(1, 10):
            margin = np.dot(w_s[class_val], x)
            if margin > max_margin:
                max_margin = margin
                max_class = class_val

        print(max_class, ' ', y_test[idx])
        if max_class != y_test[idx]:
            misclassified += 1

    print('misclassified: ', misclassified)
    print('total: ', len(X_test))
    return misclassified / len(X_test)


if __name__ == '__main__':
    main()
