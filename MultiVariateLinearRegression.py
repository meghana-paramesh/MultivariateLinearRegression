import numpy as np
from Plot import plot_error
import math


class LinearRegression:
    def __init__(self, learning_rate=1e-3, epochs=1000):
        self.n_features = None
        self.n_samples = None
        self.lr = learning_rate
        self.epochs = epochs
        self.theta_1 = None
        self.theta_0 = None
        self.theta_2 = None

    def get_prediction(self, X):
        return np.dot(X, self.theta_1) + self.theta_0

    # Initialize both theta_0 and theta_1 to 0
    def initialize_parameters(self):
        self.theta_1 = np.zeros(self.n_features)
        self.theta_1 = self.theta_1.reshape(self.n_features, 1)
        self.theta_0 = 0

    # Update theta_1 and theta_0 for the nest iteration
    def update_parameters(self, d_theta1, d_theta0):
        self.theta_1 -= self.lr * d_theta1
        self.theta_0 -= self.lr * d_theta0

    def get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        # compute the gradients of weight & bias
        d_theta_1 = (1 / self.n_samples) * np.dot(X.T, (y_pred - y))
        d_theta_0 = (1 / self.n_samples) * np.sum(y_pred - y)
        return d_theta_1, d_theta_0

    # Fit a model where the entire m is considered as the batch size where the stopping
    def fit(self, X, y, stopping_condition="cost_based", stopping_condition_value=0.000001):
        print("Stopping condition used: ",stopping_condition)
        # get number of data points & features
        self.n_samples, self.n_features = X.shape
        # initialize theta_1 & theta_0
        self.initialize_parameters()
        y_prediction = None

        theta_1_old = None
        theta_0_old = None
        E2 = None
        E1 = None
        E0 = None

        train_error_array = np.zeros(self.epochs)
        # perform gradient descent for n iterations
        i = 0

        if stopping_condition == "l2_norm_based":
            while (E1 is None and E0 is None) or math.sqrt((E0 * E0) + (E1 * E1) + (E2 * E2)) > stopping_condition_value:

                y_prediction = self.get_prediction(X)
                # compute gradients
                d_theta1, d_theta0 = self.get_gradients(X, y, y_prediction)
                # update weights & bias with gradients
                self.update_parameters(d_theta1, d_theta0)

                if theta_1_old is not None and theta_0_old is not None:
                    E2 = (theta_1_old[1] - self.theta_1[1])
                    E1 = (theta_1_old[0] - self.theta_1[0])
                    E0 = (theta_0_old - self.theta_0)

                # error = cost_function(y, y_prediction)
                # train_error_array[i] = error / len(X)
                i = i + 1
                theta_1_old = self.theta_1.copy()
                theta_0_old = self.theta_0.copy()

        if stopping_condition == "cost_based":
            old_error = None
            error = None
            while (old_error is None or error is None) or abs(old_error-error) > stopping_condition_value:

                y_prediction = self.get_prediction(X)
                # compute gradients
                d_theta1, d_theta0 = self.get_gradients(X, y, y_prediction)
                # update weights & bias with gradients
                self.update_parameters(d_theta1, d_theta0)
                if error is not None:
                    old_error = error.copy()
                error = cost_function(y, y_prediction)/len(X)
                # train_error_array[i] = error
                i = i + 1
        # plot_error(train_error_array, self.epochs)

        return y_prediction

    # Fit a model where for a given batch size
    def fit_batch(self, X, y, stopping_condition, batch_size):
        # get number of data points & features
        self.n_samples, self.n_features = X.shape
        # initialize theta_1 & theta_0
        self.initialize_parameters()
        print("X size: ", X.shape)

        train_error_array = np.zeros(self.epochs)
        # perform gradient descent for y-y_hat is lesser than the stopping condition or till a number of iterations
        for i in range(self.epochs):
            error = 0
            prev_predictions = []
            for k in range(0, len(X), batch_size):
                x_batch = X[k:k + batch_size]
                y_batch = y[k:k + batch_size]

                # get y_prediction
                y_prediction = self.get_prediction(x_batch)

                # get the partial derivatives by computing gradients
                d_theta_1, d_theta_0 = self.get_gradients(x_batch, y_batch, y_prediction)

                # update theta_1 & theta_0 with gradients
                self.update_parameters(d_theta_1, d_theta_0)
                error += cost_function(y_batch, y_prediction)
                prev_predictions = np.append(prev_predictions, y_prediction)
            train_error_array[i] = error / len(X)

            if error < stopping_condition:
                break
        plot_error(train_error_array)
        return prev_predictions

    # predict the test values using the calculated theta_0 and theta_1
    def predict(self, X):
        y_prediction = self.get_prediction(X)
        return y_prediction


def cost_function(y_true, y_prediction):
    return np.mean(np.power(y_true - y_prediction, 2))
