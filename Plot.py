import matplotlib.pyplot as plt
import numpy as np


def initial_plot():
    X, Y = np.loadtxt('data.txt', delimiter=',', unpack=True)
    plt.figure(0)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.scatter(X, Y, c="red", marker="x", linewidth=0.5, label="Training data")

    # plt.show()
    plt.savefig('initial_plot.png')


def plot_error(train_error_array, epochs):
    epoch = list(range(0, epochs))
    plt.figure(1)
    plt.xlabel('Iterations/ Epochs')
    plt.ylabel('Training error')
    plt.plot(epoch, train_error_array, color="red")
    plt.savefig('training_error.png')
