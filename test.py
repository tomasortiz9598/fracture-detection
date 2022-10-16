from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork

def _sample(array, n_samples):
    ""
    "Little utility function, sample n_samples with replacement"
    ""
    idx = np.random.choice(np.arange(len(array)), n_samples, replace = True)
    return array[idx]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.size)


def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

def saran(np_img_data, target_val):
    nn = NeuralNetwork([199, 199, 1], 'tanh')

    nn.initPesos()
    nn.fit(np_img_data, target_val, epochs=10000)

    deltas = nn.get_deltas()
    valores = []
    for arreglo in deltas:
        valores.append(arreglo[0] + arreglo[1])

    plt.plot(range(len(valores)), valores, color='b')
    plt.ylim([0, 1])
    plt.ylabel('Error')
    plt.xlabel('Épocas')
    plt.tight_layout()
    plt.show()


def train(img_data, dummies):
    x_train, x_test, y_train, y_test = train_test_split(img_data, dummies, test_size=20, random_state=4)

    learning_rate = 0.1
    iterations = 5000
    N = y_train.size

    # number of input features
    input_size = 4

    # number of hidden layers neurons
    hidden_size = 2

    # number of neurons at the output layer
    output_size = 3

    results = pd.DataFrame(columns=["mse", "accuracy"])
    # Initialize weights
    np.random.seed(10)

    # initializing weight for the hidden layer
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

    # initializing weight for the output layer
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))

    for itr in range(iterations):
        # feedforward propagation
        # on hidden layer
        Z1 = np.dot(x_train, W1)
        A1 = sigmoid(Z1)

        # on output layer
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)

        # Calculating error
        mse = mean_squared_error(A2, y_train)
        acc = accuracy(A2, y_train)
        results = results.append({"mse": mse, "accuracy": acc}, ignore_index=True)

        # backpropagation
        E1 = A2 - y_train
        dW1 = E1 * A2 * (1 - A2)

        E2 = np.dot(dW1, W2.T)
        dW2 = E2 * A1 * (1 - A1)

        # weight updates
        W2_update = np.dot(A1.T, dW1) / N
        W1_update = np.dot(x_train.T, dW2) / N

        W2 = W2 - learning_rate * W2_update
        W1 = W1 - learning_rate * W1_update