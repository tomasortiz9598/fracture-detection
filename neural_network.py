import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivada(x):
    return sigmoid(x)*(1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivada(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        self.layers = layers

        # Creo los arrays que contendran los pesos
        self.weights = []
        self.deltas = []

    def initPesos(self):
        # asignación de pesos aleatorios a cada una de las capas
        for i in range(1, len(self.layers) - 1):
            r = np.random.randn(self.layers[i - 1] + 1, self.layers[i] + 1)
            self.weights.append(r)
        # asigno aleatorios a capa de salida
        r = np.random.randn(self.layers[i] + 1, self.layers[i + 1])
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Con esto agregamos la unidad de Bias a la capa de entrada
        #ones = np.atleast_3d(np.ones(X.shape[0]))
        #X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)

            ################################################################

            # capa de salida
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # Empezamos en el segundo layer hasta el ultimo
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))
            self.deltas.append(deltas)

            # invertir
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # Actualizo el peso restandole un porcentaje del gradiente
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 1000 == 0 or k == (epochs - 1):
                deltaActual = self.deltas[k]
                print('epochs:', k, ' - error:', np.sum(deltaActual[1]))

    def predict(self, x):
        ones = np.atleast_2d(np.ones(x.shape[0]))
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def print_weights(self):
        for i in range(len(self.weights)):
            print(self.weights[i])

    def get_deltas(self):
        return self.deltas

