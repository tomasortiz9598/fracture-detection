from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from neural_network import NeuralNetwork
import random

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





def fit(X_train, X_test, y_train, y_test, mlp):
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 3000
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(mlp.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(mlp.score(X_test, y_test))

        epoch += 1

    """ Plot """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(scores_train)
    ax[0].set_title('Train')
    ax[1].plot(scores_test)
    ax[1].set_title('Test')
    fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.show()



def show_random_images():
    plt.figure(figsize=(20, 20))
    folder = r'./dataset/compuestas'
    for i in range(5):
        file = random.choice(os.listdir(folder))
        image_path = os.path.join(folder, file)
        img = mpimg.imread(image_path)
        ax = plt.subplot(1, 5, i + 1)
        ax.title.set_text(file)
        plt.imshow(img)
        plt.show()
