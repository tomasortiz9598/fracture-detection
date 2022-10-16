from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
'''
1 Open the image file. The format of the file can be JPEG, PNG, BMP, etc.
2 Resize the image to match the input size for the Input layer of the Deep Learning model.
3 Convert the image pixels to float datatype.
4 Normalize the image to have pixel values scaled down between 0 and 1 from 0 to 255.
5 Image data for Deep Learning models should be either a numpy array or a tensor object.
'''


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.size)


def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


def show_random_images():
    plt.figure(figsize=(20, 20))
    img_folder = r'./dataset/compuestas'
    for i in range(5):
        file = random.choice(os.listdir(img_folder))
        image_path = os.path.join(img_folder, file)
        img = mpimg.imread(image_path)
        ax = plt.subplot(1, 5, i+1)
        ax.title.set_text(file)
        plt.imshow(img)
        plt.show()


IMG_WIDTH = 200
IMG_HEIGHT = 200
img_folder = r'./dataset'


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


def generate_data():
    img_data, class_name = create_dataset(img_folder) #lista de np array, lista de resultados
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val = np.array([target_dict[class_name[i]] for i in range(len(class_name))]) #np array
    dummies = pd.get_dummies(target_val).values
    print(dummies[:3])

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


generate_data()
