from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from sklearn.neural_network import MLPClassifier

'''
1 Open the image file. The format of the file can be JPEG, PNG, BMP, etc.
2 Resize the image to match the input size for the Input layer of the Deep Learning model.
3 Convert the image pixels to float datatype.
4 Normalize the image to have pixel values scaled down between 0 and 1 from 0 to 255.
5 Image data for Deep Learning models should be either a numpy array or a tensor object.
'''


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


IMG_WIDTH = 200
IMG_HEIGHT = 200
img_folder = r'./dataset'


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


def generate_data():
    img_data, class_name = create_dataset(img_folder)  # lista de np array, lista de resultados
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val = np.array([target_dict[class_name[i]] for i in range(len(class_name))])  # np array
    img_data = np.array(img_data)

    nsamples, nx, ny = img_data.shape
    img_data = img_data.reshape((nsamples, nx * ny))

    X_train, X_test, y_train, y_test = train_test_split(img_data, target_val, test_size=0.25, stratify=target_val)

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)
    print(prediction)
    print(y_test)


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


generate_data()
