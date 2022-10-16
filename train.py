from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

'''
1 Open the image file. The format of the file can be JPEG, PNG, BMP, etc.
2 Resize the image to match the input size for the Input layer of the Deep Learning model.
3 Convert the image pixels to float datatype.
4 Normalize the image to have pixel values scaled down between 0 and 1 from 0 to 255.
5 Image data for Deep Learning models should be either a numpy array or a tensor object.
'''


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

    mlp = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(50, 2), random_state=1)
    mlp.fit(X_train, y_train)
    prediction = mlp.predict(X_test)

    # Esperado vs obtenido
    print(prediction)
    print(y_test)

    # Matriz de confusion y metricas
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))

    # Loss
    loss_values = mlp.loss_curve_
    plt.plot(loss_values)
    plt.show()




generate_data()
