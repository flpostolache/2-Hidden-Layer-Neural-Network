import numpy as np
import copy
import matplotlib.pyplot as plt
import functii as fc
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Loading the dataset into a variable and separate the inputs from outputs
mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T
mnist_data_pt_calcule = mnist["data"]
mnist_label = mnist["label"][0].reshape(70000, 1)

matrice_raspuns = np.zeros((mnist_label.shape[0], 10))
for i in range(len(mnist_label)):
    matrice_raspuns[i][int(mnist_label[i][0])] = 1

# Splitting the dataset into the trainset and testset
X_train, X_test, Y_train, Y_test = train_test_split(
    mnist_data, matrice_raspuns, random_state=4
)

# Transposing the matrices, to put each input/output in a column
X_train = X_train.T
Y_train = Y_train.T
X_test = X_test.T
Y_test = Y_test.T

# Centering and standardizing the dataset
X_train = X_train / 255
X_test = X_test / 255

parameters, costs = fc.L_layer_model(X_train, Y_train, 2, 20, 0.09, 4301, True)

Y_train_pred = fc.predict(X_train, Y_train, parameters)
Y_test_pred = fc.predict(X_test, Y_test, parameters)

# saving the parameters
with open("Valori_w_si_b.txt", "wb") as f:
    for i in parameters:
        np.save(f, parameters[i])

para = {}
with open("Valori_w_si_b.txt", "rb") as d:
    for i in range(1, 4):
        para["W" + str(i)] = np.load(d)
        para["b" + str(i)] = np.load(d)
