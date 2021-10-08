import copy as cp
import numpy as np
import functiii as fct


def get_sizes(X, Y):
    return X.shape[0], Y.shape[0]


def sigmoid(z):

    # my own sigmoid function
    s = 1 / (1 + np.exp(-z))
    return s, s


def tanh(z):
    s = np.tanh(z)
    return s, s


def relu(z):
    s = z * (z > 0)
    return s, s


def initialize_parameters(n_x, n_y, number_of_hidden_layers, dimension_of_layer):
    np.random.seed(2)

    parameters = {}
    if number_of_hidden_layers == 0:
        parameters["W1"] = np.random.randn(n_y, n_x) * 0.01
        parameters["b1"] = np.zeros((n_y, 1))

    else:
        parameters["W1"] = np.random.randn(dimension_of_layer, n_x) * 0.01
        parameters["b1"] = np.zeros((dimension_of_layer, 1))
        for i in range(2, number_of_hidden_layers + 1):
            parameters["W" + str(i)] = (
                np.random.randn(dimension_of_layer, dimension_of_layer) * 0.01
            )
            parameters["b" + str(i)] = np.zeros((dimension_of_layer, 1))

        parameters["W" + str(number_of_hidden_layers + 1)] = (
            np.random.randn(n_y, dimension_of_layer) * 0.01
        )
        parameters["b" + str(number_of_hidden_layers + 1)] = np.zeros((n_y, 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cp.deepcopy(cache))

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )
    caches.append(cp.deepcopy(cache))

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)
    cost = -np.sum(logprobs) * 1 / m

    cost = float(np.squeeze(cost))

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    Z = cp.deepcopy(activation_cache)
    Z[Z < 0] = 0
    Z[Z >= 0] = 1
    dZ = np.multiply(dA, Z)
    return dZ


def sigmoid_backward(dA, activation_cache):
    dZ = np.multiply(activation_cache, dA)
    dZ = np.multiply(dZ, (1.0 - activation_cache))
    return dZ


def tanh_backward(dA, activation_cache):
    Z = cp.deepcopy(activation_cache)
    Z = 1.0 - np.power(Z, 2)
    dZ = np.multiply(dA, Z)
    return dZ


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]

    dAL = -np.divide(Y, AL) + np.divide((Y - 1.0), (AL - 1.0))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dAL, current_cache, "sigmoid"
    )
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(params, grads, learning_rate):
    parameters = cp.deepcopy(params)
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    # (≈ 2 lines of code)
    for l in range(L):
        # parameters["W" + str(l+1)] = ...
        # parameters["b" + str(l+1)] = ...
        # YOUR CODE STARTS HERE
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        )
        # YOUR CODE ENDS HERE
    return parameters


def L_layer_model(
    X,
    Y,
    hidden_layer_number,
    layer_dims,
    learning_rate=0.0075,
    num_iterations=3000,
    print_cost=False,
):
    np.random.seed(2)
    costs = []
    n_x, n_y = get_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_y, hidden_layer_number, layer_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def predict(X, Y, parameters):

    m = X.shape[1]
    Y_predict = np.zeros((10, m))
    A_final, cach = L_model_forward(X, parameters)
    A_final = A_final.T

    for i in range(A_final.shape[0]):
        maxim_linie = max(A_final[i])
        for j in range(A_final.shape[1]):
            if A_final[i][j] == maxim_linie:
                Y_predict[j][i] = 1.0
            else:
                Y_predict[j][i] = 0.0

    print("Accuracy: {} %".format(100 - np.mean(np.abs(Y_predict - Y)) * 100))
    return Y_predict
