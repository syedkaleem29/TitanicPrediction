import numpy as np
import matplotlib.pyplot as plt


def nn_model(X, Y, layers_dim, learning_rate=0.05, lambd = 0,num_itr=2500, print_cost=False):

    parameters = initialize_params(layers_dim)
    costs = []

    for i in range(0, num_itr):

        # Forward Propagation
        AL, caches = l_forward_propagation(X, parameters)
        assert (AL.shape == Y.shape)
        assert_parameters(parameters, layers_dim)
        # cost calculation
        cost = compute_cost(AL, Y, parameters, lambd)

        # Backward propagation
        grads = l_backward_propagation(AL, Y, caches, lambd)

        if i < 5:
            gradient_check_n(parameters, grads, X, Y, lambd)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i%100 ==0 and i!=0:
            print("The cost after " + str(i) + " iteration is " + str(cost))
            costs.append(cost)
            plt.plot(np.squeeze(costs))

    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def initialize_params(layers):
    parameters = dict()
    for i in range(1, layers.size):
        parameters['W'+str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.1
        parameters['b'+str(i)] = np.zeros((layers[i], 1))
    return parameters


def l_forward_propagation(X, parameters):

    caches = list()
    A = X
    L = len(parameters) // 2
    for l in range(1, L):

        A_prev = A
        A, cache = linear_forward_propagation(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_forward_propagation(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def linear_forward_propagation(A_prev, W, b, activation="sigmoid"):

    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)

    return Z, linear_cache


def relu(Z):

    bArr = np.array(Z > 0).astype(int)
    A = Z * bArr
    activation_cache = Z
    return A, activation_cache


def sigmoid(Z):

    A = 1/(1 + np.exp(-Z))
    activation_cache = Z
    return A, activation_cache


def compute_cost(AL, Y, parameters, lambd):

    m = Y.shape[1]
    L = len(parameters) // 2
    sum = 0
    for l in range(1, L + 1):
        sum = sum + np.sum(np.square(parameters["W" + str(l)]))
    reg_cost = lambd / (2*m) * sum
    diff = ((Y * np.log(AL)) + ((1-Y) * np.log(1-AL)))
    cost = -1/m * np.sum(diff)
    cost = cost + reg_cost
    return cost


def l_backward_propagation(AL, Y, caches, lambd):

    grads = dict()
    dA = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    L = len(caches)
    activation = "sigmoid"
    grads["dA"+str(L)] = dA
    for l in range(L, 0, -1):
        cache = caches.__getitem__(l-1)
        dA, dWL, dbL = linear_backward_propagation(grads["dA"+str(l)], cache, activation, lambd)
        grads["dA" + str(l-1)] = dA
        grads["dW" + str(l)] = dWL
        grads["db" + str(l)] = dbL
        activation = "relu"
    return grads


def linear_backward_propagation(dAL, cache, activation, lambd):

    linear_cache, activation_cache = cache
    Aprev, WL, bL = linear_cache
    ZL = activation_cache

    if activation == "sigmoid":
        derv = derive_sigmoid(ZL)
        dZL = dAL * derv
    else:
        derv = derive_relu(ZL)
        dZL = dAL * derv

    dAprev = np.dot(WL.T, dZL)
    m = Aprev.shape[1]
    dWL = 1/m * np.dot(dZL, Aprev.T) + (lambd/m)*WL
    dbL = 1/m * np.sum(dZL, axis=1, keepdims=True)

    assert (dAprev.shape == Aprev.shape)
    assert (dWL.shape == WL.shape)
    assert (dbL.shape == bL.shape)

    return dAprev, dWL, dbL


def derive_sigmoid(Z):
    A, _ = sigmoid(Z)
    return A * (1-A)


def derive_relu(Z):
    return np.array(Z > 0).astype(int)


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters)//2

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


def predict(X, parameters):

    predictions, _ = l_forward_propagation(X, parameters)
    Y_predict = np.array(predictions > 0.5).astype(int)
    return Y_predict


def assert_parameters(parameters, layers):

    L = len(layers)
    for l in range(1, L):
        assert (parameters["W" + str(l)].shape == (layers[l], layers[l-1]))
        assert (parameters["b" + str(l)].shape == (layers[l], 1))
    return


def eval_model(Y, predictions):

    tp = np.logical_and(Y, predictions).astype(int)
    tp = np.sum(tp)
    total_positives = np.sum(Y)
    total_predicted_positives = np.sum(predictions)
    precision = (tp / total_predicted_positives) * 100
    recall = (tp / total_positives) * 100
    fscore = (2*precision*recall) / (precision + recall)
    print("The precision is " + str(precision))
    print("The recall is " + str(recall))
    print("The f score is " + str(fscore))


def gradient_check_n(parameters, gradients, X, Y, lambd, epsilon=1e-7):
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[1]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[0][i] = thetaplus[0][i] + epsilon  # Step 2
        A_plus, _ = l_forward_propagation(X, vector_to_dictionary(thetaplus))  # Step 3
        ### END CODE HERE ###
        J_plus[i] = compute_cost(A_plus, Y, parameters, lambd)
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[0][i] = thetaminus[0][i] - epsilon  # Step 2
        A_minus, _ = l_forward_propagation(X, vector_to_dictionary(thetaminus))  # Step 3
        J_minus[i] = compute_cost(A_minus, Y, parameters, lambd)
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad.T - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad.T) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


def dictionary_to_vector(parameters):

    L = len(parameters)//2
    m = parameters["W1"].reshape(-1,)
    a = parameters["b1"].reshape(-1,)
    for l in range(2,L+1):
        mat = parameters["W"+str(l)].reshape(-1,)
        m = np.concatenate((m, mat))
        arr = parameters["b"+str(l)].reshape(-1,)
        a = np.concatenate((a,arr))

    vec = np.concatenate((m, a))
    vec = vec.reshape(1, -1)
    return vec, parameters


def gradients_to_vector(gradients):
    L = len(gradients) // 3
    m = gradients["dW1"].reshape(-1, )
    a = gradients["db1"].reshape(-1, )
    for l in range(2, L + 1):
        mat = gradients["dW" + str(l)].reshape(-1, )
        m = np.concatenate((m, mat))
        arr = gradients["db" + str(l)].reshape(-1, )
        a = np.concatenate((a, arr))

    vec = np.concatenate((m, a))
    vec = vec.reshape(1, -1)
    return vec


def vector_to_dictionary(vector):

    params = dict()
    vector = vector.reshape(1, -1)
    w = vector[:,0:35]
    w = w.reshape((5, 7))
    params["W1"] = w

    w = vector[:,35:50]
    w = w.reshape((3, 5))
    params["W2"] = w

    w = vector[:,50:53]
    w = w.reshape(1, 3)
    params["W3"] = w

    b = vector[:,53:58]
    b = b.reshape(5,1)
    params["b1"] = b

    b = vector[:,58:61]
    b = b.reshape(3, 1)
    params["b2"] = b

    b = vector[:,61:62]
    b = b.reshape(1, 1)
    params["b3"] = b

    return params