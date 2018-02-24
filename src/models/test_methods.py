import numpy as np
from models import nnutil


def test_initialize_params():
    layers_dim = np.array([3, 2])
    print(nnutil.initialize_params(layers_dim))
    return


def test_linear_forward():
    A = np.array([1, 2, 3]).reshape((3, 1))
    W = np.random.randn(1, 3) * 0.01
    b = np.zeros((1, 1))
    A, _ = nnutil.linear_forward(A, W, b)
    print(A)


def test_sigmoid():
    Z = np.array([-100, 0, 100])
    print(nnutil.sigmoid(Z))


def test_relu():
    Z = np.array(range(-4, 4, 1))
    print(nnutil.relu(Z))


def test_forward_propogation_sigmoid():
    Aprev = np.array([1, 2, 3]).reshape((3, 1))
    W = np.random.randn(1, 3) * 0.01
    b = np.zeros((1, 1))
    A, _ = nnutil.linear_forward_propagation(Aprev, W, b)
    print(A)


def test_forward_propogation_relu():
    Aprev = np.array([1, 2, 3]).reshape((3, 1))
    W = np.random.randn(1, 3) * 0.01
    b = np.zeros((1, 1))
    A, _ = nnutil.linear_forward_propagation(Aprev, W, b, activation="relu")
    print(A)


def test_sigmoid_derivative():
    A = np.array([1,2,3])
    B = np.array([2,3,4])
    print(A.shape)
    print(B.shape)
    C = np.concatenate((A,B))
    print(C.shape)





np.random.seed(1)
# test_initialize_params()
# test_linear_forward()
# test_sigmoid()
# test_relu()
# test_forward_propogation_sigmoid()
#test_forward_propogation_relu()
test_sigmoid_derivative()
