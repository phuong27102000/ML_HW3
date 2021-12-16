import numpy as np


def sigmoid(x):
    """sigmoid
    TODO:
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = 1 / ( 1 + np.exp(-x) )
    return y
    # ______________________________
    # to here
    # ------------------------------
    # return None


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = a * (1 - a)
    return y
    # ______________________________
    # to here
    # ------------------------------


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = np.maximum(0,x)
    return y
    # ______________________________
    # to here
    # ------------------------------


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    # grad = None
    # return grad
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = np.greater(a,0)
    return y
    # ______________________________
    # to here
    # ------------------------------


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = np.tanh(x)
    return y
    # ______________________________
    # to here
    # ------------------------------


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = 1 - np.power(a,2)
    return y
    # ______________________________
    # to here
    # ------------------------------


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """

    # output = None
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    y = np.exp(x)
    y = y / np.sum(y, axis=1, keepdims=True)
    return y
    # ______________________________
    # to here
    # ------------------------------


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """

    # output = None
    # return None
    # ______________________________
    # Code by Phuong from here
    # ------------------------------
    c = np.max(x)
    y = np.exp(x-c)
    y = y / np.sum(y, axis=1, keepdims=True)
    return y
    # ______________________________
    # to here
    # ------------------------------
