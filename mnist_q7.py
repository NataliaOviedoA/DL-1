#
# @author = Natalia Oviedo Acosta
# 
#

#import data
import numpy as np
import mnist
import matplotlib.pyplot as plt


def init_parameters():
    # Using values from -0.5 to 0.5
    weight_1 = np.random.rand(300, 784) - 0.5
    bias_1 = np.random.rand(300, 1) - 0.5
    weight_2 = np.random.rand(10,300) - 0.5
    bias_2 = np.random.rand(10, 1) - 0.5

    return weight_1, bias_1, weight_2, bias_2

# sigmoid function for non-linearity
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# softmax function
def softmax(x):
    total = np.sum(np.exp(x))
    s = np.exp(x) / total
    return s
def softmax_d(x):
    I = np.eye(x.shape[0])
    
    return softmax(x) * (I - softmax(x).T)

# cross entropy loss
def cross_entropy_loss(v, y):
    loss = - np.sum(v * np.log(y))
    return loss

def cross_entropy_loss_d(v, y):
    return -v/(y)
    return loss


def forward(weight_1, bias_1, weight_2, bias_2, input, Y):
    #First layer for inputs and weights to create layer k
    # k = w * x + b1
    K1 = weight_1.dot(input) + bias_1
    # Second layer for h (non-linearity)
    # h = sigmoid(k)
    H1 = sigmoid(K1) #A1
    # o = v * h + b2
    O1 = weight_2.dot(H1) + bias_2 #Z2
    # Softmax function
    Y1 = softmax(O1) #A2
    # Cross-entropy loss
    loss = cross_entropy_loss(Y, Y1)

    return K1, H1, O1, Y1, loss # Z1, A1, Z2, A2

def backward(K1, H1, O1, Y1, weight_2, X, Y):
    # differentiation of all the parameters
    dl_do = np.sum(cross_entropy_loss_d(Y, Y1) * softmax_d(O1), axis = 0).reshape((-1, 1))
    # update the gradients and store in the dictionary
    #dl_do = Y1 - Y # dZ2 = A2 - one_hot_Y
    dl_dv = dl_do.dot(H1.T) #  dW2 = 1 / m * dZ2.dot(A1.T)
    dl_db2 = dl_do  # db2 = 1 / m * np.sum(dZ2)
    #dA1 = W2.T.dot(dZ2)
    dl_dh = weight_2.T.dot(dl_do) # problem with the matrix shapes
    #dZ1 = dA1 * ReLU_deriv(Z1)
    dl_dk = dl_dh * sigmoid(K1) * (1 - sigmoid(K1))

    dl_dw = dl_dk.dot(X.T)  # dW1 = 1 / m * dZ1.dot(X.T)
    dl_db1 = np.sum(dl_dk) # db1 = 1 / m * np.sum(dZ1)
  
    return dl_dw, dl_db1, dl_dv, dl_db2

# Loss function?
def update_params(weight_1, bias_1, weight_2, bias_2, dl_dw, dl_db1, dl_dv, dl_db2, alpha):
    weight_1 -= alpha * dl_dw
    bias_1 -= alpha * dl_db1    
    weight_2 -= alpha * dl_dv  
    bias_2 -= alpha * dl_db2    
    return weight_1, bias_1, weight_2, bias_2

def one_hot_vector(Y):
    vector = np.zeros((Y.size,10))
    vector[np.arange(Y.size), Y] = 1
    return vector

# gradient_descent  
def gradient_descent(X, Y, alpha, epochs):
    vector = one_hot_vector(Y)
    vector = np.expand_dims(vector[0], axis=1)
    #print(vector)
    W1, b1, W2, b2 = init_parameters()
    for i in range(epochs):
        K1, H1, O1, Y1, loss = forward(W1, b1, W2, b2, X, vector)
        dl_dw, dl_db1, dl_dv, dl_db2 = backward(K1, H1, O1, Y1, W2, X, vector)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dl_dw, dl_db1, dl_dv, dl_db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print(loss)
            print(Y1)

x_train, t_train, x_test, t_test = mnist.load()
trainX = x_train[1, :]

Y = t_train[1]
X = np.expand_dims(trainX, axis=1)
X = X/255
#print(Y)
gradient_descent(X, Y, 0.01, 100)


