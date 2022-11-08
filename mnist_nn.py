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
    bias_1 = np.random.rand(1, 300) - 0.5
    weight_2 = np.random.rand(300,10) - 0.5
    bias_2 = np.random.rand(10, 1) - 0.5

    return weight_1, bias_1, weight_2, bias_2

# sigmoid function for non-linearity
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# softmax function
def softmax(x):
    s = np.exp(x) / sum(np.exp(x))
    return s

# cross entropy loss
def cross_entropy_loss(v, y):
    return -np.log(v[y])

def forward(weight_1, bias_1, weight_2, bias_2, input, Y):
    #First layer for inputs and weights to create layer k
    # k = w * x + b1
    K1 = weight_1.dot(input) + bias_1
    # Second layer for h (non-linearity)
    # h = sigmoid(k)
    H1 = sigmoid(K1) #A1
    # o = v * h + b2
    O1 = H1.dot(weight_2) + bias_2 #Z2
    # Softmax function
    Y1 = softmax(O1) #A2
    # Cross-entropy loss
    Error = cross_entropy_loss(Y1, Y) # Y has 60000 values 
    # Y1 is a 10 x 10 value

    return K1, H1, O1, Y1, Error # Z1, A1, Z2, A2

def backward(K1, H1, O1, Y1, weight_2, X, Y):
    # differentiation of all the parameters
    # update the gradients and store in the dictionary

    dl_do = Y1 - Y # dZ2 = A2 - one_hot_Y

    dl_dv = dl_do.dot(H1.T) #  dW2 = 1 / m * dZ2.dot(A1.T)
    dl_db2 = np.sum(dl_do)   # db2 = 1 / m * np.sum(dZ2)
    #dA1 = W2.T.dot(dZ2)
    dl_dh = weight_2.T.dot(dl_do)
    #dZ1 = dA1 * ReLU_deriv(Z1)
    dl_dk = dl_dh * sigmoid(K1) * (1 - sigmoid(K1))

    dl_dw = dl_dk.dot(X.T)  # dW1 = 1 / m * dZ1.dot(X.T)
    dl_db1 = np.sum(dl_dk) # db1 = 1 / m * np.sum(dZ1)

    return dl_dw, dl_db1, dl_dv, dl_db2

# Loss function?
def update_params(weight_1, bias_1, weight_2, bias_2, dl_dw, dl_db1, dl_dv, dl_db2, alpha):
    weight_1 = weight_1 - alpha * dl_dw
    bias_1 = bias_1 - alpha * dl_db1    
    weight_2 = weight_2 - alpha * dl_dv  
    bias_2 = bias_2 - alpha * dl_db2    
    return weight_1, bias_1, weight_2, bias_2

# gradient_descent  
def gradient_descent(X, Y):
    W1, b1, W2, b2 = init_parameters()
    K1, H1, O1, Y1, error = forward(W1, b1, W2, b2, X, Y)
    return K1, H1, O1, Y1, error


x_train, t_train, x_test, t_test = mnist.load()
X = x_train[0,:]
Y = t_train
K1, H1, O1, Y1, error = gradient_descent(X, Y)
print(Y1)
'''

input = [1., -1.]
# target class
target_class = [1.,0.]
# output value with softmax
y = [0.,0.]
#Initialize layer h in zero (hidden layer)
hidden_layer = [0.,0.,0.]
output_size = 10
    

'''
#def __lossfunction(self):
    # Cross-entropy loss function
#    l = [0.,0.]
#    for i in range(2):
#        l[i] = - (self.target_class[i] * np.log(self.y[i]) + ((1 - self.target_class[i])* np.log(1 - self.y[i])))    # the error
#        loss += l[i]

#def gradient_descent(X, Y, alpha, iterations):
    #W1, b1, W2, b2 = init_parameters()
    #for i in range(iterations):
    #    Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
    #    dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
    #    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    #    if i % 10 == 0:
    #        print("Iteration: ", i)
            #predictions = get_predictions(A2)
            #print(get_accuracy(predictions, Y))
    #return W1, b1, W2, b2 
