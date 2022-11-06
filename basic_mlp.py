#
# @author = Natalia Oviedo Acosta
# 
#

import data
import math

# Varibles needed
# Input values
x = [1., -1.]
# Weights
w = [[1.,1.,1.],[-1.,-1.,-1.]]
# Bias
b = [0,0]
# Second layer weights
v = [1.,-1.,-1.]
# Loss total?
l = [0.,0.]
#Target
t = [1,0]

#Sigmoid function for non-linearity
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def forward_pass():
    #Initialize layer k in zero
    k = [0.,0.,0.]
    #First layer for inputs and weights to create layer k
    for j in range(3):
        for i in range (2):
            k[j] += w[i][j] * x [i]
        k[j] += b[0]
    #Initialize layer h in zero
    h = [0.0,0.0,0.0]
    # Second layer for h (non-linearity)
    for i in range(3): 
        h[i] = sigmoid(k[i])

    #Initialize output y
    output= [0., 0.]
    for j in range(2):
        for i in range(3):
            output[j] += h[i] * v[i]
        output[j] += b[1]

    # Softmax function
    y = [0.,0.]
    output_total = 0.
    for i in range(2):
        output_total += math.exp(output[i])

    for i in range(2):
        y[i] = math.exp(output[i])/output_total

    loss = 0.

    # Cross-entropy loss function
    for i in range(2):
        l[i] = - (t[i] * math.log(y[i]) + ((1 - t[i])* math.log(1 - y[i])))   
        loss += l[i]
    return y, h, v

def backward_pass(y, h, v):
    # Backward pass
    # dl/dy includes softmax and loss function derivate
    dl_dy = [0.,0.]
    for i in range(2):
        dl = - (t[i]/y[i]) + ((1 - t[i])/ (1 - y[i]))   # the error
        dsoftmax = y[i] * (1 - y[i])
        dl_dy[i] = dl * dsoftmax

    # Derivates for dv and dh
    dl_dv = [[0., 0.], [0., 0.], [0., 0.]] 
    dl_dh = [0., 0., 0.] # Derivate h
    for i in range(3):
        for j in range(2):
            dl_dv[i][j] = dl_dy[j] * h[i]
            dl_dh[i] +=  dl_dy[j] * v[i]

    dl_dc = dl_dy
    dl_dk = [0.0, 0.0, 0.0] # Derivate k
    dl_dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # Derivate w
    dl_db = [0.0, 0.0, 0.0]

    #dl/dk derivate 
    for i in range(3):
        dl_dk[i] = h[i] * (1 - h[i]) * dl_dh[i]

    #dl/dw derivate
    for j in range(3):
        for i in range (2):
            dl_dw[i][j] = dl_dk[j] * x[i] 
        dl_db[j] = dl_dk[j]


if __name__ == "__main__":
    y, h, v = forward_pass()
    backward_pass(y, h, v)
    (xtrain, ytrain), (xval, yval), num_cls = data.load_synth()