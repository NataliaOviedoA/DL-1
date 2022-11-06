#
# @author = Natalia Oviedo Acosta
# 
#

#import data
import math

class neuralnetwork:
    def __init__(self):
        # initialize parameters, including the first and second layer 
        # parameters and biases 
        self.input = [1., -1.]
        self.weight_1 = [[1.,1.,1.],[-1.,-1.,-1.]]
        self.bias_1 = [0.,0.,0.]
        
        self.weight_2 = [1.,-1.,-1.]
        self.bias_2 = [0.,0.]
        # target class
        self.target_class = [1.,0.]
        # output value with softmax
        self.y = [0.,0.]
        #Initialize layer h in zero (hidden layer)
        self.hidden_layer = [0.,0.,0.]

    # sigmoid function for non-linearity
    def __sigmoid(self,x, derivate = False):
        if derivate:
            return x * (1 - x)
        return 1. / (1. + math.exp(-x))
    
    # softmax function
    def __softmax(self,total, x):
        exps = math.exp(x)
        return exps / total

    def forward(self):
        #Initialize layer k in zero
        k = [0.,0.,0.]
        #First layer for inputs and weights to create layer k
        for j in range(3):
            for i in range (2):
                k[j] += self.weight_1[i][j] * self.input[i]
            k[j] += self.bias_1[j]
        # Second layer for h (non-linearity)
        for i in range(3): 
            self.hidden_layer[i] = self.__sigmoid(k[i])

        #Initialize output y
        output= [0., 0.]
        for j in range(2):
            for i in range(3):
                output[j] += self.hidden_layer[i] * self.weight_2[i]
            output[j] += self.bias_2[j]

        # Softmax function
        total = 0.
        for i in range(2):
            total += math.exp(output[i])

        for i in range(2):
            self.y[i] = self.__softmax(total, output[i])
        loss = 0.

        # Cross-entropy loss function
        l = [0.,0.]
        for i in range(2):
            l[i] = - (self.target_class[i] * math.log(self.y[i]) + ((1 - self.target_class[i])* math.log(1 - self.y[i])))    # the error
            loss += l[i]

    def backward(self):
        # Backward pass
        # dl/dy includes softmax and loss function derivate
        dl_dy = [0.,0.]
        for i in range(2):
            dl = - (self.target_class[i]/self.y[i]) + ((1 - self.target_class[i])/ (1 - self.y[i]))   # the error
            dsoftmax = self.y[i] * (1 - self.y[i])
            dl_dy[i] = dl * dsoftmax

        # Derivates for dv and dh
        dl_dv = [[0., 0.], [0., 0.], [0., 0.]] 
        dl_dh = [0., 0., 0.] # Derivate h
        for i in range(3):
            for j in range(2):
                dl_dv[i][j] = dl_dy[j] * self.hidden_layer[i]
                dl_dh[i] +=  dl_dy[j] * self.weight_2[i]
        
        dl_dc = dl_dy
        dl_dk = [0.0, 0.0, 0.0] # Derivate k
        dl_dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # Derivate w
        dl_db = [0.0, 0.0, 0.0]

        #dl/dk derivate 
        for i in range(3):
            dl_dk[i] = self.__sigmoid(self.hidden_layer[i], True) * dl_dh[i]

        #dl/dw derivate
        for j in range(3):
            for i in range (2):
                dl_dw[i][j] = dl_dk[j] * self.input[i] 
            dl_db[j] = dl_dk[j]

model = neuralnetwork()
model.forward()
model.backward()
