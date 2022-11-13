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
        # Linear layer k
        # k = w1(v) * x + b1
        for j in range(3):
            for i in range (2):
                k[j] += self.weight_1[i][j] * self.input[i]
            k[j] += self.bias_1[j]
        # Second layer for h (sigmoid function)
        # h = sigmoid(k)
        for i in range(3): 
            self.hidden_layer[i] = self.__sigmoid(k[i])
        # Output layer
        # y = w2(v) * h + b2
        for j in range(2):
            for i in range(3):
                self.y[j] += self.hidden_layer[i] * self.weight_2[i]
            self.y[j] += self.bias_2[j]
        # Softmax function on y
        total = 0.
        for i in range(2):
            total += math.exp(self.y[i])
        # y = softmax(y)
        for i in range(2):
            self.y[i] = self.__softmax(total, self.y[i])
        # Cross-entropy loss function
        l = [0.,0.]
        loss = 0.
        for i in range(2):
            l[i] = - math.log(self.y[i])
            loss += l[i]
        print(loss)

    def backward(self):
        # dl/dy includes softmax and loss function derivate
        dl_dy = [0.,0.]
        for i in range(2):
            # Binary cross-entropy loss function derivative
            dl = - (self.target_class[i]/self.y[i]) + ((1 - self.target_class[i])/ (1 - self.y[i])) 
            # Softmax derivative when i = j
            dsoftmax = self.y[i] * (1 - self.y[i])
            dl_dy[i] = dl * dsoftmax 
            #dl_dy[i] = self.y[i] - self.target_class[i] 
        # Initialize dl_dv and dl_dh in zero
        dl_dv = [[0., 0.], [0., 0.], [0., 0.]] 
        dl_dh = [0., 0., 0.] 
        for i in range(3):
            for j in range(2):
                # dl/dh = dl/dy * dy/dh
                # do/dh = w2
                dl_dh[i] +=  dl_dy[j] * self.weight_2[i]
                # dl/dv = dl/dy * dy/dv
                # do/dv = h
                dl_dv[i][j] = dl_dy[j] * self.hidden_layer[i]
        # dl_dc = derivate of b2
        # dl/dc = dl/dy * dy/dc 
        # do/dc = 1 
        dl_dc = dl_dy
        dl_dk = [0.0, 0.0, 0.0] # Derivate k
        dl_dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # Derivate w
        dl_db = [0.0, 0.0, 0.0] # Derivate of b1
        # dl/dk = dl/dy * dy/dh * dh/dk
        # dl/dk = dl/dh * dh/dk
        # Sigmoid derivative
        for i in range(3):
            dl_dk[i] = self.__sigmoid(self.hidden_layer[i], True) * dl_dh[i]
        # dl/dw = dl/dy * dy/dh * dh/dk * dk/dw
        # dl/dw = dl/dk * dk/dw
        # dk/dw = x(input)
        for j in range(3):
            for i in range (2):
                dl_dw[i][j] = dl_dk[j] * self.input[i] 
            # dl/db = derivate of b1
            # dl/db = dl/dy * dy/dh * dh/dk * dk/db
            # dl/db = dl/dk * dk/db
            # dk/db = 1 
            dl_db[j] = dl_dk[j]
        
        print(dl_dc)

model = neuralnetwork()
model.forward()
model.backward()
