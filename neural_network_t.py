#
# @author = Natalia Oviedo Acosta
# 
#

#import data
import math
import data
import random as r

class neuralnetwork:
    def __init__(self):
        # initialize parameters, including the first and second layer 
        # parameters and biases
        self.weight_1 = [
            [0.9728689562960486, 0.8828173118260882, 0.535647770488933],
            [0.8452517162995787, 0.2935964146679466, 0.8966905560133287]
            ]
        self.bias_1 = [0.,0.,0.]
        
        self.weight_2 = [ # Should weights be the same?
            [0.36794416135090513, 0.7556447692578278],
            [0.7556447692578278, 0.15827788839007184],
            [0.6510292605429753, 0.7005473553542177]
            ]
        self.bias_2 = [0.,0.]
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

    def forward(self, X, Y):
        #Initialize layer k in zero
        k = [0.,0.,0.]
        #First layer for inputs and weights to create layer k
        for j in range(3):
            for i in range (2):
                k[j] += self.weight_1[i][j] * X[i]
            k[j] += self.bias_1[j]

        # Second layer for h (non-linearity)
        for i in range(3): 
            self.hidden_layer[i] = self.__sigmoid(k[i])

        #Initialize output y
        output= [0., 0.]
        for j in range(2):
            for i in range(3):
                output[j] += self.hidden_layer[i] * self.weight_2[i][j]
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
            l[i] = - (Y[i] * math.log(self.y[i]) + ((1 - Y[i])* math.log(1 - self.y[i])))    # the error
            loss += l[i]
        return loss

    def backward(self, X, Y):
        # Backward pass
        # dl/dy includes softmax and loss function derivate
        dl_dy = [0.,0.]
        for i in range(2):
            dl = - (Y[i] / self.y[i]) + ((1 - Y[i])/ (1 - self.y[i]))   # the error
            dsoftmax = self.y[i] * (1 - self.y[i])
            dl_dy[i] = dl * dsoftmax
        # Derivates for dv and dh
        dl_dv = [[0., 0.], [0., 0.], [0., 0.]] 
        dl_dh = [0., 0., 0.] # Derivate h
        for i in range(3):
            for j in range(2):
                dl_dv[i][j] = dl_dy[j] * self.hidden_layer[i]
                dl_dh[i] +=  dl_dy[j] * self.weight_2[i][j]
        
        dl_db2 = dl_dy
        dl_dk = [0.0, 0.0, 0.0] # Derivate k
        dl_dw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # Derivate w
        dl_db = [0.0, 0.0, 0.0]

        #dl/dk derivate 
        for i in range(3):
            dl_dk[i] = self.__sigmoid(self.hidden_layer[i], True) * dl_dh[i]

        #dl/dw derivate
        for j in range(3):
            for i in range (2):
                dl_dw[i][j] = dl_dk[j] * X[i] 
            dl_db[j] = dl_dk[j]
        
        return dl_dw, dl_db, dl_dv, dl_db2
        
    def update_parameters(self, dl_dw, dl_db, dl_dv, dl_db2, alpha):
        for i in range(2):
            for j in range(3):
                self.weight_1[i][j] -= alpha * dl_dw[i][j] 
                self.weight_2[j][i] -= alpha * dl_dv[j][i]
            self.bias_1[i] -= alpha * dl_db[i] 
        for i in range(2):
            self.bias_2[i] -= alpha * dl_db2[i]
        # Should de bias also change here? Or not?

    def one_hot_vector(self, Y):
        vector = [0,0]
        if Y == 1:
            vector[0] = 1
        elif Y == 0:
            vector[1] = 1
        return vector
    
    # gradient_descent
    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            loss = self.forward(X, Y)
            dl_dw, dl_db, dl_dv, dl_db2 = self.backward(X, Y)
            self.update_parameters(dl_dw, dl_db, dl_dv, dl_db2, alpha)
            if i % 10 == 0:
                 print("Loss ",  i, " = ", loss)

(xtrain, ytrain), (xval, yval), num_cls = data.load_synth()
model = neuralnetwork()
Y = model.one_hot_vector(ytrain[0])
model.gradient_descent(xtrain[2,:], Y, 0.5, 100)