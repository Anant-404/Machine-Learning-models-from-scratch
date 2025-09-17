import numpy as np
from Activation_Functions.Activation_function import step_function #foldername.filename import functionname

class Perceptron:
    def __init__(self,learning_rate=0.1,n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights=None
        self.bias=None
    
    def fit(self, x , y):


        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        #traning loop of the model 
        for _ in range(self.n_iter):
            for i , x_samples in enumerate(x):

                y_in = np.dot(x_samples,self.weights)+self.bias

                y_out = step_function(y_in)

                error= y[i]-y_out

                self.weights=self.weights+self.learning_rate*error*x_samples
                self.bias+=self.learning_rate*error

    
    def predict(self,x):

        y_in = np.dot(x,self.weights)+self.bias

        y_out= step_function(y_in)

        return y_out