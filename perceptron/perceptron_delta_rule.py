import numpy as np
from Activation_Functions.Activation_function import step_function


class Perceptron_delta_rule:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights=np.full(n_features,0.1)
        self.bias=0.1

        for epoch in range(self.n_iterations):
            sum_sq_error = 0.0

            for i , x_samples in enumerate(X):

                y_in=np.dot(x_samples,self.weights)+self.bias

                #delta rule 
                error= y[i]-y_in

                self.weights = self.weights + self.learning_rate * error * x_samples
                self.bias = self.bias + self.learning_rate * error

                sum_sq_error+=error**2
            
            mse=sum_sq_error/n_samples
            # print(f"> Epoch {epoch+1}, MSE: {mse:.4f}")

            if epoch % 10 == 0 or epoch == self.n_iterations - 1:
                predictions = []
                for x_sample in X:
                    y_pred_in = np.dot(x_sample, self.weights) + self.bias
                    # For delta rule (regression to 0/1), classify with 0.5 threshold
                    predictions.append(1 if y_pred_in >= 0.5 else 0)
                
                predictions = np.array(predictions, dtype=int)
                if np.array_equal(predictions, y):
                    print(f"Perfect classification achieved at epoch {epoch+1}")
                    break
        
        print("\n--- Final weights and bias ---")
        print("Weights:", self.weights)
        print("Bias:", self.bias)

    
    def predict(self , x):

        y_in=np.dot(x, self.weights)+self.bias
        # Return class labels using 0.5 threshold
        return (y_in >= 0.5).astype(int)
        # return step_function(y_in)