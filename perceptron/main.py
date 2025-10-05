import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from .perceptron import Perceptron
from .perceptron_delta_rule import Perceptron_delta_rule
from sklearn.datasets import load_breast_cancer

#loading  Iris dataset

# iris=datasets.load_iris()
# # using only  petal length and width, and find Setosa vs. Not-Setosa
# X=iris.data[:100,[2,3]]
# y=iris.target[:100]

#loading breast cancer dataset 
cancer=load_breast_cancer()
X=cancer.data[:, [0, 1]]
y=cancer.target

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=123)
print("test output data", y_test)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# p=Perceptron_delta_rule(learning_rate=0.1,n_iterations=1000)
p=Perceptron(learning_rate=0.1,n_iter=1000)
p.fit(X_train,y_train)

predictions = p.predict(X_test)
print(predictions)

accuracy= np.sum(predictions==y_test)/len(y_test)
print(f"Accuracy: {accuracy*100}%")