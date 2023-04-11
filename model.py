from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
# the parameters used to classify tumors as either malignant or benign
print(data.feature_names)
# two classes malignant or benign 
print(data.target_names)

# split data, 20 percent for testing, 80 percent for training
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size = 0.2)

# choose a k value (3) that is not divisible by number of classes (2)
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x_train, y_train)

# can predict if tumor is malignant or benign with ~90%+ accuracy
print(model.score(x_test, y_test))

# if wanted to predict whether a new tumor is malignant:
# model.predict(np.array([  data for parameters go here ]))


