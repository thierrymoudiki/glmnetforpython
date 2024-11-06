import os
import sys
import glmnetforpython as glmnetpy
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regr = glmnetpy.GLMNet()

print(regr.get_params())

regr.fit(X_train, y_train)

print(regr.predict(X_test))
