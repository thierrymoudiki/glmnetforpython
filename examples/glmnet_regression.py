import numpy as np
import os
import sys
import glmnetforpython as glmnet
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time


datasets = [load_diabetes, fetch_california_housing]

for dataset in datasets:

    print(f"\n\n dataset: {dataset.__name__} -------------------")

    X, y = dataset(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    regr = glmnet.GLMNet()

    print(regr.get_params())

    start = time()
    regr.fit(X_train, y_train)
    print(f"elapsed: {time() - start}")

    regr.print()

    print(regr.predict(X_test, s=0.1))

    print(regr.predict(X_test, s=np.asarray([0.1, 0.5])))

    print(regr.predict(X_test, s=0.5))

    start = time()
    res_cvglmnet = regr.cvglmnet(X_train, y_train)
    print(f"elapsed: {time() - start}")

    print("\n best lambda: ", res_cvglmnet.lambda_min)
    print("\n best lambda std. dev: ", res_cvglmnet.lambda_1se)
    print("\n best coef: ", res_cvglmnet.best_coef)
    print("\n best GLMNet: ", res_cvglmnet.cvfit)
