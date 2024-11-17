import nnetsauce as ns
import mlsauce as ms
import numpy as np
import os
import sys
import glmnetforpython as glmnetpy
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from time import time


datasets = [load_iris, load_breast_cancer, load_wine]

for dataset in datasets:

    print(f"\n\n dataset: {dataset.__name__} -------------------")

    X, y = dataset(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    regr = glmnetpy.GLMNet(family="mgaussian")

    print(regr.get_params())

    #clf = ns.MultitaskClassifier(regr)

    #start = time()
    #clf.fit(X_train, y_train)
    #print(f"elapsed: {time() - start}")

    #regr.print()

    #print(clf.score(X_test, y_test))

    #clf = ns.SimpleMultitaskClassifier(regr)

    #start = time()
    #clf.fit(X_train, y_train)
    #print(f"elapsed: {time() - start}")

    #regr.print()

    #print(clf.score(X_test, y_test))

    clf = ms.GenericBoostingClassifier(regr)

    start = time()
    clf.fit(X_train, y_train)
    print(f"elapsed: {time() - start}")

    regr.print()

    print(clf.score(X_test, y_test))
