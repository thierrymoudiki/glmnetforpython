import nnetsauce as ns
import mlsauce as ms
import numpy as np
import glmnetforpython as glmnet
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from time import time


datasets = [load_iris, load_breast_cancer, load_wine]

for dataset in datasets:

    print(f"\n\n dataset: {dataset.__name__} -------------------")

    X, y = dataset(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = glmnet.GLMNet(family="multinomial")

    print(clf.get_params())

    start = time()
    clf.fit(X_train, y_train)
    print(f"elapsed: {time() - start}")

    #clf.print()
    #print(clf.score(X_test, y_test))
    preds = clf.predict(X_test, ptype="class")
    print(preds)

    print("accuracy: ", np.mean(preds == y_test))

