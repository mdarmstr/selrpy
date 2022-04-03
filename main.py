import pandas as pd

from selrpy import selrpy
from sklearn import preprocessing

# Just to test and make sure the code is working.

if __name__ == "__main__":
    coffee = pd.read_csv("coffee.csv", sep=",", header=0)
    X = coffee.iloc[:, 1:].to_numpy()
    y = pd.get_dummies(coffee.iloc[:, 0], drop_first=True).to_numpy(dtype="int32", copy=True).reshape(-1)
    Xs = preprocessing.StandardScaler().fit_transform(X)
    selr = selrpy(X, y, 2)
    print(selr)

