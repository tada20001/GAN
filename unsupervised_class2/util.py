from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

def relu(x):
    return x * (x > 0)

def error_rate(p, t):
    return np.mean(p != t)  # 동일하지 않을 경우 평균값 측정

def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    train = pd.read_csv("./large_files/train.csv").values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

def init_weights(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)



from glob import glob
from tqdm import tqdm
from sklearn.utils import shuffle

def get_mnist(limit=None):
    if not os.path.exists('./large_files'):
        print("You must create a folder called large_files adjacent to the class folder first")
    if not os.path.exists('./large_files/train.csv'):
        print("Looks like you haven't download the data or it's not in the right spot.")
        print("Please get train.csv from https://www.kaggle.com/c/digit-recognizer")
        print("and place it in the large_files folder.")

    print("Reading in and transforming data ...")
    df = pd.read_csv('./large_files/train.csv')
    data = df.values
    # np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    X, Y = shuffle(X, Y)
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y