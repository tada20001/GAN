from __future__ import print_function, division
from builtins import range, input

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

matplotlib.rcParams['axes.unicode_minus'] = False 

def softplus(x):
    # log1p(x) == log(1+x)
    return np.log1p(np.exp(x))


# we're going to make a neural network
# with the layer sizes (4, 3, 2)
# like a toy version of a decoder

W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2*2)

# why 2 * 2?
# we need 2 components for the mean,
# and 2 components for the standard deviation!

# ignore bias terms for simplicity.

def forward(x, W1, W2):
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)  # no activation!
    mean = output[:2]  # 아웃풋 중 2개만 선택
    stddev = softplus(output[2:])
    return mean, stddev

# make a random input
x = np.random.randn(4)

# get the parameters of the Gaussian
mean, stddev = forward(x, W1, W2)
print("mean:", mean)
print("stddev:", stddev)


# draw samples
samples = mvn.rvs(mean=mean, cov=stddev**2, size=10000)
print("sample's size:", samples.shape)
# plot the samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()