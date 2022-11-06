import numpy as np
import random as rnd
from mlxtend.data import loadlocal_mnist

#region MNIST 60k x 784
X, y = loadlocal_mnist("C:\\Users\\vlada\\Desktop\\Datasets\\MNIST-digits\\train-images.idx3-ubyte",
                       "C:\\Users\\vlada\\Desktop\\Datasets\\MNIST-digits\\train-labels.idx1-ubyte")

X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

X = X[:1000]
y = y[:1000]

X = (X - np.min(X)) / (np.max(X) - np.min(X))
y = y.reshape(-1, 1)
#endregion

#region manji ds
d1 = np.arange(0, 1, 0.1).reshape(1, 10)
d2 = np.arange(2, 3, 0.1).reshape(1, 10)

v = np.random.uniform(-100, 100, (1000, 10))*1e-2

X2 = np.zeros((1000, 10))
X2[:500] = v[:500] + d1
X2[500:] = v[500:] + d2

y2 = np.zeros((1000, 1))
y2[:500] = 0
y2[500:] = 1

idx = rnd.sample( list(range(X.shape[0])), X.shape[0] )
X2 = X2[idx]
y2 = y2[idx]
#endregion