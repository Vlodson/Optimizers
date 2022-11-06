import numpy as np
import random as rnd
from math import ceil

def batching(X: np.ndarray, y: np.ndarray, bs: int):
    """Vraca batcheve X-a i y-a za epohu, proci kroz samo duzinu jedne od tih lista"""

    idx = rnd.sample( list(range(X.shape[0])), X.shape[0] )
    Xbatches = []
    ybatches = []

    for i in range( ceil(len(idx) / bs) ):
        try:
            batchidx = idx[i*bs:(i+1)*bs]
        except IndexError:
            batchidx = idx[i*bs:]

        Xbatches.append(X[batchidx])
        ybatches.append(y[batchidx])

    return Xbatches, ybatches