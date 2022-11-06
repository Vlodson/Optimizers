import numpy as np
import matplotlib.pyplot as plt

import ds
from utils import *
from batching import *

#region ds
X = ds.X[:900]
y = ds.y[:900]
tX = ds.X[900:]
ty = ds.y[900:]
#endregion

#region w i b
w1 = np.random.uniform(-10, 10, (X.shape[1], 256))*1e-1
b1 = np.random.uniform(-10, 10, (1, 256))*1e-1

w2 = np.random.uniform(-10, 10, (256, 1))*1e-1
b2 = np.random.uniform(-10, 10, (1,1))*1e-1

Gw1 = np.zeros_like(w1).ravel()
Gb1 = np.zeros_like(b1).ravel()
Gw2 = np.zeros_like(w2).ravel()
Gb2 = np.zeros_like(b2).ravel()

prevdw1 = np.zeros_like(w1)
prevdb1 = np.zeros_like(b1)
prevdw2 = np.zeros_like(w2)
prevdb2 = np.zeros_like(b2)
#endregion

#region hiperparam
epochs = int( 1e1 )
eta = 1e-2
eps = 1e-1 # MORA BITI MALO, NE SLUSAJ ZOHAREE

bs = 50

L = []
bL = []
#endregion

#region ucenje
for iter in range(1, epochs+1):
    if iter % (epochs / 10) == 0:
        print("Iteracija {} od {}".format(iter, epochs))
    
    Xbatches, ybatches = batching(X, y, bs)
    for i in range(len(Xbatches)):
        
        #region ff
        z1 = Xbatches[i] @ w1 + b1
        a1 = tanh(z1)

        z2 = a1 @ w2 + b2
        yh = sigmoid(z2)

        bL.append( MSLoss(ybatches[i], yh) )
        #endregion

        #region bp
        # grad
        dyh = yh - ybatches[i]
        dz2 = dsigmoid(z2) * dyh
        dw2 = np.transpose(a1) @ dz2
        db2 = np.sum(dz2, axis = 0)

        da1 = dz2 @ np.transpose(w2)
        dz1 = dtanh(z1) * da1
        dw1 = np.transpose(Xbatches[i]) @ dz1
        db1 = np.sum(dz1, axis = 0)

        # G += grad**2
        Gw1 += prevdw1.ravel()**2
        Gb1 += prevdb1.ravel()**2

        Gw2 += prevdw2.ravel()**2
        Gb2 += prevdb2.ravel()**2

        # update
        thisdw1 = (eta / (Gw1**0.5 + eps) * dw1.ravel()).reshape(w1.shape)
        thisdb1 = (eta / (Gb1**0.5 + eps) * db1.ravel()).reshape(b1.shape)

        thisdw2 = (eta / (Gw2**0.5 + eps) * dw2.ravel()).reshape(w2.shape)
        thisdb2 = (eta / (Gb2**0.5 + eps) * db2.ravel()).reshape(b2.shape)

        w2 -= thisdw2
        b2 -= thisdb2

        w1 -= thisdw1
        b1 -= thisdb1

        # prev = this
        prevdw1 = thisdw1
        prevdb1 = thisdb1

        prevdw2 = thisdw2
        prevdb2 = thisdb2
        #endregion

    L.append(np.sum(bL))
    bL = []

#endregion

#region test
tz1 = tX @ w1 + b1
ta1 = tanh(tz1)

tz2 = ta1 @ w2 + b2
tyh = sigmoid(tz2)

tacc = np.where(tyh > 0.5, 1, 0)
acc = np.sum(tacc == ty) / ty.shape[0] * 100
#endregion

fig, ax = plt.subplots(2,1)
ax[0].plot(L)
ax[1].plot(ty, 'ro', tyh, 'bo')
plt.show()