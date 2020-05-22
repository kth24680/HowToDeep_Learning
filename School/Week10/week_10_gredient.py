#!/usr/bin/env python3
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 2, 3, 2, 2
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H,D_out)

print("x\n",x)
print("y\n",y)
print("w1\n",w1)
print("w2\n",w2,"\n")

for t in range(1000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    print("h\n",h)

    y_pred = h.dot(w2)
    print("y[출력갑]\n",y_pred)

    loss = np.square(y_pred - y).sum()
    print(t,"번째 loss[손실값 합]...", loss,"\n");

    grad_y_pred = 2.0 * (y_pred - y)

    grad_w2 = h.T.dot(grad_y_pred)

    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1-h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2