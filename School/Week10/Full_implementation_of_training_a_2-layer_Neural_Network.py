import numpy as np
from numpy.random import randn

#D_in은 Input Layer, H는 Hidden Layer, D_out은 출력.
N, D_in, H, D_out = 64, 1000, 100, 10
#입력으로는 64개의 데이터가 있고 1000개의 요소를 가짐.
#출력은 데이터 64개 출력은 10개
#정규분포로 난수를 발생 : randn
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H,D_out)

for t in range(2000):
    h = 1/ (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    #그레디언트를 구한 것.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1-h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

