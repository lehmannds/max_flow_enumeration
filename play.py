import numpy as np
from numpy.linalg import pinv


def perceptron(X, y):
    
    w = np.array(X.shape[1])
    idx = 0
    while True:
        found = False
        for i in np.arange(X.shape[0]):
            idx += 1
            val = X[i] @ w
            if val * y <= 0:
                w += X[i] * (-val / np.norm(X[i]) + 0.5) 
                found = True

        if not found:
            break
    print("runs", idx)
    return w

X = np.random.uniform(-1,1,(3,3))
w = np.random.uniform(-1,1,3)
y = (X @ w) 
y = np.sign(y)

w_pred = perceptron(X, y)

print({"w": w, "w_pred": w_pred})