import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2
    
    
def invf(y):
    return y**0.5
    
    
def df(x):
    return 2*x
    

x = np.arange(-15, 15, 0.1)
y = f(x)

np.random.seed(0)
nabla_y = 200
lr = 0.1

for _ in range(100):
    nabla_y -= lr*df(nabla_y)
    plt.scatter([invf(nabla_y)], [nabla_y], c="r", marker="o", alpha=0.2)


print("Found min.:", invf(nabla_y), nabla_y)

plt.plot(x, y)
plt.scatter([invf(nabla_y)], [nabla_y], c="r", marker="o")
plt.show()
