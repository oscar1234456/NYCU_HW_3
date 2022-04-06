import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

# m: mean
# s: variance
def random_data_generator(m, s):
    U = np.random.rand()
    V = np.random.rand()
    X = np.sqrt((-2 * ln(U)))*np.cos(2*np.pi*V)
    # X is drawn from normal distribution N(0, 1)
    # E[ax+b] = a * E[x]+b ; var[ax+b] = a^2 * var[x]
    # Therefore, we need to let a^2 be s => a = sqrt(s)
    a = np.sqrt(s)
    convert_X = a * X + m
    return convert_X






if __name__ == "__main__":
    samples = list()
    for i in range(20000):
         samples.append(random_data_generator(5,1))
    plt.hist(samples, 50)
    plt.show()

