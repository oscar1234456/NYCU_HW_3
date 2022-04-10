import numpy as np
from matplotlib import pyplot as plt


def visualization(a, mu, lambda_post_inverse, x_point_show, y_point_show, plt_title):
    x_num = 800
    x = np.linspace(-2, 2, x_num)

    pred_mean = np.zeros(x_num)
    pred_var = np.zeros(x_num)
    a = 1 / a  # a: the var of likelihood<univariate>
    n = mu.shape[0]

    for i in range(x_num):
        X = np.array([[np.power(x[i], _) for _ in range(n)]])
        # predicted distribution (given data point x)
        pred_mean[i] = (mu.T @ X.T).item()
        pred_var[i] = ((1 / a) + X @ lambda_post_inverse @ X.T).item()

    plt.plot(x_point_show, y_point_show, 'bo')
    plt.plot(x, pred_mean, "k-")
    # plot +- 1 variance line
    plt.plot(x, pred_mean + pred_var, "r-")
    plt.plot(x, pred_mean - pred_var, "r-")
    plt.xlim(-2, 2)
    plt.ylim(-15, 25)
    plt.title(plt_title)
    plt.show()
