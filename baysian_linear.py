import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
import data_generator

point_x = list()
point_y = list()
mean_list=[]
variance_list=[]
def baysian_linear_regression(b=1, n=4, a=1, w=np.array([1, 2, 3, 4])):
    # poly_data_generator(n, a, w)
    # posterior: Gaussian(mu,lambda.inv) <multivariate>
    # prior: Gaussian(m, S.inv) <multivariate>
    # likelihood: Gaussian(X.T@w, 1/a) <univariate>
    # mu = lambda.inv @ (a * X.T @ Y + S@m)
    # lambda = a * X.T @ X.T + S
    # X: Design matrix = [[x_1, phi1(x_1), phi2(x_1),...phik(x_1)],...,[x_n, phi1(x_n),...,phik(x_n)]]
    # a: the inverse of variance of likelihood
    # Y: Ground truth
    # m: the mean of prior
    # S.inv: the inverse of cov matrix of prior
    # mu: the mean of posterior
    # lambda: the inverse of cov matrix of posterior
    eps = 1e-3
    max_iter = 1000

    point_x_pool = list()
    point_y_pool = list()


    X = None
    Y = None
    S_inv = (1/b) * np.identity(n)
    m = np.zeros((n, 1))
    a = 1/a

    for _ in range(max_iter):
        new_point_x, new_point_y = data_generator.poly_data_generator(n, a, w)
        print(f"Add data point:({new_point_x}, {new_point_y})")
        point_x_pool.append(new_point_x)
        point_y_pool.append(new_point_y)
        # if X is None:
        #     X = np.array([[np.power(new_point_x, i) for i in range(n)]])
        # else:
        #     X = np.append(X, np.array([[np.power(new_point_x, i) for i in range(n)]]), axis=0)
        X = np.array([[np.power(new_point_x, i) for i in range(n)]])
        Y = np.array([[new_point_y]])
        # Y = np.array([[y] for y in point_y_pool])

        S = inv(S_inv)
        lambda_post = a * X.T @ X + S
        lambda_post_inverse = inv(lambda_post)
        mu = lambda_post_inverse @ (a * X.T @ Y + S@m)

        print("Posterior mean:")
        print(mu)
        print()

        print("Posterior variance:")
        print(lambda_post_inverse)

        m = mu
        S_inv = lambda_post_inverse

        # predictive distribution
        predictive_mean = (m.T @ X.T).item()
        predictive_variance = ((1/a) + X @ S_inv @ X.T).item()
        print('Predictive distribution ~ N({:.5f},{:.5f})'.format(predictive_mean, predictive_variance))
        print('--------------------------------------------------')
        point_x.append(new_point_x)
        point_y.append(new_point_y)
        if _ == 9 or _ == 49 or _ == max_iter-1:
            mean_list.append(m)
            variance_list.append(S_inv)


def plot(num_points, x, mean, variance, title, n=3, a=3):
    mean_predict = np.zeros(500)
    variance_predict = np.zeros(500)
    a = 1/a
    for i in range(len(x)):
        X = np.array([[np.power(x[i], k) for k in range(n)]])
        mean_predict[i] = (mean.T @ X.T).item()
        variance_predict[i] = ((1/a) + X @ variance @ X.T).item()

    plt.plot(point_x[:num_points], point_y[:num_points], 'bo')
    plt.plot(x, mean_predict, 'k-')
    plt.plot(x, mean_predict + variance_predict, 'r-')
    plt.plot(x, mean_predict - variance_predict, 'r-')
    plt.xlim(-2, 2)
    plt.ylim(-20, 20)
    plt.title(title)
    plt.show()



if __name__ == "__main__":
    baysian_linear_regression(a=3, n=3, b=1, w=np.array([1,2,3]))
    x = np.linspace(-2, 2, 500)
    plot(10, x, mean_list[0], variance_list[0], 'After 10 incomes')
    plot(50, x, mean_list[1], variance_list[1], 'After 50 incomes')
    plot(1000, x, mean_list[2], variance_list[2], 'Predict result (10000 incomes)')
    plot(0, x, np.array([1, 2, 3]), np.zeros((3, 3)), 'Ground truth')
