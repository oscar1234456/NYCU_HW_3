import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
import data_generator
from plot import visualization


def baysian_linear_regression(b=1, n=4, a=1, w=np.array([1, 2, 3, 4]), max_iter=10000):
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
    mean_record = list()
    var_record = list()

    point_x_pool = list()
    point_y_pool = list()

    X = None
    Y = None
    S_inv = (1 / b) * np.identity(n)
    m = np.zeros((n, 1))
    a = 1 / a

    for _ in range(max_iter):
        print(f"__________epoch{_}__________")
        new_point_x, new_point_y = data_generator.poly_data_generator(n, a, w)  # generate new points
        print(f"Add data point:({new_point_x}, {new_point_y})")
        # record the generated points
        point_x_pool.append(new_point_x)
        point_y_pool.append(new_point_y)

        X = np.array([[np.power(new_point_x, i) for i in range(n)]])  # design matrix [n * k] = [1 * k]
        Y = np.array([[new_point_y]])  # ground truth [n * 1]

        S = inv(S_inv)  # S: cov matrix of prior [k * k]
        lambda_post = a * X.T @ X + S  # the inverse of cov matrix of posterior [k * k]
        lambda_post_inverse = inv(lambda_post)
        mu = lambda_post_inverse @ (a * X.T @ Y + S @ m)  # the mean of posterior [k * 1]

        print("Posterior mean:")
        print(mu)
        print()

        print("Posterior variance:")
        print(lambda_post_inverse)

        # predictive distribution
        predictive_mean = (mu.T @ X.T).item()  # the mean of posterior.T, design matrix.T  [1 * k] [k * n=1] scalar
        predictive_variance = (
                (1 / a) + X @ lambda_post_inverse @ X.T).item()  # cov matrix of posterior [n=1 * n=1] scalar
        print(f'Predictive distribution ~ N({predictive_mean:.5f},{predictive_variance:.5f})')
        print("________________________")

        # check converged
        if np.allclose(m, mu, rtol=1e-3) and np.allclose(S_inv, lambda_post_inverse, rtol=1e-3):
            print("__converged! Early stop!")
            mean_record.append(mu)
            var_record.append(lambda_post_inverse)
            break

        # record the mean and var of posterior
        if _ == 9 or _ == 49 or _ == max_iter - 1:
            mean_record.append(mu)
            var_record.append(lambda_post_inverse)

        # update mean and var from posterior to prior (online)
        m = mu
        S_inv = lambda_post_inverse

    print(f"total runs: {_}")
    return _, point_x_pool, point_y_pool, mean_record, var_record


if __name__ == "__main__":
    b = 1
    n = 3
    a = 3
    w = np.array([1, 2, 3])
    max_iter = 10000
    _, point_x_pool, point_y_pool, mean_record, var_record = baysian_linear_regression(a=a, n=n, b=b, w=w,
                                                                                       max_iter=max_iter)

    mean_record_10 = mean_record[0]
    mean_record_50 = mean_record[1]
    mean_record_full = mean_record[2]
    mean_gt = w

    var_record_10 = var_record[0]
    var_record_50 = var_record[1]
    var_record_full = var_record[2]
    var_gt = np.zeros((n, n))  # control by a

    visualization(a, mean_record_10, var_record_10, point_x_pool[:10], point_y_pool[:10], "After 10 incomes")
    visualization(a, mean_record_50, var_record_50, point_x_pool[:50], point_y_pool[:50], "After 50 incomes")
    visualization(a, mean_record_full, var_record_full, point_x_pool, point_y_pool,
                  f"Predict result (After {_} incomes)")
    visualization(a, mean_gt, var_gt, point_x_pool[:0], point_y_pool[:0], "Ground truth")
