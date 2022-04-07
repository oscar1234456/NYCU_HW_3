import numpy as np
import data_generator


def baysian_linear_regression(b=1, n=4, a=1, w=np.array([1, 2, 3, 4])):
    # poly_data_generator(n, a, w)
    eps = 1e-3

    new_point_x, new_point_y = data_generator.poly_data_generator(n, a, w)
    print(f"Add data point:({new_point_x}, {new_point_y})")




if __name__ == "__main__":
    baysian_linear_regression()
