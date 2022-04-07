import data_generator


def seq_estimator(m, s):
    eps = 1e-4
    max_iter = 100000
    n = 0
    mean_now = 0
    variance_now = 0
    M2 = 0

    print(f"Data point source function: N({m}, {s})")

    for _ in range(max_iter):
        new_point = data_generator.random_data_generator(m, s)
        print(f"Add data point: {new_point}")
        n += 1
        mean_new = mean_now + ((new_point - mean_now) / n)
        M2 += (new_point - mean_now) * (new_point - mean_new)
        variance_new = (0 if n == 1 else (M2 / (n - 1)))  # Sample variance [unbiased]
        print(f"Mean = {mean_new}   Variance = {variance_new}")

        if (abs(mean_new - mean_now) <= eps) and (abs(variance_new - variance_now) <= eps):
            print("___converge___ (early stop)")
            break

        mean_now = mean_new
        variance_now = variance_new

    print(f"total runs: {n}")


if __name__ == "__main__":
    # seq_estimator(3.0,5.0)
    seq_estimator(241.0, 8.0)