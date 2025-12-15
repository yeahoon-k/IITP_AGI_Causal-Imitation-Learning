if __name__ == '__main__':
    lr = 0.05
    n = 500
    r_i = 1
    for _ in range(n):
        r_i -= r_i * lr
    print(1 - r_i)
