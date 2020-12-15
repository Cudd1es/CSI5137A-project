def poly_single(x):
    return x ** 3 + 4 * x ** 2 + 2 * x + 1


def poly_multiple(a, b, c):
    return a * b + a * c + b * c


def logical_func(a, b):
    if a > b:
        return a + b
    else:
        return a - b
