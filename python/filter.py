import numpy as np


def Selective(old_array: np.ndarray, periodic: bool):
    new_array = old_array.copy()
    width = 5
    sigma = 1.0
    coeff = np.array([
        +0.001446093078167,
        -0.012396449873964,
        +0.049303775636020,
        -0.120198310245186,
        +0.199250131285813,
        0.234810479761700,  # d[0]
        -0.199250131285813,
        +0.120198310245186,
        -0.049303775636020,
        +0.012396449873964,
        -0.001446093078167,
    ])
    for i in range(1, 1 + width):
        assert coeff[width - i] == -coeff[width + i]
    for i in range(width, len(old_array) - width):
        d_i = coeff.dot(old_array[i - width : i + width + 1])
        new_array[i] -= sigma * d_i
    if periodic:
        for i in range(width):
            d_i = old_array[i] * coeff[width]
            for j in range(1, 1 + width):
                d_i += old_array[i + j] * coeff[width + j]
                d_i += old_array[i - j] * coeff[width - j]
            new_array[i] -= sigma * d_i
        for i in range(len(old_array) - width, len(old_array)):
            d_i = old_array[i] * coeff[width]
            for j in range(1, 1 + width):
                d_i += old_array[(i + j) % len(old_array)] * coeff[width + j]
                d_i += old_array[i - j] * coeff[width - j]
            new_array[i] -= sigma * d_i
    return new_array

