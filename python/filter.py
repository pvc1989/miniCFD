import numpy as np


class Selective:

    coeff_ = np.array([
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

    def __init__(self, periodic) -> None:
        self._periodic = periodic
        width = 5
        for i in range(1, 1 + width):
            assert self.coeff_[width - i] == -self.coeff_[width + i]

    def filter(self, old_array: np.ndarray, sigma) -> np.ndarray:
        new_array = old_array.copy()
        width = 5
        n_point = len(old_array)
        for i in range(width, n_point - width):
            d_i = self.coeff_.dot(old_array[i - width : i + width + 1])
            new_array[i] -= sigma * d_i
        if self._periodic:
            for i in range(width):
                d_i = old_array[i] * self.coeff_[width]
                for j in range(1, 1 + width):
                    d_i += old_array[i + j] * self.coeff_[width + j]
                    d_i += old_array[i - j] * self.coeff_[width - j]
                new_array[i] -= sigma * d_i
            for i in range(n_point - width, n_point):
                d_i = old_array[i] * self.coeff_[width]
                for j in range(1, 1 + width):
                    k = (i + j) % n_point
                    d_i += old_array[k] * self.coeff_[width + j]
                    d_i += old_array[i - j] * self.coeff_[width - j]
                new_array[i] -= sigma * d_i
        return new_array

    def damping(self, theta):
        width = 5
        damp = self.coeff_[width] * np.cos(0 * theta)
        for j in range(1, 1 + width):
            damp += 2 * self.coeff_[width + j] * np.cos(j * theta)
        return damp
