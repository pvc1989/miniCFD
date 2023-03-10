from scipy import special


def fixed_quad_global(function: callable, x_left, x_right, n_point=5):
    x_center = (x_right + x_left) / 2
    jacobian = (x_right - x_left) / 2
    def integrand(x_local):
        x_global = x_center + jacobian * x_local
        return function(x_global)
    return jacobian * fixed_quad_local(integrand, n_point)

def fixed_quad_local(function: callable, n_point=5):
    roots, weights = special.roots_legendre(n_point)
    value = weights[0] * function(roots[0])
    for i in range(1, len(roots)):
        value += weights[i] * function(roots[i])
    return value


if __name__ == '__main__':
    pass
