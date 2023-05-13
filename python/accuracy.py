import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

errors = dict()
degree_range = range(2, 5)
for degree in degree_range:
    errors[degree] = dict()

n_element_range = (10, 20, 40, 80)


def plot(errors, figname: str):
    fig = plt.figure(figsize=(9, 6))
    markers = ['1', '2', '3', '4']
    labels = ['', r'$L_1$', r'$L_2$', r'$L_\infty$']
    i_frame = -1
    for i_error in (1, 2, 3):
        plt.subplot(1, 3, i_error)
        for degree in degree_range:
            xdata = 2.0 / np.array(n_element_range)
            ydata = []
            for n_element in n_element_range:
                ydata.append(errors[degree][n_element][i_frame][i_error])
            ydata = np.array(ydata)
            slope = stats.linregress(np.log(xdata), np.log(ydata))[0]
            plt.plot(xdata, ydata, marker=markers[i_error],
                label=r'$p=$'+f'{degree}, slope = {slope:.2f}')
        plt.title(labels[i_error])
        plt.xlabel(r'$h$')
        plt.ylabel(r'$\Vert u^h - u\Vert$')
        plt.loglog()
        plt.axis('equal')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname)


if __name__ == '__main__':
    # python3 solver.py --method LagrangeFR --degree 2 --n_element 10 --n_step 200 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][10] = np.array([
        [ 0.0, 1.023872e-16, 1.230184e-16, 5.327902e-02 ],
        [ 2.5, 4.310336e-01, 3.406199e-01, 3.615924e-01 ],
        [ 5.0, 5.633576e-01, 4.474766e-01, 4.590075e-01 ],
        [ 7.5, 5.709910e-01, 4.434005e-01, 4.459729e-01 ],
        [ 10.0, 4.996582e-01, 3.937789e-01, 3.963108e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 20 --n_step 400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][20] = np.array([
        [ 0.0, 5.890350e-17, 8.093393e-17, 6.895956e-03 ],
        [ 2.5, 4.188786e-02, 3.354287e-02, 3.791334e-02 ],
        [ 5.0, 6.631983e-02, 5.223123e-02, 5.595128e-02 ],
        [ 7.5, 7.855052e-02, 6.165286e-02, 6.463195e-02 ],
        [ 10.0, 8.263711e-02, 6.477296e-02, 6.710629e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 40 --n_step 800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][40] = np.array([
        [ 0.0, 7.015993e-17, 8.669299e-17, 8.695268e-04 ],
        [ 2.5, 4.227763e-03, 3.355010e-03, 3.721164e-03 ],
        [ 5.0, 6.634941e-03, 5.228002e-03, 5.556431e-03 ],
        [ 7.5, 7.937426e-03, 6.239653e-03, 6.505192e-03 ],
        [ 10.0, 8.450302e-03, 6.640129e-03, 6.849354e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 80 --n_step 1600 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][80] = np.array([
        [ 0.0, 5.873003e-17, 8.826487e-17, 1.089275e-04 ],
        [ 2.5, 6.312024e-04, 4.960946e-04, 5.237476e-04 ],
        [ 5.0, 9.954479e-04, 7.818895e-04, 7.988980e-04 ],
        [ 7.5, 1.191653e-03, 9.363126e-04, 9.502559e-04 ],
        [ 10.0, 1.271393e-03, 9.985472e-04, 1.009891e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 10 --n_step 300 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][10] = np.array([
        [ 0.0, 2.393960e-16, 2.088632e-16, 7.109956e-03 ],
        [ 2.5, 3.975443e-02, 3.176138e-02, 4.311049e-02 ],
        [ 5.0, 6.309110e-02, 4.936179e-02, 5.597985e-02 ],
        [ 7.5, 7.406208e-02, 5.806262e-02, 6.227626e-02 ],
        [ 10.0, 7.745002e-02, 6.078688e-02, 6.174096e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 20 --n_step 600 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][20] = np.array([
        [ 0.0, 2.610982e-16, 2.978541e-16, 4.643827e-04 ],
        [ 2.5, 4.182111e-03, 3.294881e-03, 4.549823e-03 ],
        [ 5.0, 6.673143e-03, 5.244460e-03, 6.248899e-03 ],
        [ 7.5, 7.995311e-03, 6.282614e-03, 7.082353e-03 ],
        [ 10.0, 8.517567e-03, 6.693360e-03, 7.329283e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 40 --n_step 1200 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][40] = np.array([
        [ 0.0, 2.268061e-16, 2.538131e-16, 2.927053e-05 ],
        [ 2.5, 5.006531e-04, 3.934650e-04, 4.748268e-04 ],
        [ 5.0, 8.018129e-04, 6.294682e-04, 6.938945e-04 ],
        [ 7.5, 9.630173e-04, 7.558643e-04, 8.067612e-04 ],
        [ 10.0, 1.028096e-03, 8.068821e-04, 8.470612e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 80 --n_step 2400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][80] = np.array([
        [ 0.0, 2.510308e-16, 2.532121e-16, 1.833281e-06 ],
        [ 2.5, 6.142712e-05, 4.826626e-05, 5.252267e-05 ],
        [ 5.0, 9.838493e-05, 7.729285e-05, 8.070614e-05 ],
        [ 7.5, 1.181850e-04, 9.284566e-05, 9.558015e-05 ],
        [ 10.0, 1.261958e-04, 9.913811e-05, 1.013283e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 10 --n_step 600 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][10] = np.array([
        [ 0.0, 5.797319e-16, 5.132326e-16, 7.525160e-04 ],
        [ 2.5, 3.876541e-03, 3.039266e-03, 5.239900e-03 ],
        [ 5.0, 6.124877e-03, 4.808059e-03, 6.519021e-03 ],
        [ 7.5, 7.327603e-03, 5.753903e-03, 7.176092e-03 ],
        [ 10.0, 7.836905e-03, 6.128486e-03, 7.102003e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --n_step 1200 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][20] = np.array([
        [ 0.0, 2.900440e-16, 3.082579e-16, 2.431785e-05 ],
        [ 2.5, 4.701162e-04, 3.693059e-04, 4.148093e-04 ],
        [ 5.0, 7.531028e-04, 5.908831e-04, 6.274607e-04 ],
        [ 7.5, 9.046012e-04, 7.095394e-04, 7.388598e-04 ],
        [ 10.0, 9.657855e-04, 7.574351e-04, 7.809218e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --n_step 2400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][40] = np.array([
        [ 0.0, 3.151449e-16, 3.775880e-16, 7.663231e-07 ],
        [ 2.5, 5.982152e-05, 4.697323e-05, 4.768905e-05 ],
        [ 5.0, 9.581582e-05, 7.523049e-05, 7.580214e-05 ],
        [ 7.5, 1.151000e-04, 9.036974e-05, 9.082557e-05 ],
        [ 10.0, 1.229023e-04, 9.649479e-05, 9.685814e-05 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 80 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][80] = np.array([
        [ 0.0, 2.368917e-16, 3.106168e-16, 2.399777e-08 ],
        [ 2.5, 7.564419e-06, 5.941351e-06, 5.948184e-06 ],
        [ 5.0, 1.211603e-05, 9.516312e-06, 9.520558e-06 ],
        [ 7.5, 1.455483e-05, 1.143182e-05, 1.143424e-05 ],
        [ 10.0, 1.554182e-05, 1.220704e-05, 1.220818e-05 ],
    ])

    plot(errors, 'compare_errors_cfl_fixed.pdf')

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][80] = np.array([
        [ 0.0, 2.368917e-16, 3.106168e-16, 2.399777e-08 ],
        [ 2.5, 2.667463e-08, 2.184768e-08, 5.257010e-08 ],
        [ 5.0, 3.239331e-08, 2.590277e-08, 4.839213e-08 ],
        [ 7.5, 3.614387e-08, 2.864644e-08, 4.499473e-08 ],
        [ 10.0, 3.750394e-08, 2.961242e-08, 4.166413e-08 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][40] = np.array([
        [ 0.0, 3.151449e-16, 3.775880e-16, 7.663231e-07 ],
        [ 2.5, 1.129298e-06, 9.040213e-07, 2.275042e-06 ],
        [ 5.0, 1.571367e-06, 1.239032e-06, 2.098414e-06 ],
        [ 7.5, 1.825719e-06, 1.437482e-06, 1.994388e-06 ],
        [ 10.0, 1.928673e-06, 1.515582e-06, 1.884610e-06 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][20] = np.array([
        [ 0.0, 2.900440e-16, 3.082579e-16, 2.431785e-05 ],
        [ 2.5, 2.802282e-05, 2.311158e-05, 7.761890e-05 ],
        [ 5.0, 3.601144e-05, 2.881530e-05, 6.707494e-05 ],
        [ 7.5, 4.113691e-05, 3.247620e-05, 5.860888e-05 ],
        [ 10.0, 4.300079e-05, 3.384715e-05, 5.227184e-05 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][10] = np.array([
        [ 0.0, 5.797319e-16, 5.132326e-16, 7.525160e-04 ],
        [ 2.5, 7.338255e-04, 6.299584e-04, 2.485578e-03 ],
        [ 5.0, 8.738545e-04, 7.212364e-04, 2.055872e-03 ],
        [ 7.5, 9.837593e-04, 7.870151e-04, 1.844400e-03 ],
        [ 10.0, 9.976349e-04, 8.087596e-04, 1.596192e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][80] = np.array([
        [ 0.0, 2.510308e-16, 2.532121e-16, 1.833281e-06 ],
        [ 2.5, 4.924748e-06, 3.868652e-06, 5.688013e-06 ],
        [ 5.0, 7.649464e-06, 6.009433e-06, 6.848701e-06 ],
        [ 7.5, 9.139109e-06, 7.177395e-06, 7.629660e-06 ],
        [ 10.0, 9.739989e-06, 7.648746e-06, 7.916710e-06 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][40] = np.array([
        [ 0.0, 2.268061e-16, 2.538131e-16, 2.927053e-05 ],
        [ 2.5, 5.816774e-05, 4.570590e-05, 9.503100e-05 ],
        [ 5.0, 8.606050e-05, 6.758196e-05, 9.444030e-05 ],
        [ 7.5, 1.017195e-04, 7.988829e-05, 9.531060e-05 ],
        [ 10.0, 1.080189e-04, 8.482169e-05, 9.432008e-05 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][20] = np.array([
        [ 0.0, 2.610982e-16, 2.978541e-16, 4.643827e-04 ],
        [ 2.5, 7.971729e-04, 6.270173e-04, 1.581166e-03 ],
        [ 5.0, 1.104507e-03, 8.716972e-04, 1.411118e-03 ],
        [ 7.5, 1.292441e-03, 1.015279e-03, 1.315824e-03 ],
        [ 10.0, 1.365395e-03, 1.072031e-03, 1.242975e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][10] = np.array([
        [ 0.0, 2.393960e-16, 2.088632e-16, 7.109956e-03 ],
        [ 2.5, 1.707514e-02, 1.346880e-02, 2.176925e-02 ],
        [ 5.0, 2.534757e-02, 1.994425e-02, 2.112129e-02 ],
        [ 7.5, 2.976359e-02, 2.351321e-02, 2.511729e-02 ],
        [ 10.0, 3.185606e-02, 2.487061e-02, 2.557702e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][80] = np.array([
        [ 0.0, 5.873003e-17, 8.826487e-17, 1.089275e-04 ],
        [ 2.5, 4.350583e-04, 3.426547e-04, 4.043383e-04 ],
        [ 5.0, 6.744584e-04, 5.301753e-04, 5.519896e-04 ],
        [ 7.5, 8.052072e-04, 6.326730e-04, 6.501568e-04 ],
        [ 10.0, 8.578967e-04, 6.739486e-04, 6.883325e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][40] = np.array([
        [ 0.0, 7.015993e-17, 8.669299e-17, 8.695268e-04 ],
        [ 2.5, 2.866479e-03, 2.342559e-03, 2.802704e-03 ],
        [ 5.0, 4.464521e-03, 3.546097e-03, 3.965391e-03 ],
        [ 7.5, 5.331639e-03, 4.210527e-03, 4.550602e-03 ],
        [ 10.0, 5.680408e-03, 4.475005e-03, 4.752664e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][20] = np.array([
        [ 0.0, 5.890350e-17, 8.093393e-17, 6.895956e-03 ],
        [ 2.5, 3.500466e-02, 2.831048e-02, 3.309518e-02 ],
        [ 5.0, 5.583482e-02, 4.407076e-02, 4.815377e-02 ],
        [ 7.5, 6.654548e-02, 5.228256e-02, 5.552902e-02 ],
        [ 10.0, 7.041991e-02, 5.524484e-02, 5.775329e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][10] = np.array([
        [ 0.0, 1.023872e-16, 1.230184e-16, 5.327902e-02 ],
        [ 2.5, 3.999947e-01, 3.196617e-01, 3.430372e-01 ],
        [ 5.0, 5.548498e-01, 4.349462e-01, 4.493301e-01 ],
        [ 7.5, 5.704174e-01, 4.433368e-01, 4.455532e-01 ],
        [ 10.0, 5.066991e-01, 4.019542e-01, 4.040432e-01 ],
    ])

    plot(errors, 'compare_errors_dt_small.pdf')

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[4][80] = np.array([
        [ 0.0, 2.368917e-16, 3.106168e-16, 2.399777e-08 ],
        [ 2.5, 2.283232e-08, 2.133747e-08, 1.095029e-07 ],
        [ 5.0, 2.285756e-08, 2.150791e-08, 1.110458e-07 ],
        [ 7.5, 2.296062e-08, 2.178902e-08, 1.125888e-07 ],
        [ 10.0, 2.324547e-08, 2.217660e-08, 1.141317e-07 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[4][40] = np.array([
        [ 0.0, 3.151449e-16, 3.775880e-16, 7.663231e-07 ],
        [ 2.5, 7.405830e-07, 6.804392e-07, 3.448978e-06 ],
        [ 5.0, 7.395830e-07, 6.804302e-07, 3.453115e-06 ],
        [ 7.5, 7.385562e-07, 6.804996e-07, 3.457334e-06 ],
        [ 10.0, 7.375294e-07, 6.805967e-07, 3.461553e-06 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[4][20] = np.array([
        [ 0.0, 2.900440e-16, 3.082579e-16, 2.431785e-05 ],
        [ 2.5, 2.345268e-05, 2.173822e-05, 1.102133e-04 ],
        [ 5.0, 2.340831e-05, 2.175749e-05, 1.103278e-04 ],
        [ 7.5, 2.342590e-05, 2.180314e-05, 1.109162e-04 ],
        [ 10.0, 2.348690e-05, 2.187987e-05, 1.115617e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[4][10] = np.array([
        [ 0.0, 5.797319e-16, 5.132326e-16, 7.525160e-04 ],
        [ 2.5, 7.688763e-04, 7.077403e-04, 3.354131e-03 ],
        [ 5.0, 8.999450e-04, 7.614339e-04, 3.597739e-03 ],
        [ 7.5, 9.764740e-04, 8.490435e-04, 3.660350e-03 ],
        [ 10.0, 1.122216e-03, 9.566213e-04, 3.811075e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[3][80] = np.array([
        [ 0.0, 2.510308e-16, 2.532121e-16, 1.833281e-06 ],
        [ 2.5, 2.135326e-06, 1.953767e-06, 8.552621e-06 ],
        [ 5.0, 2.145532e-06, 1.956066e-06, 8.547696e-06 ],
        [ 7.5, 2.159910e-06, 1.959894e-06, 8.542770e-06 ],
        [ 10.0, 2.177076e-06, 1.965241e-06, 8.537844e-06 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[3][40] = np.array([
        [ 0.0, 2.268061e-16, 2.538131e-16, 2.927053e-05 ],
        [ 2.5, 3.503134e-05, 3.142523e-05, 1.361380e-04 ],
        [ 5.0, 3.647650e-05, 3.196719e-05, 1.355427e-04 ],
        [ 7.5, 3.838138e-05, 3.285138e-05, 1.349474e-04 ],
        [ 10.0, 4.054760e-05, 3.405115e-05, 1.347907e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[3][20] = np.array([
        [ 0.0, 2.610982e-16, 2.978541e-16, 4.643827e-04 ],
        [ 2.5, 6.509492e-04, 5.415163e-04, 2.101376e-03 ],
        [ 5.0, 8.188890e-04, 6.534579e-04, 2.052966e-03 ],
        [ 7.5, 1.020268e-03, 8.064481e-04, 2.044768e-03 ],
        [ 10.0, 1.243519e-03, 9.814674e-04, 2.036614e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[3][10] = np.array([
        [ 0.0, 2.393960e-16, 2.088632e-16, 7.109956e-03 ],
        [ 2.5, 1.787293e-02, 1.433736e-02, 2.611302e-02 ],
        [ 5.0, 3.231977e-02, 2.544438e-02, 2.735044e-02 ],
        [ 7.5, 4.709399e-02, 3.713689e-02, 3.907204e-02 ],
        [ 10.0, 6.209887e-02, 4.894591e-02, 5.131903e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[2][80] = np.array([
        [ 0.0, 5.873003e-17, 8.826487e-17, 1.089275e-04 ],
        [ 2.5, 2.514808e-04, 2.151476e-04, 3.932036e-04 ],
        [ 5.0, 3.840980e-04, 3.383455e-04, 4.229390e-04 ],
        [ 7.5, 5.759785e-04, 4.777033e-04, 5.738234e-04 ],
        [ 10.0, 7.680668e-04, 6.224586e-04, 7.247055e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[2][40] = np.array([
        [ 0.0, 7.015993e-17, 8.669299e-17, 8.695268e-04 ],
        [ 2.5, 3.043231e-03, 2.675481e-03, 3.340155e-03 ],
        [ 5.0, 6.081459e-03, 4.927416e-03, 5.732216e-03 ],
        [ 7.5, 9.124923e-03, 7.268743e-03, 8.123079e-03 ],
        [ 10.0, 1.216694e-02, 9.633669e-03, 1.051273e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[2][20] = np.array([
        [ 0.0, 5.890350e-17, 8.093393e-17, 6.895956e-03 ],
        [ 2.5, 4.624939e-02, 3.743289e-02, 4.336645e-02 ],
        [ 5.0, 9.210893e-02, 7.296567e-02, 7.929743e-02 ],
        [ 7.5, 1.372286e-01, 1.083598e-01, 1.146114e-01 ],
        [ 10.0, 1.815686e-01, 1.433210e-01, 1.492783e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.0 | grep error
    errors[2][10] = np.array([
        [ 0.0, 1.023872e-16, 1.230184e-16, 5.327902e-02 ],
        [ 2.5, 5.172761e-01, 4.125648e-01, 4.421419e-01 ],
        [ 5.0, 8.927840e-01, 6.991611e-01, 7.208071e-01 ],
        [ 7.5, 1.140520e+00, 8.865157e-01, 8.902474e-01 ],
        [ 10.0, 1.260083e+00, 9.990211e-01, 1.003824e+00 ],
    ])

    plot(errors, 'compare_inviscid_errors_dt_small.pdf')
