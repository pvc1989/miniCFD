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
        # plt.axis('equal')
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

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][40] = np.array([
       [ 0.0, 3.151449e-16, 1.962144e-15, 2.030197e-05 ],
       [ 2.5, 6.780217e-06, 3.098266e-05, 1.959434e-04 ],
       [ 5.0, 1.082290e-05, 4.925595e-05, 2.855817e-04 ],
       [ 7.5, 1.299277e-05, 5.909841e-05, 3.330345e-04 ],
       [ 10.0, 1.386921e-05, 6.308336e-05, 3.502626e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][20] = np.array([
       [ 0.0, 2.900440e-16, 1.190799e-15, 3.353849e-04 ],
       [ 2.5, 2.400355e-05, 8.893468e-05, 1.041773e-03 ],
       [ 5.0, 2.877830e-05, 1.025369e-04, 9.236321e-04 ],
       [ 7.5, 3.202799e-05, 1.109171e-04, 8.129536e-04 ],
       [ 10.0, 3.319752e-05, 1.131894e-04, 7.183358e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 10 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][10] = np.array([
        [ 0.0, 5.797319e-16, 1.474262e-15, 5.404930e-03 ],
        [ 2.5, 7.332263e-04, 1.894610e-03, 1.611856e-02 ],
        [ 5.0, 8.728844e-04, 2.227024e-03, 1.442682e-02 ],
        [ 7.5, 9.820488e-04, 2.439648e-03, 1.264946e-02 ],
        [ 10.0, 9.963395e-04, 2.510373e-03, 1.127273e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 80 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][80] = np.array([
        [ 0.0, 2.510308e-16, 1.905418e-15, 9.557854e-05 ],
        [ 2.5, 9.528324e-06, 6.153667e-05, 5.815299e-04 ],
        [ 5.0, 1.522257e-05, 9.731467e-05, 7.699613e-04 ],
        [ 7.5, 1.827930e-05, 1.165858e-04, 8.721319e-04 ],
        [ 10.0, 1.951491e-05, 1.243620e-04, 9.084385e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 40 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][40] = np.array([
        [ 0.0, 2.268061e-16, 1.287009e-15, 7.808473e-04 ],
        [ 2.5, 6.041203e-05, 2.927111e-04, 2.808792e-03 ],
        [ 5.0, 9.014772e-05, 4.219748e-04, 2.981300e-03 ],
        [ 7.5, 1.066496e-04, 4.935347e-04, 3.017557e-03 ],
        [ 10.0, 1.133575e-04, 5.213995e-04, 2.925537e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 20 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][20] = np.array([
        [ 0.0, 2.610982e-16, 9.915719e-16, 6.482713e-03 ],
        [ 2.5, 7.998819e-04, 2.789429e-03, 2.158982e-02 ],
        [ 5.0, 1.111043e-03, 3.795365e-03, 2.073050e-02 ],
        [ 7.5, 1.298594e-03, 4.368461e-03, 1.972380e-02 ],
        [ 10.0, 1.371956e-03, 4.586078e-03, 1.904274e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 10 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][10] = np.array([
        [ 0.0, 2.393960e-16, 6.368509e-16, 5.439847e-02 ],
        [ 2.5, 1.708044e-02, 4.239975e-02, 1.651910e-01 ],
        [ 5.0, 2.535554e-02, 6.237009e-02, 1.911353e-01 ],
        [ 7.5, 2.977157e-02, 7.334214e-02, 2.170871e-01 ],
        [ 10.0, 3.186581e-02, 7.748639e-02, 2.346074e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 80 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][80] = np.array([
        [ 0.0, 5.873003e-17, 5.682067e-16, 5.710347e-03 ],
        [ 2.5, 4.422213e-04, 2.931230e-03, 2.774047e-02 ],
        [ 5.0, 6.863253e-04, 4.418516e-03, 3.448050e-02 ],
        [ 7.5, 8.194554e-04, 5.235594e-03, 3.785690e-02 ],
        [ 10.0, 8.732409e-04, 5.562809e-03, 3.892834e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 40 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][40] = np.array([
        [ 0.0, 7.015993e-17, 4.785912e-16, 2.343644e-02 ],
        [ 2.5, 2.872330e-03, 1.384815e-02, 8.739699e-02 ],
        [ 5.0, 4.474158e-03, 2.059170e-02, 1.132117e-01 ],
        [ 7.5, 5.342870e-03, 2.437491e-02, 1.297621e-01 ],
        [ 10.0, 5.692359e-03, 2.589465e-02, 1.352361e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 20 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][20] = np.array([
        [ 0.0, 5.890350e-17, 2.898351e-16, 9.800240e-02 ],
        [ 2.5, 3.500901e-02, 1.185390e-01, 4.892222e-01 ],
        [ 5.0, 5.584137e-02, 1.850374e-01, 7.234898e-01 ],
        [ 7.5, 6.655286e-02, 2.197504e-01, 8.592294e-01 ],
        [ 10.0, 7.042728e-02, 2.323253e-01, 9.112987e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 10 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][10] = np.array([
        [ 0.0, 1.023872e-16, 3.452150e-16, 4.009062e-01 ],
        [ 2.5, 3.999973e-01, 9.926776e-01, 2.894889e+00 ],
        [ 5.0, 5.548519e-01, 1.352347e+00, 4.014905e+00 ],
        [ 7.5, 5.704191e-01, 1.378813e+00, 4.161877e+00 ],
        [ 10.0, 5.066995e-01, 1.249967e+00, 3.796112e+00 ],
    ])

    plot(errors, 'compare_errors_dt_fixed.pdf')

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 80 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][80] = np.array([
        [ 0.0, 2.368917e-16, 2.233226e-15, 1.246519e-06 ],
        [ 2.5, 2.667463e-08, 1.933398e-07, 2.796981e-06 ],
        [ 5.0, 3.239331e-08, 2.278625e-07, 2.698880e-06 ],
        [ 7.5, 3.614387e-08, 2.468670e-07, 2.565848e-06 ],
        [ 10.0, 3.750394e-08, 2.513238e-07, 2.397843e-06 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][40] = np.array([
        [ 0.0, 3.151449e-16, 1.962144e-15, 2.030197e-05 ],
        [ 2.5, 1.129298e-06, 5.682627e-06, 6.196763e-05 ],
        [ 5.0, 1.571367e-06, 7.551607e-06, 6.104314e-05 ],
        [ 7.5, 1.825719e-06, 8.596917e-06, 5.910268e-05 ],
        [ 10.0, 1.928673e-06, 8.972789e-06, 5.596292e-05 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][20] = np.array([
        [ 0.0, 2.900440e-16, 1.190799e-15, 3.353849e-04 ],
        [ 2.5, 2.802282e-05, 1.020279e-04, 1.028525e-03 ],
        [ 5.0, 3.601144e-05, 1.263143e-04, 9.218623e-04 ],
        [ 7.5, 4.113691e-05, 1.404693e-04, 8.310785e-04 ],
        [ 10.0, 4.300079e-05, 1.451794e-04, 7.480224e-04 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 10 --rk_order 4 --n_step 10000 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][10] = np.array([
        [ 0.0, 5.797319e-16, 1.474262e-15, 5.404930e-03 ],
        [ 2.5, 7.338255e-04, 1.896512e-03, 1.608700e-02 ],
        [ 5.0, 8.738545e-04, 2.229685e-03, 1.436728e-02 ],
        [ 7.5, 9.837593e-04, 2.442677e-03, 1.259951e-02 ],
        [ 10.0, 9.976349e-04, 2.513452e-03, 1.121897e-02 ],
    ])

    plot(errors, 'compare_errors_dt_small.pdf')
