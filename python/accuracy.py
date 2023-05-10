import numpy as np
from matplotlib import pyplot as plt

errors = dict()
degree_range = range(2, 5)
for degree in degree_range:
    errors[degree] = dict()

n_element_range = (20, 40, 80)


def plot(errors, figname: str):
    fig = plt.figure(figsize=(9, 6))
    markers = ['1', '2', '3', '4']
    ylabels = ['', r'$L_1$', r'$L_2$', r'$L_\infty$']
    i_frame = -1
    for i_error in (1, 2, 3):
        plt.subplot(1, 3, i_error)
        for i in range(len(degree_range)):
            degree = degree_range[i]
            xdata = n_element_range
            ydata = []
            for n_element in n_element_range:
                ydata.append(errors[degree][n_element][i_frame][i_error])
            plt.plot(xdata, ydata, marker=markers[i], label=r'$p=$'+f'{degree}')
        plt.xlabel(r'$N_\mathrm{Elements}$')
        plt.ylabel(f'{ylabels[i_error]} Error')
        plt.loglog()
        plt.axis('equal')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname)


if __name__ == '__main__':
    # python3 solver.py --method LagrangeFR --degree 2 --n_element 20 --n_step 400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][20] = np.array([
        [ 0.0, 5.890350e-17, 2.898351e-16, 9.800240e-02 ],
        [ 2.5, 4.188786e-02, 1.402433e-01, 5.735485e-01 ],
        [ 5.0, 6.631983e-02, 2.189130e-01, 8.540817e-01 ],
        [ 7.5, 7.855052e-02, 2.587670e-01, 1.010068e+00 ],
        [ 10.0, 8.263711e-02, 2.720789e-01, 1.068306e+00 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 40 --n_step 800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][40] = np.array([
        [ 0.0, 7.015993e-17, 4.785912e-16, 2.343644e-02 ],
        [ 2.5, 4.227763e-03, 1.967752e-02, 1.162502e-01 ],
        [ 5.0, 6.634941e-03, 3.023285e-02, 1.606710e-01 ],
        [ 7.5, 7.937426e-03, 3.602522e-02, 1.870487e-01 ],
        [ 10.0, 8.450302e-03, 3.833938e-02, 1.963745e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 2 --n_element 80 --n_step 1600 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[2][80] = np.array([
        [ 0.0, 5.873003e-17, 5.682067e-16, 5.710347e-03 ],
        [ 2.5, 6.312024e-04, 4.105777e-03, 3.501969e-02 ],
        [ 5.0, 9.954479e-04, 6.361115e-03, 4.638609e-02 ],
        [ 7.5, 1.191653e-03, 7.589652e-03, 5.284564e-02 ],
        [ 10.0, 1.271393e-03, 8.085199e-03, 5.573649e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 20 --n_step 600 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][20] = np.array([
        [ 0.0, 2.610982e-16, 9.915719e-16, 6.482713e-03 ],
        [ 2.5, 4.182111e-03, 1.386175e-02, 6.714190e-02 ],
        [ 5.0, 6.673143e-03, 2.207131e-02, 9.781184e-02 ],
        [ 7.5, 7.995311e-03, 2.644306e-02, 1.142277e-01 ],
        [ 10.0, 8.517567e-03, 2.817322e-02, 1.201100e-01 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 40 --n_step 1200 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][40] = np.array([
        [ 0.0, 2.268061e-16, 1.287009e-15, 7.808473e-04 ],
        [ 2.5, 5.006531e-04, 2.279838e-03, 1.333953e-02 ],
        [ 5.0, 8.018129e-04, 3.647166e-03, 2.023974e-02 ],
        [ 7.5, 9.630173e-04, 4.379458e-03, 2.390587e-02 ],
        [ 10.0, 1.028096e-03, 4.675026e-03, 2.532269e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 3 --n_element 80 --n_step 2400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[3][80] = np.array([
        [ 0.0, 2.510308e-16, 1.905418e-15, 9.557854e-05 ],
        [ 2.5, 6.142712e-05, 3.910401e-04, 2.862822e-03 ],
        [ 5.0, 9.838493e-05, 6.261669e-04, 4.472291e-03 ],
        [ 7.5, 1.181850e-04, 7.521484e-04, 5.330396e-03 ],
        [ 10.0, 1.261958e-04, 8.031158e-04, 5.670493e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 20 --n_step 1200 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][20] = np.array([
        [ 0.0, 2.900440e-16, 1.190799e-15, 3.353849e-04 ],
        [ 2.5, 4.701162e-04, 1.554672e-03, 6.939998e-03 ],
        [ 5.0, 7.531028e-04, 2.487033e-03, 1.067703e-02 ],
        [ 7.5, 9.046012e-04, 2.986531e-03, 1.265434e-02 ],
        [ 10.0, 9.657855e-04, 3.188216e-03, 1.341922e-02 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 40 --n_step 2400 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][40] = np.array([
        [ 0.0, 3.151449e-16, 1.962144e-15, 2.030197e-05 ],
        [ 2.5, 5.982152e-05, 2.720891e-04, 1.471298e-03 ],
        [ 5.0, 9.581582e-05, 4.358170e-04, 2.336006e-03 ],
        [ 7.5, 1.151000e-04, 5.235451e-04, 2.797922e-03 ],
        [ 10.0, 1.229023e-04, 5.590442e-04, 2.983189e-03 ],
    ])

    # python3 solver.py --method LagrangeFR --degree 4 --n_element 80 --n_step 4800 --problem Smooth --wave_number 3 --viscous_coeff 0.001 | grep error
    errors[4][80] = np.array([
        [ 0.0, 2.368917e-16, 2.233226e-15, 1.246519e-06 ],
        [ 2.5, 7.564419e-06, 4.812315e-05, 3.367534e-04 ],
        [ 5.0, 1.211603e-05, 7.708385e-05, 5.388866e-04 ],
        [ 7.5, 1.455483e-05, 9.260176e-05, 6.471593e-04 ],
        [ 10.0, 1.554182e-05, 9.888229e-05, 6.909385e-04 ],
    ])

    plot(errors, 'compare_errors_rk=3.pdf')

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

    plot(errors, 'compare_errors.pdf')
