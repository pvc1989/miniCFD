import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import sys


def read(folder, degree_range, n_element_range) -> dict:
    print(folder)
    errors = dict()
    for degree in degree_range:
        errors[degree] = dict()
        for n_element in n_element_range:
            name = f'{folder}/p={degree}_n={n_element}.csv'
            data = np.loadtxt(name, delimiter=',', skiprows=1)
            errors[degree][n_element] = data
    return errors


def plot(errors: dict, figname: str):
    fig = plt.figure(figsize=(9, 6))
    markers = ['1', '2', '3', '4']
    ylabels = ['', r'$\Vert u^h - u\Vert_1$', r'$\Vert u^h - u\Vert_2$',
        r'$\Vert u^h - u\Vert_\infty$' ]
    i_frame = -1
    for i_error in (1, 2, 3):
        plt.subplot(1, 3, i_error)
        for degree, subdict in errors.items():
            xdata = []
            ydata = []
            assert isinstance(subdict, dict)
            for n_element, data in subdict.items():
                xdata.append(2 / n_element)
                ydata.append(data[i_frame][i_error])
            ydata = np.array(ydata)
            slope = stats.linregress(np.log(xdata), np.log(ydata))[0]
            plt.plot(xdata, ydata, marker=markers[i_error],
                label=r'$p=$'+f'{degree}, slope = {slope:.2f}')
        plt.xlabel(r'$h$')
        plt.ylabel(ylabels[i_error])
        plt.loglog()
        plt.axis('equal')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname)


if __name__ == '__main__':
    errors = read(sys.argv[1], (2, 3, 4), (10, 20, 40, 80))
    plot(errors, 'compare_errors.pdf')
