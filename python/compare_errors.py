import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import sys


def read(folder, degree_range, n_element_range) -> dict:
    """Read the data from CSV files in a given folder.

    Suppose degree_range = (2,3,4), n_element_range = (10,20,40,80), then the CSV files could be obtained by running the following commands in shell:

    1. Solve a problem with various (p, n) combinations:
        for p in {2,3,4,5} ; do for n in {10,20,40,80} ; do python3 ~/code/miniCFD/python/solver.py --method GaussLagrangeDG --degree $p --n_element $n --rk_order 4 --n_step 100000 --t_end 2.0 --problem Smooth --wave_number 2 --physical_viscosity 0.01 --output pdf > p=$p_n=$n.log & ; done ; done

    2. Filter out the errors:
        for x in *.log ; do cat $x | grep error > ${x:0:8}.csv ; done

    3. Replace string:
        for x in *0.csv ; do sed -i 's/\ ],//g' $x ; done
        for x in *0.csv ; do sed -i 's/t_curr,\ error_1,\ error_2,\ error_∞\ =\ \[\ //g' $x ; done
    """
    errors = dict()
    for degree in degree_range:
        errors[degree] = dict()
        for n_element in n_element_range:
            name = f'{folder}/p={degree}_n={n_element}.csv'
            data = np.loadtxt(name, delimiter=',', skiprows=0)
            errors[degree][n_element] = data
    return errors


def plot_slope(errors: dict):
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
    plt.savefig('compare_error_slopes.pdf')


def plot_history(errors: dict):
    fig = plt.figure(figsize=(9, 6))
    markers = ['1', '2', '3', '4']
    ylabels = ['', r'$\Vert u^h - u\Vert_1$', r'$\Vert u^h - u\Vert_2$',
        r'$\Vert u^h - u\Vert_\infty$' ]
    for i_error in (1, 2, 3):
        plt.subplot(1, 3, i_error)
        for degree, subdict in errors.items():
            assert isinstance(subdict, dict)
            for n_element, data in subdict.items():
                xdata = data[:, 0]
                ydata = data[:, i_error]
                plt.plot(xdata, ydata,
                    label=r'$p=$'+f'{degree}, '+r'$h=2/$'+f'{n_element}')
        plt.xlabel(r'$t$')
        plt.ylabel(ylabels[i_error])
        plt.semilogy()
        plt.legend()
        plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig('compare_error_histories.pdf')


def compare_schemes(scheme_to_errors: dict):
    fig = plt.figure(figsize=(9, 6))
    markers = ['1', '2', '3', '4']
    ylabels = ['', r'$\Vert u^h - u\Vert_1$', r'$\Vert u^h - u\Vert_2$',
        r'$\Vert u^h - u\Vert_\infty$' ]
    for i_error in (1, 2, 3):
        plt.subplot(1, 3, i_error)
        for scheme, errors in scheme_to_errors.items():
            assert isinstance(errors, dict)
            for degree, subdict in errors.items():
                assert isinstance(subdict, dict)
                for n_element, data in subdict.items():
                    xdata = data[:, 0]
                    ydata = data[:, i_error]
                    plt.plot(xdata, ydata, label=r'$p=$'+f'{degree}, '+r'$h=2/$'+f'{n_element}, '+scheme)
        plt.xlabel(r'$t$')
        plt.ylabel(ylabels[i_error])
        plt.semilogy()
        plt.legend()
        plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig('compare_schemes.pdf')


if __name__ == '__main__':
    degree_range = (3,4)
    n_element_range =(20,40)
    scheme_to_errors = dict()
    scheme_to_errors['DG'] = read(sys.argv[1], degree_range, n_element_range)
    scheme_to_errors['FR'] = read(sys.argv[2], degree_range, n_element_range)
    compare_schemes(scheme_to_errors)
    plot_history(scheme_to_errors['FR'])
