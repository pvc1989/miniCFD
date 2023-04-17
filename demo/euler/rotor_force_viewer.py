#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
import warnings


if __name__ == '__main__':
    if len(sys.argv) < 5:
       print(f'python3 {sys.argv[0]}.py <case> <i_frame_min> <i_frame_max> <n_part>')
       exit(-1)
    case = sys.argv[1]
    i_frame_min = int(sys.argv[2])
    i_frame_max = int(sys.argv[3])
    n_part = int(sys.argv[4])
    for i_frame in range(i_frame_min, i_frame_max+1):
        data = []
        n_row = 0
        for i_part in range(n_part):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data.append(np.loadtxt(f'{case}/Frame{i_frame}/{i_part}.csv',
                    delimiter=',', skiprows=1))
            n_row += len(data[-1])
        points = np.ndarray((n_row, 3))
        forces = np.ndarray((n_row, 3))
        weights = np.ndarray(n_row)
        i_row = 0
        for i_part in range(n_part):
            for row in data[i_part]:
                points[i_row] = row[0:3]
                forces[i_row] = row[3:6]
                weights[i_row] = row[6]
                i_row += 1
        assert i_row == n_row
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')
        ax.quiver(points[:,0], points[:,1], points[:,2],
            forces[:,0], forces[:,1], forces[:,2], length=0.2, normalize=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Frame[{i_frame}]')
        plt.show()
