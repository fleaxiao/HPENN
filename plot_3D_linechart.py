import numpy as np
import matplotlib.pyplot as plt

def main():

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    x = np.array([1, 10000])
    y1 = np.array([0.034, 4.71])
    y2 = np.array([0.003, 0.368])
    y3 = np.array([322, 322*1e4])
    plt.figure(figsize=(6, 4))
    plt.plot(x, y1, marker='o', color='green', markersize=8, label='HPINN Framework')
    plt.plot(x, y2, marker='^', color='orange', markersize=8, label='NN Model')
    plt.plot(x, y3, marker='s', markersize=8, label='FEM Simulation')

    plt.ylim(1e-3, 1e10)
    plt.yscale('log')
    plt.xlabel('Number of cases', fontsize=16)
    plt.ylabel('Process time (s)', fontsize=16)
    plt.xticks(size = 15)
    plt.yticks(size = 15)

    plt.grid()
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig('figs/Fig_linechart_speed.png',dpi=200)
    plt.show()

if __name__ == "__main__":
    main()