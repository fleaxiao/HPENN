import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def main():

    # Accuracy
    data = np.array([4.56, 4.2681, 4.22673, 4.39868, 3.9727, 3.56439])
    labels = ['Exp.','FEM', 'HPINN', 'PINN', 'NN', 'Anal.']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(data)), data, color=['steelblue', 'lightblue', 'green', 'forestgreen', 'silver', 'silver'], edgecolor='black', alpha = 0.8, tick_label=labels, width=0.6, zorder=3)

    plt.xlabel('Method', fontsize=24)
    plt.ylabel('Resistance (m\u2126)', fontsize=24)
    plt.xticks(size = 22)
    plt.yticks(size = 22)
    y = MultipleLocator(1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)

    # plt.ylim(0, 15)

    plt.grid(zorder=0)
    plt.tight_layout()
    plt.savefig('figs/Fig_bar_accuracy.png',dpi=200)

    print("figure saved!")
    plt.show()

    # # Compute Burden
    # data_3D = [265, 353, 1765]
    # data_2D = [38, 0, 0]
    # labels = ['HPINN', 'NN\n(200 data)', 'NN\n(1000 data)']

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.figure(figsize=(6, 4))
    # plt.bar(range(len(data_3D)), data_3D, color='orange', edgecolor='black', tick_label=labels, label='3D Data', width=0.6, zorder=3, alpha=0.8)
    # plt.bar(range(len(data_2D)), data_2D, color='gold', edgecolor='black', tick_label=labels, label='2D Data', width=0.6, zorder=3, bottom=data_3D, alpha=0.8)

    # plt.ylim(0, 2000)
    # plt.xlabel('Method', fontsize=18)
    # plt.ylabel('SBU', fontsize=18)
    # plt.xticks(size = 16)
    # plt.yticks(size = 16)
    # y = MultipleLocator(400)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y)

    # plt.grid(zorder=0)
    # plt.legend(fontsize=18)
    # plt.tight_layout()
    # plt.savefig('figs/Fig_bar_burden.png',dpi=200)

    # print("figure saved!")
    # plt.show()

if __name__ == "__main__":
    main()