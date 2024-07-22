import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def main():

    # # Accuracy
    # loss = np.array([4.66, 4.3681, 4.33065, 3.4299, 3.56439])
    # inductance = np.array([474, 468, 464, 513, 629])
    # labels = ['Exp.','FEM', 'HPINN', 'NN', 'Anal.']

    # # Loss
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(len(loss)), loss, color=['steelblue', 'lightblue', 'forestgreen', 'silver', 'silver'], edgecolor='black', alpha = 0.8, tick_label=labels, width=0.6, zorder=3)

    # plt.xlabel('Method', fontsize=24)
    # plt.ylabel('Resistance (m\u2126)', fontsize=24)
    # plt.xticks(size = 22)
    # plt.yticks(size = 22)
    # y = MultipleLocator(1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y)

    # plt.grid(zorder=0)
    # plt.tight_layout()
    # plt.savefig('figs/Fig_bar_accuracy_Loss.png',dpi=200)

    # print("figure saved!")
    # plt.show()

    # # Inductance
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(len(inductance)), inductance, color=['steelblue', 'lightblue', 'forestgreen', 'silver', 'silver'], edgecolor='black', alpha = 0.8, tick_label=labels, width=0.6, zorder=3)

    # plt.xlabel('Method', fontsize=24)
    # plt.ylabel('Inductance (nH)', fontsize=24)
    # plt.xticks(size = 22)
    # plt.yticks(size = 22)
    # y = MultipleLocator(100)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y)

    # plt.grid(zorder=0)
    # plt.tight_layout()
    # plt.savefig('figs/Fig_bar_accuracy_Inductance.png',dpi=200)

    # print("figure saved!")
    # plt.show()

    # Compute Burden
    loss_3D = [548, 822, 8220]
    loss_2D = [225, 0, 0]
    labels = ['HPINN\n50 3D data + 18000 2D data', 'NN\n75 3D data', 'NN\n750 3D data']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(loss_3D)), loss_3D, color='darkorange', edgecolor='black', tick_label=labels, label='3D Data', width=0.6, zorder=3, alpha=0.8)
    plt.bar(range(len(loss_2D)), loss_2D, color='gold', edgecolor='black', tick_label=labels, label='2D Dara', width=0.6, zorder=3, bottom=loss_3D, alpha=0.8)

    plt.ylim(0, 10000)
    plt.xlabel('Method', fontsize=18)
    plt.ylabel('SBU', fontsize=18)
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    y = MultipleLocator(2000)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)

    plt.grid(zorder=0)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig('figs/Fig_bar_burden.png',dpi=200)

    print("figure saved!")
    plt.show()

if __name__ == "__main__":
    main()