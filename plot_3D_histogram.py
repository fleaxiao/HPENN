import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    n = df.iloc[0].values.reshape(-1,1)
   
    return n

def main():

    n = []
    i = 0

    if i == 0:
        name = 'C'
        color = 'darkorange'
    elif i == 1:
        name = 'P'
        color = 'forestgreen'

    n_i = get_dataset(f'results_coef/train_3D_error_{name}.csv')
    n.append(n_i)

    n = np.concatenate(n, axis=1)
    n_avg = np.mean(n, axis=0)
    n_rms = np.sqrt(np.mean(n ** 2,axis=0))
    n_max = np.max(n, axis=0)
    print(f"n_avg: {n_avg}")
    print(f"n_rms: {n_rms}")
    print(f"n_max: {n_max}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    weights = np.ones_like(n[:,0])/float(len(n[:,0]))
    plt.figure(figsize=(5.5, 4))
    for i in range (int(n.shape[1])):
        plt.hist(n[:,i], bins=np.arange(0,30,2), alpha=0.8, color=color, weights=weights, edgecolor='black')
    # plt.title('n Error Dinameibution of Lp in OW Section')
    plt.xlabel('Error(%)', fontsize=23)
    plt.ylabel('Distribution', fontsize=23)
    plt.xlim(0, 20)
    plt.xticks(size = 22)
    plt.yticks(size = 22)
    y = MultipleLocator(0.1)
    x = MultipleLocator(5) 
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    # plt.legend(labels=['C','P'], fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figs/Fig_3D_error_{name}.png',dpi=200)

    print("figure saved!")

if __name__ == "__main__":
    main()