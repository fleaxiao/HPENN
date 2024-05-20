import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    Rac = df.iloc[:, 0].values.reshape(-1,1)
   
    return Rac

def main():

    Rac = []

    for i in range(1,7):
        Rac_i = get_dataset(f'results_loss/results_OW/train_error_OW_{i}.csv')
        Rac_i = np.pad(Rac_i, ((0, 10000 - len(Rac_i)), (0, 0)), mode='constant', constant_values=0)
        Rac.append(Rac_i)

    Rac = np.concatenate(Rac, axis=1)
    Rac_avg = np.mean(Rac, axis=0)
    Rac_rms = np.sqrt(np.mean(Rac ** 2,axis=0))
    Rac_max = np.max(Rac, axis=0)
    print(f"Rac_avg: {Rac_avg}")
    print(f"Rac_rms: {Rac_rms}")
    print(f"Rac_max: {Rac_max}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    colors = plt.cm.viridis(np.linspace(0, 1, Rac.shape[1]))
    weights = np.ones_like(Rac[:,1])/float(len(Rac[:,1]))
    plt.figure(figsize=(6, 4))
    for i in range (int(Rac.shape[1])):
        plt.hist(Rac[:,i], bins=np.arange(0,4,0.1), alpha=0.4, color=colors[i], weights=weights, edgecolor='black')
    # plt.title('Rac Error Distribution of Lp in OW Section')
    plt.xlabel('Error(%)', fontsize=23)
    plt.ylabel('Distribution', fontsize=23)
    plt.xlim(0, 4)
    plt.xticks(size = 22)
    plt.yticks(size = 22)
    y = MultipleLocator(0.2)
    x = MultipleLocator(0.5) 
    ax = plt.gca()
    ax.yaxis.set_major_locator(y)
    ax.xaxis.set_major_locator(x)
    plt.legend(labels=['Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Layer 6'], fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig('figs/Fig_Rac_OW_Ls.png',dpi=200)

    # Llk = get_dataset('results_inductor/train_error_OW_outside.csv')

    # Llk_avg = np.mean(Llk, axis=0)
    # Llk_rms = np.sqrt(np.mean(Llk ** 2,axis=0))
    # Llk_max = np.max(Llk, axis=0)
    # print(f"Rac_avg: {Llk_avg}")
    # print(f"Rac_rms: {Llk_rms}")
    # print(f"Rac_max: {Llk_max}")

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # colors = plt.cm.viridis(np.linspace(0, 1, 4))
    # weights = np.ones_like(Llk)/float(len(Llk))
    # plt.figure(figsize=(6, 4))
    # plt.hist(Llk, bins=np.arange(0,5,0.2), alpha=0.4, color=colors[2], weights=weights, edgecolor='black')
    # # plt.title('Llk Error Distribution of Ls in OW Section')
    # plt.xlabel('Error(%)', fontsize=23)
    # plt.ylabel('Distribution', fontsize=23)
    # plt.xlim(0, 5)
    # plt.xticks(size = 22)
    # plt.yticks(size = 22)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('figs/Fig_Llk_OW_Lp.png',dpi=200)

    print("figure saved!")

if __name__ == "__main__":
    main()