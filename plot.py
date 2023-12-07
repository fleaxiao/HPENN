import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    Rac = df.iloc[:, 0].values.reshape(-1,1)
    Llk = df.iloc[:, 1].values.reshape(-1,1)
   
    return Rac, Llk

def main():

    Rac = []
    Llk = []

    for i in range(7,13):
        Rac_i, Llk_i = get_dataset(f'results_IW/train_error_IW_{i}.csv')
        Rac_i = np.pad(Rac_i, ((0, 10000 - len(Rac_i)), (0, 0)), mode='constant', constant_values=0)
        Llk_i = np.pad(Llk_i, ((0, 10000 - len(Llk_i)), (0, 0)), mode='constant', constant_values=0)
        Rac.append(Rac_i)
        Llk.append(Llk_i)

    Rac = np.concatenate(Rac, axis=1)
    Rac_avg = np.mean(Rac, axis=0)
    Rac_rms = np.sqrt(np.mean(Rac ** 2,axis=0))
    Rac_max = np.max(Rac, axis=0)
    print(f"Rac_avg: {Rac_avg}")
    print(f"Rac_rms: {Rac_rms}")
    print(f"Rac_max: {Rac_max}")

    Llk = np.concatenate(Llk, axis=1)
    Llk_avg = np.mean(Llk, axis=0)
    Llk_rms = np.sqrt(np.mean(Llk ** 2,axis=0))
    Llk_max = np.max(Llk, axis=0)
    print(f"Llk_avg: {Llk_avg}")
    print(f"Llk_rms: {Llk_rms}")
    print(f"Llk_max: {Llk_max}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    colors = plt.cm.viridis(np.linspace(0, 1, Rac.shape[1]))

    plt.figure(figsize=(6, 4))
    for i in range (int(Rac.shape[1])):
        plt.hist(Rac[:,i], bins=18, alpha=0.6, color=colors[i], edgecolor='black')
    plt.title('Rac Error Distribution in Lp Winding')
    plt.xlabel('Error(%)')
    plt.xlim(0, 6)
    plt.ylabel('Distribution')
    plt.legend(labels=['Ls_layer_1','Ls_layer_2','Ls_layer_3','Ls_layer_4','Ls_layer_5','Ls_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Rac_Lp.png',dpi=600)

    plt.figure(figsize=(6, 4))
    for i in range (int(Llk.shape[1])):
        plt.hist(Llk[:,i], bins=18, alpha=0.6, color=colors[i], edgecolor='black')
    plt.title('Llk Error Distribution in Lp Winding')
    plt.xlabel('Error(%)')
    plt.xlim(0, 6)
    plt.ylabel('Distribution')
    plt.legend(labels=['Ls_layer_1','Ls_layer_2','Ls_layer_3','Ls_layer_4','Ls_layer_5','Ls_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Llk_Lp.png',dpi=600)

    print("figure saved!")

if __name__ == "__main__":
    main()