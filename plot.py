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

    for i in range(1,7):
        Rac_i, Llk_i = get_dataset(f'results_IW/train_error_IW_{i}.csv')
        Rac_i = np.pad(Rac_i, ((0, 10000 - len(Rac_i)), (0, 0)), mode='constant', constant_values=0)
        Llk_i = np.pad(Rac_i, ((0, 10000 - len(Rac_i)), (0, 0)), mode='constant', constant_values=0)
        Rac.append(Rac_i)
        Llk.append(Llk_i)

    Rac = np.concatenate(Rac, axis=1)
    Llk = np.concatenate(Llk, axis=1)

    # Visualization
    Error_Rac_Ls = 0
    Error_Rac_Lp = 0
     
    colors = plt.cm.viridis(np.linspace(0, 1, Rac.shape[1]))
    bindwidth = 1e2

    plt.figure(figsize=(8, 5))
    for i in range (int(Rac.shape[1])):
        plt.hist(Rac[:,i], bins=18, density=True, alpha=0.6, color=colors[i], edgecolor='black')
        Error_Rac_Ls += np.sum(Rac[:,i] > 5)
    plt.title('Rac Error Distribution in Ls Winding')
    plt.xlabel('Error(%)')
    plt.ylabel('Distribution')
    plt.legend(labels=['Ls_layer_1','Ls_layer_2','Ls_layer_3','Ls_layer_4','Ls_layer_5','Ls_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Rac_Ls.png',dpi=600)

    print("figure saved!")

if __name__ == "__main__":
    main()