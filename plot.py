import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    Rac = df.iloc[:, 0].values.reshape(-1,1)
   
    return Rac

def main():

    # Rac = []

    # for i in range(1,7):
    #     Rac_i = get_dataset(f'results_OW/train_error_OW_{i}.csv')
    #     Rac_i = np.pad(Rac_i, ((0, 10000 - len(Rac_i)), (0, 0)), mode='constant', constant_values=0)
    #     Rac.append(Rac_i)

    # Rac = np.concatenate(Rac, axis=1)
    # Rac_avg = np.mean(Rac, axis=0)
    # Rac_rms = np.sqrt(np.mean(Rac ** 2,axis=0))
    # Rac_max = np.max(Rac, axis=0)
    # print(f"Rac_avg: {Rac_avg}")
    # print(f"Rac_rms: {Rac_rms}")
    # print(f"Rac_max: {Rac_max}")

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # colors = plt.cm.viridis(np.linspace(0, 1, Rac.shape[1]))
    # weights = np.ones_like(Rac[:,1])/float(len(Rac[:,1]))
    # plt.figure(figsize=(6, 4))
    # for i in range (int(Rac.shape[1])):
    #     plt.hist(Rac[:,i], bins=np.arange(0,3,0.1), alpha=0.4, color=colors[i], weights=weights, edgecolor='black')
    # plt.title('Rac Error Distribution of Lp in OW Section')
    # plt.xlabel('Error(%)')
    # plt.xlim(0, 3)
    # plt.ylabel('Distribution')
    # plt.legend(labels=['Ls_layer_1','Ls_layer_2','Ls_layer_3','Ls_layer_4','Ls_layer_5','Ls_layer_6'])
    # plt.grid()
    # plt.savefig('figs/Fig_Rac_OW_Ls.png',dpi=600)

    Llk = get_dataset('results_inductor/train_error_OW_inside.csv')

    Llk_avg = np.mean(Llk, axis=0)
    Llk_rms = np.sqrt(np.mean(Llk ** 2,axis=0))
    Llk_max = np.max(Llk, axis=0)
    print(f"Rac_avg: {Llk_avg}")
    print(f"Rac_rms: {Llk_rms}")
    print(f"Rac_max: {Llk_max}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    weights = np.ones_like(Llk)/float(len(Llk))
    plt.figure(figsize=(6, 4))
    plt.hist(Llk, bins=np.arange(0,3,0.1), alpha=0.4, color=colors[3], weights=weights, edgecolor='black')
    plt.title('Llk Error Distribution of Ls in OW Section')
    plt.xlabel('Error(%)')
    plt.xlim(0, 3)
    plt.ylabel('Distribution')
    plt.grid()
    plt.savefig('figs/Fig_Llk_OW_Ls.png',dpi=600)


    print("figure saved!")

if __name__ == "__main__":
    main()