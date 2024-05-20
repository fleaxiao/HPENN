# Import necessary packages
import pandas as pd
import numpy as np

def main():
    adr = 'dataset_2D/Trainset_5w_IW_paper.csv'
    df = pd.read_csv(adr, header=None)
    data_length = 50_000
   
    inputs = df.iloc[:12, 0:data_length].values
    L_real = df.iloc[24:36, 0:data_length].values
    L_real = np.sum(L_real, axis=0)

    Np = inputs[0]
    Ns = inputs[1]
    hw1 = inputs[2]
    hw2 = inputs[3]
    dww_i_x = inputs[4]
    dww_o_x = inputs[5]
    dww_ii_x = inputs[6]
    dww_oo_x = inputs[7]
    dww_x = inputs[8]
    lcore_x1 = inputs[9]
    hw = inputs[10]
    dw = inputs[11]

    permeability = 1.256629e-6
    conductivity = 5.96e7
    skin_depth = np.sqrt(2/(2*3.14*10e3*permeability*conductivity))
    delta_1 = np.sqrt(hw1/hw)*dww_i_x / skin_depth
    delta_2 = np.sqrt(hw2/hw)*dww_o_x / skin_depth
    factor_1_1 = (np.sinh(2*delta_1)-np.sin(2*delta_1)) / (np.cosh(2*delta_1)-np.cos(2*delta_1))
    factor_1_2 = (np.sinh(delta_1)-np.sin(delta_1)) / (np.cosh(delta_1)-np.cos(delta_1))
    factor_2_1 = (np.sinh(2*delta_2)-np.sin(2*delta_2)) / (np.cosh(2*delta_2)-np.cos(2*delta_2))
    factor_2_2 = (np.sinh(delta_2)-np.sin(delta_2)) / (np.cosh(delta_2)-np.cos(delta_2))

    F_L_1 = 1/(2*Ns**2*delta_1)*((4*Ns**2-1)*factor_1_1-2*(Ns**2-1)*factor_1_2)
    F_L_2 = 1/(2*Np**2*delta_2)*((4*Np**2-1)*factor_2_1-2*(Np**2-1)*factor_2_2)

    L = permeability*Ns**2/hw*((dww_i_x*Ns/3*F_L_1+dww_ii_x*(Ns-1)/(2*Ns))+(dww_o_x*Np/3*F_L_2+dww_oo_x*(Np-1)/(2*Np))+dww_x)
    L = L.reshape(1, -1)
    L_real = L_real.reshape(1, -1)
    F_L_1 = F_L_1.reshape(1, -1)
    F_L_2 = F_L_2.reshape(1, -1)
    coef = (L_real / L)

    coef = np.concatenate((inputs, coef), axis=0)
    np.savetxt("inductor/dataset_coef/dataset_IW_coef.csv", coef, delimiter=',')

    print(np.max(coef,axis=1))
    print(np.min(coef,axis=1))

  
if __name__ == "__main__":
    main()