# Import necessary packages
import random
import pandas as pd
import numpy as np

def main():
    adr = 'dataset_2D/trainset_IW_5w_4.0.csv'
    df = pd.read_csv(adr, header=None)
    data_length = 50_000
   
    inputs = df.iloc[:10, 0:data_length].values #! IW -> 10, OW -> 8
    R_real_1 = df.iloc[10:16, 0:data_length].values
    R_real_2 = df.iloc[16:22, 0:data_length].values

    Np = inputs[0]
    Ns = inputs[1]
    hw1 = inputs[2]
    hw2 = inputs[3]
    dww_ii_x = inputs[4]
    dww_oo_x = inputs[5]
    dww_x = inputs[6]
    lcore_x1 = inputs[7]
    hw = inputs[8]
    dw = inputs[9]

    permeability = 1.256629e-6
    conductivity = 5.96e7
    skin_depth = np.sqrt(2/(2*3.14*1e3*permeability*conductivity))
    delta_1 = np.sqrt(hw1 / hw) * 1e-3 / skin_depth
    delta_2 = np.sqrt(hw2 / hw) * 1e-3 / skin_depth
    factor_1_1 = (np.sinh(2*delta_1)+np.sin(2*delta_1)) / (np.cosh(2*delta_1)-np.cos(2*delta_1))
    factor_1_2 = (np.sinh(delta_1)-np.sin(delta_1)) / (np.cosh(delta_1)+np.cos(delta_1))
    factor_2_1 = (np.sinh(2*delta_2)+np.sin(2*delta_2)) / (np.cosh(2*delta_2)-np.cos(2*delta_2))
    factor_2_2 = (np.sinh(delta_2)-np.sin(delta_2)) / (np.cosh(delta_2)+np.cos(delta_2))

    R_dc_1 = 1 / (conductivity*1e-3*hw1)
    R_dc_2 = 1 / (conductivity*1e-3*hw2)

    R_ac_1 = np.zeros((6, data_length))
    R_ac_2 = np.zeros((6, data_length))
    coef_1 = np.zeros((6, data_length))
    coef_2 = np.zeros((6, data_length))

    for i in range(0,6):
        R_ac_1[i] = R_dc_1*delta_1*(factor_1_1+2*(i+1)*i*factor_1_2)
        coef_1[i] = np.where(i >= Ns, 0, R_real_1[i] / R_ac_1[i])

    for i in range(0,6):
        R_ac_2[i] = R_dc_2*delta_2*(factor_2_1+2*(i+1)*i*factor_2_2)
        coef_2[i] = np.where(i >= Np, 0, R_real_2[i] / R_ac_2[i])

    coef = np.concatenate((inputs, coef_1, coef_2), axis=0)
    np.savetxt("dataset_2D/dataset_IW_coef.csv", coef, delimiter=',')
   
if __name__ == "__main__":
    main()