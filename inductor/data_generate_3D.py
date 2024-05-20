import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Neural Network Structure
output_size = 1
hidden_layers = 3

# Define model structures and functions
class Net(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
   
        # Input layer
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()] 

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CombinedModel(nn.Module):
    def __init__(self, net1, net2, net3, net4, net5, net6):
        super(CombinedModel, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.net4 = net4
        self.net5 = net5
        self.net6 = net6

    def forward(self, x):
        output1 = self.net1(x)
        output2 = self.net2(x)
        output3 = self.net3(x)
        output4 = self.net4(x)
        output5 = self.net5(x)
        output6 = self.net6(x)
        return torch.cat((output1, output2, output3, output4, output5, output6), dim = 1)


def get_dataset(adr, Np, Ns):
    df = pd.read_csv(adr, skiprows=5, header=None)
    data_length, _ = df.shape
   
    inputs = df.iloc[0:data_length, 0:21].values / 1e3
    hw1 = inputs[0:,0].reshape(-1,1)
    hw2 = inputs[0:,1].reshape(-1,1)
    dw1 = inputs[0:,2].reshape(-1,1)
    dw2 = inputs[0:,3].reshape(-1,1)
    lcs = inputs[0:,4].reshape(-1,1)
    dcs = inputs[0:,5].reshape(-1,1)
    dww_ii_x = inputs[0:,6].reshape(-1,1)
    dww_oo_x = inputs[0:,7].reshape(-1,1)
    dww_x = inputs[0:,8].reshape(-1,1)
    lcore_x1 = inputs[0:,9].reshape(-1,1)
    lcore_x2 = inputs[0:,10].reshape(-1,1)
    lcore_y1 = inputs[0:,11].reshape(-1,1)
    lcore_y2 = inputs[0:,12].reshape(-1,1)

    hw = np.maximum(hw1+2*lcore_y1, hw2+2*lcore_y2)
    dw = lcore_x1 + Ns*dw1 + (Ns-1)*dww_ii_x + dww_x + Np*dw2 + (Np-1)*dww_oo_x + lcore_x2
    inputs = np.concatenate((Np*np.ones_like(hw1), Ns*np.ones_like(hw1), hw1, hw2, dw1, dw2, dww_ii_x, dww_oo_x, dww_x, lcore_x1, hw, dw, lcs, dcs), axis = 1) 

    outputs = df.iloc[0:data_length:, 21:].values
    Rac = np.sum(outputs[0:,0:12], axis=1).reshape(-1,1) 
    Llk = np.sum(outputs[0:,12:24], axis=1).reshape(-1,1)

    # Dowell's equation
    permeability = 1.256629e-6
    conductivity = 5.96e7
    skin_depth = np.sqrt(2/(2*3.14*10e3*permeability*conductivity))
    delta_1 = np.sqrt(hw1/hw)*dw1 / skin_depth
    delta_2 = np.sqrt(hw2/hw)*dw2 / skin_depth
    factor_1_1 = (np.sinh(2*delta_1)-np.sin(2*delta_1)) / (np.cosh(2*delta_1)-np.cos(2*delta_1))
    factor_1_2 = (np.sinh(delta_1)-np.sin(delta_1)) / (np.cosh(delta_1)-np.cos(delta_1))
    factor_2_1 = (np.sinh(2*delta_2)-np.sin(2*delta_2)) / (np.cosh(2*delta_2)-np.cos(2*delta_2))
    factor_2_2 = (np.sinh(delta_2)-np.sin(delta_2)) / (np.cosh(delta_2)-np.cos(delta_2))

    F_L_1 = 1/(2*Ns**2*delta_1)*((4*Ns**2-1)*factor_1_1-2*(Ns**2-1)*factor_1_2)
    F_L_2 = 1/(2*Np**2*delta_2)*((4*Np**2-1)*factor_2_1-2*(Np**2-1)*factor_2_2)

    L = permeability*Ns**2/hw*((dw1*Ns/3*F_L_1+dww_ii_x*(Ns-1)/(2*Ns))+(dw2*Np/3*F_L_2+dww_oo_x*(Np-1)/(2*Np))+dww_x)

    return inputs, Rac, Llk, L

def preprocess(inputs, Llk, Rac):
    # preprocess
    inputs_pre = np.copy(inputs)
    Llk_pre = np.copy(Llk)
    Rac_pre = np.copy(Rac)
    inputs_pre[0:,:2] = inputs_pre[0:,:2]/10
    inputs_pre[0:,2:] = inputs_pre[0:,2:]

    # log tranfer
    inputs_pre = np.log10(inputs_pre)
    Llk_pre = np.log10(Llk_pre)
    Rac_pre = np.log10(Rac_pre)

    # normalization
    inputs_pre[0:,0] = (inputs_pre[0:,0] - np.log10(3e-1)) / (np.log10(6e-1) - np.log10(3e-1))
    inputs_pre[0:,1] = (inputs_pre[0:,1] - np.log10(3e-1)) / (np.log10(6e-1) - np.log10(3e-1))
    inputs_pre[0:,2] = (inputs_pre[0:,2] - np.log10(2e-2)) / (np.log10(7e-2) - np.log10(2e-2))
    inputs_pre[0:,3] = (inputs_pre[0:,3] - np.log10(1e-2)) / (np.log10(7e-2) - np.log10(1e-2))
    inputs_pre[0:,4] = (inputs_pre[0:,4] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs_pre[0:,5] = (inputs_pre[0:,5] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs_pre[0:,6] = (inputs_pre[0:,6] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs_pre[0:,7] = (inputs_pre[0:,7] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs_pre[0:,8] = (inputs_pre[0:,8] - np.log10(5e-4)) / (np.log10(6e-3) - np.log10(5e-4))
    inputs_pre[0:,9] = (inputs_pre[0:,9] - np.log10(1e-3)) / (np.log10(4e-3) - np.log10(1e-3))
    inputs_pre[0:,10] = (inputs_pre[0:,10] - np.log10(2.2e-2)) / (np.log10(8.2e-2) - np.log10(2.2e-2))
    inputs_pre[0:,11] = (inputs_pre[0:,11] - np.log10(5.5e-3)) / (np.log10(2.26e-2) - np.log10(5.5e-3))

    # tensor transfer
    input_tensor_IW = torch.tensor(inputs_pre[0:,0:12], dtype=torch.float32)
    input_tensor_OW = torch.tensor(inputs_pre[0:,0:11], dtype=torch.float32)

    Llk_tensor = torch.tensor(Llk_pre, dtype=torch.float32)
    Rac_tensor = torch.tensor(Rac_pre, dtype=torch.float32)

    return torch.utils.data.TensorDataset(input_tensor_IW, Llk_tensor, Rac_tensor), torch.utils.data.TensorDataset(input_tensor_OW, Llk_tensor, Rac_tensor)

def get_inductor_model_output(model, device, data_loader):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((-0.6) - (0.7)) + (0.7)
    yy_pred = 10**yy_pred

    return yy_pred

def main():

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        kwargs = {'num_workers': 0, 'pin_memory': True}
        print("Now this program runs on cpu")
 
    # Load dataset       
    folder_path = 'dataset_3D'
    pattern = re.compile(r'results_(\d+)_(\d+)_(\d+)\.csv')
    inputs = np.array([]).reshape(-1,14)
    Rac = np.array([]).reshape(-1,1)
    Llk = np.array([]).reshape(-1,1)
    L_dowell = np.array([]).reshape(-1,1)

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            Ns = int(match.group(1)) 
            Np = int(match.group(2)) 

            file_path = os.path.join(folder_path, filename)
            inputs_i, Rac_i, Llk_i, L_dowell_i = get_dataset(file_path, Np, Ns)

        inputs = np.concatenate((inputs, inputs_i), axis = 0)
        Rac = np.concatenate((Rac, Rac_i), axis = 0)
        Llk = np.concatenate((Llk, Llk_i), axis = 0)
        L_dowell = np.concatenate((L_dowell, L_dowell_i), axis = 0)

    # Preprocess
    dataset_IW, dataset_OW = preprocess(inputs, Llk, Rac)
    data_loader_IW = torch.utils.data.DataLoader(dataset_IW, batch_size=4, shuffle=False, **kwargs)
    data_loader_OW = torch.utils.data.DataLoader(dataset_OW, batch_size=4, shuffle=False, **kwargs)

    # Load inductor model
    # IW
    input_size = 12
    hidden_size = 100
    model_inductor_IW = Net(input_size, hidden_size).to(device)
    model_inductor_IW.load_state_dict(torch.load('inductor/results_inductor/Model_2D_IW.pth', map_location = torch.device('cpu')))

    # OW
    input_size = 11
    hidden_size = 100
    model_inductor_OW = Net(input_size, hidden_size).to(device)
    model_inductor_OW.load_state_dict(torch.load('inductor/results_inductor/Model_2D_OW.pth', map_location = torch.device('cpu')))

    # Output 2D data    
    model_inductor_IW.eval()
    model_inductor_OW.eval()

    inductor_IW = get_inductor_model_output(model_inductor_IW, device, data_loader_IW)*L_dowell
    inductor_OW = get_inductor_model_output(model_inductor_OW, device, data_loader_OW)*L_dowell

    # Calculate section and corner loss/inductance
    section_inductor_IW = inductor_IW.T*inputs[:,12]
    section_inductor_OW = inductor_OW.T*inputs[:,13]*2

    corner_radius_Ls = np.zeros((np.shape(inputs)[0], 6))
    corner_radius_Lp = np.zeros((np.shape(inputs)[0], 6))
    length_Ls = np.zeros((np.shape(inputs)[0], 6))
    length_Lp = np.zeros((np.shape(inputs)[0], 6))

    for i in range(0, np.shape(inputs)[0]):
        for j in range(0,6):
            if j < inputs[i,1]:
                corner_radius_Ls[i,j] = inputs[i,9] + j*(inputs[i,6] + inputs[i,4]) + inputs[i,4]/2
                length_Ls[i,j] = 2*inputs[i,12] + 4*inputs[i,13] + 2*3.14*corner_radius_Ls[i,j] 
            else:
                corner_radius_Ls[i,j] = 0
                length_Ls[i,j] = 0
    
    for i in range(0, np.shape(inputs)[0]):
        for j in range(0,6):
            if j < inputs[i,0]:
                corner_radius_Lp[i,j] = inputs[i,9] + j*(inputs[i,7] + inputs[i,5]) + inputs[i,5]/2 + (inputs[i,1]-1)*(inputs[i,6] + inputs[i,4]) + inputs[i,4] + inputs[i,8]
                length_Lp[i,j] = 2*inputs[i,12] + 4*inputs[i,13] + 2*3.14*corner_radius_Lp[i,j] 
            else:
                corner_radius_Lp[i,j] = 0
                length_Lp[i,j] = 0

    corner_inductor_IW = 2*3.14*inductor_IW.T*(np.mean(corner_radius_Ls, axis=1) + np.mean(corner_radius_Lp, axis=1))/2
    corner_inductor_OW = 2*3.14*inductor_OW.T*(np.mean(corner_radius_Ls, axis=1) + np.mean(corner_radius_Lp, axis=1))/2
 
    # Obtain correction factor
    inductance = np.squeeze(Llk.reshape(1,-1))
    rest_inductor = inductance/2 - (section_inductor_IW + section_inductor_OW)
    coef_inductor = rest_inductor / ((section_inductor_IW + section_inductor_OW) + (corner_inductor_IW + corner_inductor_OW)/2)
    print(f"inductor max corf: {np.max(coef_inductor)}")
    print(f"inductor min corf: {np.min(coef_inductor)}")

    # Save data
    save_data_inductor = np.vstack([inputs.T, inductance, section_inductor_IW, section_inductor_OW, corner_inductor_IW, corner_inductor_OW, coef_inductor])
    save_data_inductor = save_data_inductor.T
    np.random.shuffle(save_data_inductor)
    save_data_inductor = save_data_inductor.T
    np.savetxt("inductor/dataset_coef/dataset_3D_inductor.csv", save_data_inductor, delimiter=',')

if __name__ == "__main__":
    main()