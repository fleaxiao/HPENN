import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Neural Network Structure
output_size = 1
hidden_layers = 4 

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
        return output1, output2, output3, output4, output5, output6


def get_dataset(adr, Np, Ns):
    df = pd.read_csv(adr, skiprows=5, header=None)
    data_length = 50_000
   
    inputs = df.iloc[0:data_length, 0:21].values
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
    dc2c = inputs[0:,13].reshape(-1,1)

    hw = np.maximum(hw1+2*lcore_y1, hw2+2*lcore_y2)
    dw = lcore_x1 + Ns*dw1 + (Ns-1)*dww_ii_x + dww_x + Np*dw2 + (Np-1)*dww_oo_x + lcore_x2
    inputs = np.concatenate((Np*np.ones_like(hw1), Ns*np.ones_like(hw1), hw1, hw2, dww_ii_x, dww_oo_x, dww_x, lcore_x1, hw, dw), axis = 1) 

    outputs = df.iloc[0:data_length:, 21:].values
    Llk = np.sum(outputs[0:,0:12], axis=1).reshape(-1,1)
    Rac = np.sum(outputs[0:,12:24], axis=1).reshape(-1,1)
        
    return inputs, Llk, Rac

def preprocess(inputs, Llk, Rac):
    inputs[0:,:2] = inputs[0:,:2]/10

    # log tranfer
    inputs = np.log10(inputs)
    Llk = np.log10(Llk)
    Rac = np.log10(Rac)

    # normalization
    inputs_max = np.max(inputs, axis=0, keepdims=True)
    inputs_min = np.min(inputs, axis=0, keepdims=True)
    diff = inputs_max - inputs_min
    diff[diff == 0] = 1
    inputs = np.where(diff == 1, 1, (inputs - inputs_min) / diff)

    # tensor transfer
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    Llk_tensor = torch.tensor(Llk, dtype=torch.float32)
    Rac_tensor = torch.tensor(Rac, dtype=torch.float32)

    return input_tensor, Llk_tensor, Rac_tensor

def load_model(input_size, hidden_size_sequence, device):
    net1 = Net(input_size, hidden_size_sequence[0]).to(device)
    net2 = Net(input_size, hidden_size_sequence[1]).to(device)
    net3 = Net(input_size, hidden_size_sequence[2]).to(device)
    net4 = Net(input_size, hidden_size_sequence[3]).to(device)
    net5 = Net(input_size, hidden_size_sequence[4]).to(device)
    net6 = Net(input_size, hidden_size_sequence[5]).to(device)
    model = CombinedModel(net1, net2, net3, net4, net5, net6)

    return model

def main():

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        print("Now this program runs on cpu")
 
    # Load dataset       
    folder_path = 'dataset_3D'
    pattern = re.compile(r'results_(\d+)_(\d+)_(\d+)\.csv')
    inputs = np.array([]).reshape(-1,10)
    Llk = np.array([]).reshape(-1,1)
    Rac = np.array([]).reshape(-1,1)

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            Np = int(match.group(1))
            Ns = int(match.group(2))

            file_path = os.path.join(folder_path, filename)
            inputs_i, Llk_i, Rac_i = get_dataset(file_path, Np, Ns)

        inputs = np.concatenate((inputs, inputs_i), axis = 0)
        Llk = np.concatenate((Llk, Llk_i), axis = 0)
        Rac = np.concatenate((Rac, Rac_i), axis = 0)

    # Preprocess
    input_tensor, Llk_tensor, Rac_tensor = preprocess(inputs, Llk, Rac)
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}

    # IW Ls Load model
    input_size = 10
    hidden_size_sequence = [100, 100, 100, 134, 127, 117]
    model_IW_Ls = load_model(input_size, hidden_size_sequence, device)
    model_IW_Ls.load_state_dict(torch.load('Model_2D_IW_Ls.pth', map_location = torch.device('cpu')))

    # IW Lp Load model
    input_size = 10
    hidden_size_sequence = [100, 143, 134, 143, 140, 143]
    model_IW_Lp = load_model(input_size, hidden_size_sequence, device)
    model_IW_Lp.load_state_dict(torch.load('Model_2D_IW_Lp.pth', map_location = torch.device('cpu')))
 
    # OW Ls Load model
    input_size = 8
    hidden_size_sequence = [100, 100, 100, 100, 128, 139]
    model_OW_Ls = load_model(input_size, hidden_size_sequence, device)
    model_OW_Ls.load_state_dict(torch.load('Model_2D_OW_Ls.pth', map_location = torch.device('cpu')))

    # OW Lp Load model
    input_size = 8
    hidden_size_sequence = [81, 109, 123, 81, 81, 125]
    model_OW_Lp = load_model(input_size, hidden_size_sequence, device)
    model_OW_Lp.load_state_dict(torch.load('Model_2D_OW_Lp.pth', map_location = torch.device('cpu')))

    # # Evaluation
    # net.eval()
    # x_meas = []
    # y_meas = []
    # y_pred = []
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         y_pred.append(net(inputs.to(device)))
    #         y_meas.append(labels.to(device))
    #         x_meas.append(inputs)

    # y_meas = torch.cat(y_meas, dim=0)
    # y_pred = torch.cat(y_pred, dim=0)
    
    # # tensor is transferred to numpy
    # yy_pred = y_pred.cpu().numpy()
    # yy_meas = y_meas.cpu().numpy()

    # yy_pred = yy_pred * (test_outputs_max - test_outputs_min) + test_outputs_min
    # yy_meas = yy_meas * (test_outputs_max - test_outputs_min) + test_outputs_min

    # yy_pred = 10**yy_pred
    # yy_meas = 10**yy_meas

if __name__ == "__main__":
    main()