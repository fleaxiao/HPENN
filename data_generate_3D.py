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
        return torch.cat((output1, output2, output3, output4, output5, output6), dim = 1)


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
    inputs = np.concatenate((Np*np.ones_like(hw1), Ns*np.ones_like(hw1), hw1, hw2, dww_ii_x, dww_oo_x, dww_x, lcore_x1, hw, dw, lcs, dcs, dc2c), axis = 1) 

    outputs = df.iloc[0:data_length:, 21:].values
    Rac = np.sum(outputs[0:,0:12], axis=1).reshape(-1,1)
    Llk = np.sum(outputs[0:,12:24], axis=1).reshape(-1,1)

    return inputs, Rac, Llk

def preprocess(inputs, Llk, Rac):
    # preprocess
    inputs_pre = np.copy(inputs)
    Llk_pre = np.copy(Llk)
    Rac_pre = np.copy(Rac)
    inputs_pre[0:,:2] = inputs_pre[0:,:2]/10
    inputs_pre[0:,2:] = inputs_pre[0:,2:]/1e3

    # log tranfer
    inputs_pre = np.log10(inputs_pre)
    Llk_pre = np.log10(Llk_pre)
    Rac_pre = np.log10(Rac_pre)

    # normalization
    inputs_pre[0:,0] = (inputs_pre[0:,0] - np.log10(0.3)) / (np.log10(0.6) - np.log10(0.3))
    inputs_pre[0:,1] = (inputs_pre[0:,1] - np.log10(0.3)) / (np.log10(0.6) - np.log10(0.3))
    inputs_pre[0:,2] = (inputs_pre[0:,2] - np.log10(0.1)) / (np.log10(0.6) - np.log10(0.1))
    inputs_pre[0:,3] = (inputs_pre[0:,3] - np.log10(0.055)) / (np.log10(0.6) - np.log10(0.055))
    inputs_pre[0:,4] = (inputs_pre[0:,4] - np.log10(0.001)) / (np.log10(0.005) - np.log10(0.001))
    inputs_pre[0:,5] = (inputs_pre[0:,5] - np.log10(0.001)) / (np.log10(0.005) - np.log10(0.001))
    inputs_pre[0:,6] = (inputs_pre[0:,6] - np.log10(0.04)) / (np.log10(0.1) - np.log10(0.04))
    inputs_pre[0:,7] = (inputs_pre[0:,7] - np.log10(0.01)) / (np.log10(0.05) - np.log10(0.01))
    inputs_pre[0:,8] = (inputs_pre[0:,8] - np.log10(0.135)) / (np.log10(0.8) - np.log10(0.135))
    inputs_pre[0:,9] = (inputs_pre[0:,9] - np.log10(0.102)) / (np.log10(0.307) - np.log10(0.102))
    
    # tensor transfer
    input_tensor_IW = torch.tensor(inputs_pre[0:,0:10], dtype=torch.float32)
    input_tensor_OW = torch.tensor(inputs_pre[0:,0:8], dtype=torch.float32)

    Llk_tensor = torch.tensor(Llk_pre, dtype=torch.float32)
    Rac_tensor = torch.tensor(Rac_pre, dtype=torch.float32)

    return torch.utils.data.TensorDataset(input_tensor_IW, Llk_tensor, Rac_tensor), torch.utils.data.TensorDataset(input_tensor_OW, Llk_tensor, Rac_tensor)

def load_loss_model(input_size, hidden_size_sequence, device):
    net1 = Net(input_size, hidden_size_sequence[0]).to(device)
    net2 = Net(input_size, hidden_size_sequence[1]).to(device)
    net3 = Net(input_size, hidden_size_sequence[2]).to(device)
    net4 = Net(input_size, hidden_size_sequence[3]).to(device)
    net5 = Net(input_size, hidden_size_sequence[4]).to(device)
    net6 = Net(input_size, hidden_size_sequence[5]).to(device)
    model = CombinedModel(net1, net2, net3, net4, net5, net6)

    return model

def get_loss_model_output(model, device, data_loader, inputs, winding_number):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((-3.2) - (-4.6)) + (-4.6)
    yy_pred = 10**yy_pred

    mask = np.zeros_like(yy_pred)
    for i, num_ones in enumerate(inputs[0:,winding_number]):
        mask[i, :int(num_ones)] = 1
    yy_pred = mask*yy_pred  
    
    return yy_pred

def get_inductor_model_output(model, device, data_loader, inputs, winding_number):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((-5.0) - (-7.4)) + (-7.4)
    yy_pred = 10**yy_pred

    mask = np.zeros_like(yy_pred)
    for i, num_ones in enumerate(inputs[0:,winding_number]):
        mask[i, :int(num_ones)] = 1
    yy_pred = mask*yy_pred  
    
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
    inputs = np.array([]).reshape(-1,13)
    Rac = np.array([]).reshape(-1,1)
    Llk = np.array([]).reshape(-1,1)

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            Np = int(match.group(1)) 
            Ns = int(match.group(2)) 

            file_path = os.path.join(folder_path, filename)
            inputs_i, Rac_i, Llk_i = get_dataset(file_path, Np, Ns)

        inputs = np.concatenate((inputs, inputs_i), axis = 0)
        Rac = np.concatenate((Rac, Rac_i), axis = 0)
        Llk = np.concatenate((Llk, Llk_i), axis = 0)
    
    # Preprocess
    dataset_IW, dataset_OW = preprocess(inputs, Llk, Rac)
    data_loader_IW = torch.utils.data.DataLoader(dataset_IW, batch_size=4, shuffle=False, **kwargs)
    data_loader_OW = torch.utils.data.DataLoader(dataset_OW, batch_size=4, shuffle=False, **kwargs)

    # Load loss model
    # IW Ls
    input_size = 10
    hidden_size_sequence = [143, 143, 122, 143, 127, 102]
    model_loss_IW_Ls = load_loss_model(input_size, hidden_size_sequence, device)
    model_loss_IW_Ls.load_state_dict(torch.load('results_loss/Model_2D_IW_Ls.pth', map_location = torch.device('cpu')))

    # IW Lp
    input_size = 10
    hidden_size_sequence = [100, 143, 134, 143, 140, 116]
    model_loss_IW_Lp = load_loss_model(input_size, hidden_size_sequence, device)
    model_loss_IW_Lp.load_state_dict(torch.load('results_loss/Model_2D_IW_Lp.pth', map_location = torch.device('cpu')))
 
    # OW Ls
    input_size = 8
    hidden_size_sequence = [100, 100, 122, 122, 126, 102]
    model_loss_OW_Ls = load_loss_model(input_size, hidden_size_sequence, device)
    model_loss_OW_Ls.load_state_dict(torch.load('results_loss/Model_2D_OW_Ls.pth', map_location = torch.device('cpu')))

    # OW Lp
    input_size = 8
    hidden_size_sequence = [81, 109, 123, 81, 111, 125]
    model_loss_OW_Lp = load_loss_model(input_size, hidden_size_sequence, device)
    model_loss_OW_Lp.load_state_dict(torch.load('results_loss/Model_2D_OW_Lp.pth', map_location = torch.device('cpu')))

    # Load inductor model
    # IW Ls
    input_size = 10 
    hidden_size = 107
    model_inductor_IW_Ls = Net(input_size, hidden_size).to(device)
    model_inductor_IW_Ls.load_state_dict(torch.load('results_inductor/Model_2D_IW_inside.pth', map_location = torch.device('cpu')))

    # IW Lp
    input_size = 10 
    hidden_size = 101
    model_inductor_IW_Lp = Net(input_size, hidden_size).to(device)
    model_inductor_IW_Lp.load_state_dict(torch.load('results_inductor/Model_2D_IW_outside.pth', map_location = torch.device('cpu')))

    # OW Ls
    input_size = 8
    hidden_size = 101
    model_inductor_OW_Ls = Net(input_size, hidden_size).to(device)
    model_inductor_OW_Ls.load_state_dict(torch.load('results_inductor/Model_2D_OW_inside.pth', map_location = torch.device('cpu')))

    # OW Lp
    input_size = 8
    hidden_size = 101
    model_inductor_OW_Lp = Net(input_size, hidden_size).to(device)
    model_inductor_OW_Lp.load_state_dict(torch.load('results_inductor/Model_2D_OW_outside.pth', map_location = torch.device('cpu')))

    # Output 2D data
    model_loss_IW_Ls.eval()
    model_loss_IW_Lp.eval()
    model_loss_OW_Ls.eval()
    model_loss_OW_Lp.eval()

    loss_IW_Ls = get_loss_model_output(model_loss_IW_Ls, device, data_loader_IW, inputs, 1)
    loss_IW_Lp = get_loss_model_output(model_loss_IW_Lp, device, data_loader_IW, inputs, 0)
    loss_OW_Ls = get_loss_model_output(model_loss_OW_Ls, device, data_loader_OW, inputs, 1)
    loss_OW_Lp = get_loss_model_output(model_loss_OW_Lp, device, data_loader_OW, inputs, 0)

    model_inductor_IW_Ls.eval()
    model_inductor_IW_Lp.eval()
    model_inductor_OW_Ls.eval()
    model_inductor_OW_Lp.eval()

    inductor_IW_Ls = get_inductor_model_output(model_inductor_IW_Ls, device, data_loader_IW, inputs, 1)
    inductor_IW_Lp = get_inductor_model_output(model_inductor_IW_Lp, device, data_loader_IW, inputs, 0)
    inductor_OW_Ls = get_inductor_model_output(model_inductor_OW_Ls, device, data_loader_OW, inputs, 1)
    inductor_OW_Lp = get_inductor_model_output(model_inductor_OW_Lp, device, data_loader_OW, inputs, 0)

    # Calculate section and corner loss/inductance
    section_loss_IW_Ls = np.sum(loss_IW_Ls, axis=1)*inputs[:,10]/1e3
    section_loss_IW_Lp = np.sum(loss_IW_Lp, axis=1)*inputs[:,10]/1e3
    section_loss_OW_Ls = np.sum(loss_OW_Ls, axis=1)*(inputs[:,11]+inputs[:,12])*2/1e3
    section_loss_OW_Lp = np.sum(loss_OW_Lp, axis=1)*(inputs[:,11]+inputs[:,12])*2/1e3

    section_inductor_IW_Ls = inductor_IW_Ls.T*inputs[:,10]/1e3
    section_inductor_IW_Lp = inductor_IW_Lp.T*inputs[:,10]/1e3
    section_inductor_OW_Ls = inductor_OW_Ls.T*(inputs[:,11]+inputs[:,12])*2/1e3
    section_inductor_OW_Lp = inductor_OW_Lp.T*(inputs[:,11]+inputs[:,12])*2/1e3

    corner_radius_Ls = np.zeros((np.shape(inputs)[0], 6))
    corner_radius_Lp = np.zeros((np.shape(inputs)[0], 6))
    length_Ls = np.zeros((np.shape(inputs)[0], 6))
    length_Lp = np.zeros((np.shape(inputs)[0], 6))

    for i in range(0, np.shape(inputs)[0]):
        for j in range(0,6):
            if j < inputs[i,1]:
                corner_radius_Ls[i,j] = inputs[i,7] + (j)*(inputs[i,4]+ 1) + 0.5
                length_Ls[i,j] = 2*inputs[i,10] + 4*(inputs[i,11]+inputs[i,12]) + 2*3.14*corner_radius_Ls[i,j] 
            else:
                corner_radius_Ls[i,j] = 0
                length_Ls[i,j] = 0
    
    for i in range(0, np.shape(inputs)[0]):
        for j in range(0,6):
            if j < inputs[i,0]:
                corner_radius_Lp[i,j] = inputs[i,7] + j*(inputs[i,5]+ 1) + 0.5 + inputs[i,1]*(inputs[i,4]+ 1) + inputs[i,6]
                length_Lp[i,j] = 2*inputs[i,10] + 4*(inputs[i,11]+inputs[i,12]) + 2*3.14*corner_radius_Lp[i,j] 
            else:
                corner_radius_Lp[i,j] = 0
                length_Lp[i,j] = 0
    
    corner_loss_IW_Ls = np.sum(2*3.14*loss_IW_Ls*corner_radius_Ls, axis=1)/1e3
    corner_loss_IW_Lp = np.sum(2*3.14*loss_IW_Lp*corner_radius_Lp, axis=1)/1e3
    corner_loss_OW_Ls = np.sum(2*3.14*loss_OW_Ls*corner_radius_Ls, axis=1)/1e3
    corner_loss_OW_Lp = np.sum(2*3.14*loss_OW_Lp*corner_radius_Lp, axis=1)/1e3

    corner_inductor_IW_Ls = 2*3.14*inductor_IW_Ls.T*np.mean(corner_radius_Ls, axis=1)/1e3
    corner_inductor_IW_Lp = 2*3.14*inductor_IW_Lp.T*np.mean(corner_radius_Lp, axis=1)/1e3
    corner_inductor_OW_Ls = 2*3.14*inductor_OW_Ls.T*np.mean(corner_radius_Ls, axis=1)/1e3
    corner_inductor_OW_Lp = 2*3.14*inductor_OW_Lp.T*np.mean(corner_radius_Lp, axis=1)/1e3
 
    # Obtain correction factor
    # total_1 = (np.sum(loss_IW_Ls*length_Ls, axis=1) + np.sum(loss_IW_Lp*length_Lp, axis=1)) / 1e3
    # total_2 = (np.sum(loss_OW_Ls*length_Ls, axis=1) + np.sum(loss_OW_Lp*length_Lp, axis=1)) / 1e3
    loss = np.squeeze(Rac.reshape(1,-1))  
    rest_loss = loss - (section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp)*2
    coef_loss = rest_loss / ((corner_loss_IW_Lp + corner_loss_IW_Ls + corner_loss_OW_Lp + corner_loss_OW_Ls)/2)
    print(f"loss max corf: {np.max(coef_loss)}")
    print(f"loss min corf: {np.min(coef_loss)}")

    inductance = np.squeeze(Llk.reshape(1,-1))
    rest_inductor = inductance/2 - (section_inductor_IW_Ls + section_inductor_IW_Lp + section_inductor_OW_Ls + section_inductor_OW_Lp)*2
    coef_inductor = rest_inductor / ((corner_inductor_IW_Lp + corner_inductor_IW_Ls + corner_inductor_OW_Lp + corner_inductor_OW_Ls)/2)
    print(f"inductor max corf: {np.max(coef_inductor)}")
    print(f"inductor min corf: {np.min(coef_inductor)}")

    # Save data
    save_data = np.vstack([inputs.T, loss, section_loss_IW_Ls, section_loss_IW_Lp, section_loss_OW_Ls, section_loss_OW_Lp, corner_loss_IW_Ls, corner_loss_IW_Lp, corner_loss_OW_Ls, corner_loss_OW_Lp, coef_loss])
    np.savetxt("dataset_coef/dataset_3D_loss.csv", save_data, delimiter=',')

if __name__ == "__main__":
    main()