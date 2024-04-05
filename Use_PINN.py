import numpy as np
import torch
import torch.nn as nn
import time

start_time = time.time()

# Calculate Number of Parameters
number = 1

# Neural Network Structure
output_size = 1

# Define model structures and functions
class Net(nn.Module):
    
    def __init__(self, hidden_layers, input_size, hidden_size):
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

def preprocess(inputs):
    # preprocess
    inputs_pre = np.copy(inputs)
    inputs_pre[0:,:2] = inputs_pre[0:,:2]/10
    inputs_pre[0:,2:] = inputs_pre[0:,2:]
    print(inputs_pre)
    # log tranfer
    inputs_pre = np.log10(inputs_pre)

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
    inputs_pre[0:,12] = (inputs_pre[0:,12] - np.log10(2e-2)) / (np.log10(8e-2) - np.log10(2e-2))
    inputs_pre[0:,13] = (inputs_pre[0:,13] - np.log10(6e-3)) / (np.log10(4e-2) - np.log10(6e-3))

    # tensor transfer
    input_tensor_IW = torch.tensor(inputs_pre[0:,0:12], dtype=torch.float32)
    input_tensor_OW = torch.tensor(inputs_pre[0:,0:11], dtype=torch.float32)
    input_tensor_3D = torch.tensor(inputs_pre[0:,0:14], dtype=torch.float32)

    R = torch.tensor(np.zeros((number,1)), dtype=torch.float32)

    return torch.utils.data.TensorDataset(input_tensor_IW, R), torch.utils.data.TensorDataset(input_tensor_OW, R), torch.utils.data.TensorDataset(input_tensor_3D, R)

def load_2D_model(hidden_layers, input_size, hidden_size_sequence, device):
    net1 = Net(hidden_layers, input_size, hidden_size_sequence[0]).to(device)
    net2 = Net(hidden_layers, input_size, hidden_size_sequence[1]).to(device)
    net3 = Net(hidden_layers, input_size, hidden_size_sequence[2]).to(device)
    net4 = Net(hidden_layers, input_size, hidden_size_sequence[3]).to(device)
    net5 = Net(hidden_layers, input_size, hidden_size_sequence[4]).to(device)
    net6 = Net(hidden_layers, input_size, hidden_size_sequence[5]).to(device)
    model = CombinedModel(net1, net2, net3, net4, net5, net6)

    return model

def load_3D_model(hidden_layers, input_size, hidden_size_sequence, device):
    model = Net(hidden_layers, input_size, hidden_size_sequence).to(device)

    return model

def get_2D_model_output(model, device, data_loader, inputs, winding_number):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((1) - (-0.15)) + (-0.15)
    yy_pred = 10**yy_pred

    mask = np.zeros_like(yy_pred)
    for i, num_ones in enumerate(inputs[0:,winding_number]):
        mask[i, :int(num_ones)] = 1
    yy_pred = mask
    
    return yy_pred

def get_3D_model_output(model, device, data_loader):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((0.25) - (-0.05)) + (-0.05)  #! modified
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

    Np = np.array([[4]])
    Ns = np.array([[4]])
    hw1 = np.array([[35e-3]])
    hw2 = np.array([[25e-3]])
    dw1 = np.array([[0.2e-3]])
    dw2 = np.array([[0.2e-3]])
    dww_ii_x = np.array([[0.3e-3]])
    dww_oo_x = np.array([[0.3e-3]])
    dww_x = np.array([[4.5e-3]])
    lcore_x1 = np.array([[3e-3]])
    lcore_x2 = np.array([[3e-3]])
    lcore_y1 = np.array([[5e-3]])
    lcore_y2 = np.array([[5e-3]])
    lcs = np.array([[32e-3]])
    dcs = np.array([[11e-3]])
    
    hw = np.maximum(hw1+2*lcore_y1, hw2+2*lcore_y2)
    dw = lcore_x1 + Ns*dw1 + (Ns-1)*dww_ii_x + dww_x + Np*dw2 + (Np-1)*dww_oo_x + lcore_x2
    
    inputs = np.concatenate([Np, Ns, hw1, hw2, dw1, dw2, dww_ii_x, dww_oo_x, dww_x, lcore_x1, hw, dw, lcs, dcs], axis=1)

    # Preprocess
    dataset_IW, dataset_OW, dataset_3D = preprocess(inputs)
    data_loader_IW = torch.utils.data.DataLoader(dataset_IW, batch_size=4, shuffle=False, **kwargs)
    data_loader_OW = torch.utils.data.DataLoader(dataset_OW, batch_size=4, shuffle=False, **kwargs)
    data_loader_3D = torch.utils.data.DataLoader(dataset_3D, batch_size=4, shuffle=False, **kwargs)

    #! 2D PINN
    # Dowell's equation
    permeability = 1.256629e-6
    conductivity = 5.96e7
    skin_depth = np.sqrt(2/(2*3.14*10e3*permeability*conductivity))
    delta_1 = np.sqrt(hw1/hw)*dw1 / skin_depth
    delta_2 = np.sqrt(hw2/hw)*dw2 / skin_depth
    factor_1_1 = (np.sinh(2*delta_1)+np.sin(2*delta_1)) / (np.cosh(2*delta_1)-np.cos(2*delta_1))
    factor_1_2 = (np.sinh(delta_1)-np.sin(delta_1)) / (np.cosh(delta_1)+np.cos(delta_1))
    factor_2_1 = (np.sinh(2*delta_2)+np.sin(2*delta_2)) / (np.cosh(2*delta_2)-np.cos(2*delta_2))
    factor_2_2 = (np.sinh(delta_2)-np.sin(delta_2)) / (np.cosh(delta_2)+np.cos(delta_2))

    R_dc_1 = 1 / (conductivity*dw1*hw1)
    R_dc_2 = 1 / (conductivity*dw2*hw2)

    R_dowell_i = np.zeros((number, 6))
    R_dowell_o = np.zeros((number, 6))

    for i in range(0,6):
        R_dowell_i[:,i] = np.squeeze(R_dc_1*delta_1*(factor_1_1+2*(i+1)*i*factor_1_2))

    for i in range(0,6):
        R_dowell_o[:,i] = np.squeeze(R_dc_2*delta_2*(factor_2_1+2*(i+1)*i*factor_2_2))
    R_dowell_o = R_dowell_o[:, ::-1]

    R_dowell = np.concatenate((R_dowell_i, R_dowell_o), axis=1)

    # Load 2D model
    hidden_layers = 3
    # IW Ls
    input_size = 12
    hidden_size_sequence = [100, 100, 100, 100, 100, 100]
    model_loss_IW_Ls = load_2D_model(hidden_layers, input_size, hidden_size_sequence, device)
    model_loss_IW_Ls.load_state_dict(torch.load('results_loss/Model_2D_IW_Ls.pth', map_location = torch.device('cpu')))

    # IW Lp
    input_size = 12
    hidden_size_sequence = [100, 100, 100, 100, 100, 100]
    model_loss_IW_Lp = load_2D_model(hidden_layers, input_size, hidden_size_sequence, device)
    model_loss_IW_Lp.load_state_dict(torch.load('results_loss/Model_2D_IW_Lp.pth', map_location = torch.device('cpu')))
 
    # OW Ls
    input_size = 11
    hidden_size_sequence = [100, 100, 100, 100, 100, 100]
    model_loss_OW_Ls = load_2D_model(hidden_layers, input_size, hidden_size_sequence, device)
    model_loss_OW_Ls.load_state_dict(torch.load('results_loss/Model_2D_OW_Ls.pth', map_location = torch.device('cpu')))

    # OW Lp
    input_size = 11
    hidden_size_sequence = [100, 100, 100, 100, 100, 100]
    model_loss_OW_Lp = load_2D_model(hidden_layers, input_size, hidden_size_sequence, device)
    model_loss_OW_Lp.load_state_dict(torch.load('results_loss/Model_2D_OW_Lp.pth', map_location = torch.device('cpu')))


    # Output 2D data
    model_loss_IW_Ls.eval()
    model_loss_IW_Lp.eval()
    model_loss_OW_Ls.eval()
    model_loss_OW_Lp.eval()
    
    loss_IW_Ls = get_2D_model_output(model_loss_IW_Ls, device, data_loader_IW, inputs, 1)*R_dowell[:,:6]
    loss_IW_Lp = get_2D_model_output(model_loss_IW_Lp, device, data_loader_IW, inputs, 0)*R_dowell[:,6:]
    loss_OW_Ls = get_2D_model_output(model_loss_OW_Ls, device, data_loader_OW, inputs, 1)*R_dowell[:,:6]
    loss_OW_Lp = get_2D_model_output(model_loss_OW_Lp, device, data_loader_OW, inputs, 0)*R_dowell[:,6:]

    #! 3D PINN
    # Calculate section and corner loss
    section_loss_IW_Ls = np.sum(loss_IW_Ls, axis=1)*inputs[:,12]
    section_loss_IW_Lp = np.sum(loss_IW_Lp, axis=1)*inputs[:,12]
    section_loss_OW_Ls = np.sum(loss_OW_Ls, axis=1)*inputs[:,13]*2
    section_loss_OW_Lp = np.sum(loss_OW_Lp, axis=1)*inputs[:,13]*2

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

    corner_loss_IW_Ls = np.sum(2*3.14*loss_IW_Ls*corner_radius_Ls, axis=1)
    corner_loss_IW_Lp = np.sum(2*3.14*loss_IW_Lp*corner_radius_Lp, axis=1)
    corner_loss_OW_Ls = np.sum(2*3.14*loss_OW_Ls*corner_radius_Ls, axis=1)
    corner_loss_OW_Lp = np.sum(2*3.14*loss_OW_Lp*corner_radius_Lp, axis=1)

    # Load 3D model
    hidden_layers = 1
    input_size = 14
    hidden_size = 100
    model_loss_3D = load_3D_model(hidden_layers, input_size, hidden_size, device)
    model_loss_3D.load_state_dict(torch.load('results_coef/Model_3D_loss_PINN.pth', map_location = torch.device('cpu')))

    model_loss_3D.eval()

    coef = get_3D_model_output(model_loss_3D, device, data_loader_3D)
    loss = (section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp) + coef*((section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp)+(corner_loss_IW_Lp + corner_loss_IW_Ls + corner_loss_OW_Lp + corner_loss_OW_Ls)/2)
    print(f"Coef: {coef}")
    print(f"Loss: {loss}")

    end_time = time.time()
    print(f"Program run time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()