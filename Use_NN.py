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
    input_tensor_3D = torch.tensor(inputs_pre[0:,0:14], dtype=torch.float32)

    R = torch.tensor(np.zeros((number,1)), dtype=torch.float32)

    return torch.utils.data.TensorDataset(input_tensor_3D, R)

def load_3D_model(hidden_layers, input_size, hidden_size_sequence, device):
    model = Net(hidden_layers, input_size, hidden_size_sequence).to(device)

    return model

def get_3D_model_output(model, device, data_loader):
    y_pred = []
    with torch.no_grad():
        for inputs_tensor, _ in data_loader:
            outputs_tensor = model(inputs_tensor.to(device))
            y_pred.append(outputs_tensor)
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = y_pred.cpu().numpy()
    yy_pred = yy_pred * ((-1.4) - (-3.1)) + (-3.1)  #! modified
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
    dww_ii_x = np.array([[0.2e-3]])
    dww_oo_x = np.array([[0.2e-3]])
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
    dataset_3D = preprocess(inputs)
    data_loader_3D = torch.utils.data.DataLoader(dataset_3D, batch_size=4, shuffle=False, **kwargs)

    # Load 3D model
    hidden_layers = 1
    input_size = 14
    hidden_size = 100
    model_loss_3D = load_3D_model(hidden_layers, input_size, hidden_size, device)
    model_loss_3D.load_state_dict(torch.load('results_coef/Model_3D_loss_NN.pth', map_location = torch.device('cpu')))

    model_loss_3D.eval()

    loss = get_3D_model_output(model_loss_3D, device, data_loader_3D)
    print(f"Loss: {loss}")

    end_time = time.time()
    print(f"Program run time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()