# Import necessary packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
NUM_EPOCH = 1000
BATCH_SIZE = 32
LR_INI = 1e-4
WEIGHT_DECAY = 1e-8
DECAY_EPOCH = 30
DECAY_RATIO = 0.95

# Neural Network Structure
input_size = 10
output_size = 1
hidden_size = 143
hidden_layers = 4

train_layer = 12

# Define model structures and functions
class Net(nn.Module):
    
    def __init__(self):
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

def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    cols_drop = df.iloc[9+train_layer][df.iloc[9+train_layer] == 0].index # delect the row where the element in line x is zero
    df = df.drop(columns=cols_drop)
    data_length = 50_000
   
    # pre-process
    inputs = df.iloc[:10, 0:data_length].values #! IW -> 10, OW -> 8
    inputs[:2] = inputs[:2]/10
    
    outputs = df.iloc[10:, 0:data_length].values
    outputs = outputs[train_layer-1:train_layer] # train specific layer power loss
    # outputs = np.concatenate([outputs[train_layer-1:train_layer],outputs[train_layer+11:train_layer+12]*1e3]) # train specific layer with two outputs
    # outputs[outputs == 0] = 1 # outputs = np.where(outputs <= 0, 1e-10, outputs) # train multiple layers
    # outputs = np.sum(outputs, axis = 0).reshape(1,-1) # train the total inductance of a whole section   
    
    # log tranfer
    inputs = np.log10(inputs)
    outputs = np.log10(outputs)

    # normalization
    inputs_max = np.max(inputs, axis=1, keepdims=True)
    inputs_min = np.min(inputs, axis=1, keepdims=True)
    outputs_max = np.max(outputs, axis=1, keepdims=True)
    outputs_min = np.min(outputs, axis=1, keepdims=True)
    diff = inputs_max - inputs_min
    diff[diff == 0] = 1
    inputs = np.where(diff == 1, 1, (inputs - inputs_min) / diff)
    outputs = (outputs - outputs_min) / (outputs_max - outputs_min)

    # tensor transfer
    inputs = inputs.T
    outputs = outputs.T
    outputs_max = outputs_max.T
    outputs_min = outputs_min.T

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
   
    return torch.utils.data.TensorDataset(input_tensor, output_tensor), outputs_max, outputs_min

def main():

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        print("Now this program runs on cpu")
    
    # Load dataset and model
    test_dataset, test_outputs_max, test_outputs_min = get_dataset('dataset_2D/trainset_OW_5w_4.0.csv')
    print(test_outputs_max)
    print(test_outputs_min)

    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    net = Net().to(device)
    net.load_state_dict(torch.load('results_IW/Model_2D_IW_10.pth', map_location = torch.device('cpu')))

  # Evaluation
    net.eval()
    x_meas = []
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))
            x_meas.append(inputs)

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset) * 1e5:.5f}")  # f denotes formatting string
    
    # tensor is transferred to numpy
    yy_pred = y_pred.cpu().numpy()
    yy_meas = y_meas.cpu().numpy()

    yy_pred = yy_pred * (test_outputs_max - test_outputs_min) + test_outputs_min
    yy_meas = yy_meas * (test_outputs_max - test_outputs_min) + test_outputs_min

    yy_pred = 10**yy_pred
    yy_meas = 10**yy_meas

    # Relative Error
    Error_re = np.zeros_like(yy_meas)
    Error_re[yy_meas != 0] = abs(yy_pred[yy_meas != 0] - yy_meas[yy_meas != 0]) / abs(yy_meas[yy_meas != 0]) * 100
    # Error_re = np.squeeze(Error_re, axis=0)

    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    print(f"Relative Error: {Error_re_avg:.8f}%")
    print(f"RMS Error: {Error_re_rms:.8f}%")
    print(f"MAX Error: {Error_re_max:.8f}%")
   
if __name__ == "__main__":
    main()