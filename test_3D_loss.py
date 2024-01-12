# Import necessary packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
NUM_EPOCH = 1000
BATCH_SIZE = 64
LR_INI = 1e-4
WEIGHT_DECAY = 1e-8
DECAY_EPOCH = 30
DECAY_RATIO = 0.95

# Neural Network Structure
input_size = 12
output_size = 1
hidden_size = 10
hidden_layers = 1

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
    data_length = 1000

    # pre-process
    inputs = df.iloc[:12, 0:data_length].values
    inputs[:2] = inputs[:2]/10
    inputs[2:] = inputs[2:]/1e3
    outputs = df.iloc[22:23, 0:data_length].values
    
    # log tranfer
    inputs = np.log10(inputs)
    outputs = np.log10(outputs)

    # normalization
    inputs_max = np.max(inputs, axis=1, keepdims=True)
    inputs_min = np.min(inputs, axis=1, keepdims=True)
    denom = inputs_max - inputs_min
    denom[denom == 0] = 1
    inputs = (inputs - inputs_min) / denom

    outputs_max = np.max(outputs, axis=1, keepdims=True)
    outputs_min = np.min(outputs, axis=1, keepdims=True)
    outputs = (outputs - outputs_min) / (outputs_max - outputs_min)

    # tensor transfer
    inputs = inputs.T
    outputs = outputs.T
    outputs_max = outputs_max.T
    outputs_min = outputs_min.T

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
   
    return torch.utils.data.TensorDataset(input_tensor, output_tensor), outputs_max, outputs_min

def get_data_loss(adr):
    df = pd.read_csv(adr, header=None)
    data_length = 1000

    loss = df.iloc[13:14, 0:data_length].values
    section_loss_IW_Ls = df.iloc[14:15, 0:data_length].values
    section_loss_IW_Lp = df.iloc[15:16, 0:data_length].values
    section_loss_OW_Ls = df.iloc[16:17, 0:data_length].values
    section_loss_OW_Lp = df.iloc[17:18, 0:data_length].values
    corner_loss_IW_Ls = df.iloc[18:19, 0:data_length].values
    corner_loss_IW_Lp = df.iloc[19:20, 0:data_length].values
    corner_loss_OW_Ls = df.iloc[20:21, 0:data_length].values
    corner_loss_OW_Lp = df.iloc[21:22, 0:data_length].values

    return loss, section_loss_IW_Ls, section_loss_IW_Lp, section_loss_OW_Ls, section_loss_OW_Lp, corner_loss_IW_Ls, corner_loss_IW_Lp, corner_loss_OW_Ls, corner_loss_OW_Lp

def main():

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        print("Now this program runs on cpu")
    
    # Load dataset and model
    test_dataset, test_outputs_max, test_outputs_min = get_dataset('dataset_coef/dataset_3D_loss.csv')
    loss, section_loss_IW_Ls, section_loss_IW_Lp, section_loss_OW_Ls, section_loss_OW_Lp, corner_loss_IW_Ls, corner_loss_IW_Lp, corner_loss_OW_Ls, corner_loss_OW_Lp = get_data_loss('dataset_coef/dataset_3D_loss.csv')

    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    net = Net().to(device)
    net.load_state_dict(torch.load('results_coef/Model_3D_loss.pth', map_location = torch.device('cpu')))

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

    yy_pred = 10**yy_pred.T
    yy_meas = 10**yy_meas.T

    # Relative Error
    Error_re = np.zeros_like(yy_meas)
    Error_re[yy_meas != 0] = abs(yy_pred[yy_meas != 0] - yy_meas[yy_meas != 0]) / abs(yy_meas[yy_meas != 0]) * 100

    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    print(f"Relative Error: {Error_re_avg:.8f}%")
    print(f"RMS Error: {Error_re_rms:.8f}%")
    print(f"MAX Error: {Error_re_max:.8f}%")

    loss_pred = yy_pred *((corner_loss_IW_Lp + corner_loss_IW_Ls + corner_loss_OW_Lp + corner_loss_OW_Ls)/2) + (section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp)*2
    Error_loss = abs(loss_pred - loss) / abs(loss) * 100

    Error_loss_avg = np.mean(Error_loss)
    Error_loss_rms = np.sqrt(np.mean(Error_loss ** 2))
    Error_loss_max = np.max(Error_loss)

    print(f"Loss Relative Error: {Error_loss_avg:.8f}%")
    print(f"Loss RMS Error: {Error_loss_rms:.8f}%")
    print(f"Loss Max Error: {Error_loss_max:.8f}%")
   
if __name__ == "__main__":
    main()