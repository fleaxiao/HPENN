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
input_size = 14
output_size = 1
hidden_size = 100
hidden_layers = 1

data_length = 700
begin = 0
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
    
    # pre-process
    inputs = df.iloc[:14, begin:(begin+data_length)].values
    inputs[:2] = inputs[:2]/10
    inputs[2:] = inputs[2:]
    outputs = df.iloc[23:24,begin:(begin+data_length)].values
    
    # log tranfer
    inputs = np.log10(inputs)
    outputs = np.log10(outputs)

    # normalization
    inputs[0] = (inputs[0] - np.log10(3e-1)) / (np.log10(6e-1) - np.log10(3e-1))
    inputs[1] = (inputs[1] - np.log10(3e-1)) / (np.log10(4e-1) - np.log10(3e-1))
    inputs[2] = (inputs[2] - np.log10(2e-2)) / (np.log10(7e-2) - np.log10(2e-2))
    inputs[3] = (inputs[3] - np.log10(1e-2)) / (np.log10(7e-2) - np.log10(1e-2))
    inputs[4] = (inputs[4] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs[5] = (inputs[5] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs[6] = (inputs[6] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs[7] = (inputs[7] - np.log10(1e-4)) / (np.log10(3e-4) - np.log10(1e-4))
    inputs[8] = (inputs[8] - np.log10(5e-4)) / (np.log10(6e-3) - np.log10(5e-4))
    inputs[9] = (inputs[9] - np.log10(1e-3)) / (np.log10(4e-3) - np.log10(1e-3))
    inputs[10] = (inputs[10] - np.log10(2.2e-2)) / (np.log10(8.2e-2) - np.log10(2.2e-2))
    inputs[11] = (inputs[11] - np.log10(5.5e-3)) / (np.log10(2.26e-2) - np.log10(5.5e-3))
    inputs[12] = (inputs[12] - np.log10(2e-2)) / (np.log10(8e-2) - np.log10(2e-2))
    inputs[13] = (inputs[13] - np.log10(6e-3)) / (np.log10(4e-2) - np.log10(6e-3))
    # inputs_max = np.max(inputs, axis=1, keepdims=True)
    # inputs_min = np.min(inputs, axis=1, keepdims=True)
    # denom = inputs_max - inputs_min
    # denom[denom == 0] = 1
    # inputs = (inputs - inputs_min) / denom

    # outputs_max = np.max(outputs, axis=1, keepdims=True)
    # outputs_min = np.min(outputs, axis=1, keepdims=True)
    outputs_max = np.array([0.25])
    outputs_min = np.array([-0.05])
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

    loss = df.iloc[14:15, begin:(begin+data_length)].values
    section_loss_IW_Ls = df.iloc[15:16, begin:(begin+data_length)].values
    section_loss_IW_Lp = df.iloc[16:17, begin:(begin+data_length)].values
    section_loss_OW_Ls = df.iloc[17:18, begin:(begin+data_length)].values
    section_loss_OW_Lp = df.iloc[18:19, begin:(begin+data_length)].values
    corner_loss_IW_Ls = df.iloc[19:20, begin:(begin+data_length)].values
    corner_loss_IW_Lp = df.iloc[20:21, begin:(begin+data_length)].values
    corner_loss_OW_Ls = df.iloc[21:22, begin:(begin+data_length)].values
    corner_loss_OW_Lp = df.iloc[22:23, begin:(begin+data_length)].values

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
    test_dataset, test_outputs_max, test_outputs_min = get_dataset('dataset_coef/dataset_3D_loss_ref.csv')
    loss, section_loss_IW_Ls, section_loss_IW_Lp, section_loss_OW_Ls, section_loss_OW_Lp, corner_loss_IW_Ls, corner_loss_IW_Lp, corner_loss_OW_Ls, corner_loss_OW_Lp = get_data_loss('dataset_coef/dataset_3D_loss_ref.csv')

    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    net = Net().to(device)
    net.load_state_dict(torch.load('results_coef/Model_3D_loss_PINN.pth', map_location = torch.device('cpu')))

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

    # Power Losses Error
    loss_pred = yy_pred *((corner_loss_IW_Lp + corner_loss_IW_Ls + corner_loss_OW_Lp + corner_loss_OW_Ls)/2 + (section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp)) + (section_loss_IW_Ls + section_loss_IW_Lp + section_loss_OW_Ls + section_loss_OW_Lp)
    Error_loss = abs(loss_pred - loss) / abs(loss) * 100

    Error_loss_avg = np.mean(Error_loss)
    Error_loss_rms = np.sqrt(np.mean(Error_loss ** 2))
    Error_loss_max = np.max(Error_loss)

    print(f"Loss Relative Error: {Error_loss_avg:.8f}%")
    print(f"Loss RMS Error: {Error_loss_rms:.8f}%")
    print(f"Loss Max Error: {Error_loss_max:.8f}%")

   
if __name__ == "__main__":
    main()