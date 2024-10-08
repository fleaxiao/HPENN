# Import necessary packages
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_layer = 1 #! adjusted in each trainning
LOG_FILE = f"train_IW_{train_layer}.txt"
MODEL_FILE = f"Model_2D_IW_{train_layer}.pth"
ERROR_FILE = f"train_error_IW_{train_layer}.csv"

# Hyperparameters
NUM_EPOCH = 1 #! 600
BATCH_SIZE = 32
LR_INI = 0.0006758499286351658
DECAY_EPOCH = 100
DECAY_RATIO = 0.5

# Neural Network Structure
input_size = 12 #! IW -> 12, OW -> 11
output_size = 1
hidden_size = 100
hidden_layers = 2

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, outputs, labels):
        # loss = torch.sum((outputs[labels != 0] - labels[labels != 0])**2) / labels.numel()
        # loss = torch.mean((outputs[labels != 0] - labels[labels != 0])**2)
        rms_loss = torch.sqrt(torch.mean((outputs - labels) ** 2))
        max_loss, _ = torch.max(torch.abs(outputs - labels), dim=0)
        loss = rms_loss + max_loss
        return loss

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    cols_drop = df.iloc[11+train_layer][df.iloc[11+train_layer] == 0].index # delect the row where the element in line x is zero
    df = df.drop(columns=cols_drop)
    data_length = 50_000
   
    # pre-process
    inputs = df.iloc[:12, 0:data_length].values #! IW -> 12, OW -> 11
    inputs[:2] = inputs[:2]/10
    print(np.min(inputs, axis=1))
    # print(np.min(inputs, axis=1))
    # print(np.max(inputs, axis=1))
    
    outputs = df.iloc[12:, 0:data_length].values
    outputs = outputs[train_layer-1:train_layer] # train specific layer
    # outputs = np.concatenate([outputs[train_layer-1:train_layer],outputs[train_layer+11:train_layer+12]*1e3]) # train specific layer with two outputs
    # outputs[outputs == 0] = 1 # outputs = np.where(outputs <= 0, 1e-10, outputs) # train multiple layers
    # outputs = np.sum(outputs, axis = 0).reshape(1,-1) # train the total inductance of a whole section   
    
    # log tranfer
    inputs = np.log10(inputs)
    outputs =  np.log10(outputs)

    # normalization
    inputs[0] = (inputs[0] - np.log10(3e-1)) / (np.log10(6e-1) - np.log10(3e-1))
    inputs[1] = (inputs[1] - np.log10(3e-1)) / (np.log10(6e-1) - np.log10(3e-1))
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

    # outputs_max = np.max(outputs, axis=1, keepdims=True)
    # outputs_min = np.min(outputs, axis=1, keepdims=True)
    outputs_max = np.array([1.2])
    outputs_min = np.array([-0.1])
    outputs = (outputs - outputs_min) / (outputs_max - outputs_min)

    # tensor transfer
    inputs = inputs.T
    outputs = outputs.T
    outputs_max = outputs_max.T
    outputs_min = outputs_min.T

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
   
    return torch.utils.data.TensorDataset(input_tensor, output_tensor), outputs_max, outputs_min

# Config the model training
def main():

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        print("Now this program runs on cpu")

    # Load and spit dataset
    dataset, test_outputs_max , test_outputs_min = get_dataset('dataset_coef/dataset_IW_coef.csv') #! adjusted in each trainning
    train_size = int(0.6 * len(dataset)) 
    valid_size = int(0.2 * len(dataset))
    test_size  = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    
    # Setup network
    net = Net().to(device)

    # Log the number of parameters
    with open(LOG_FILE,'w', encoding='utf-8') as f:
        f.write(f"Number of parameters: {count_parameters(net)}\n")

    # Setup optimizer
    # criterion = myLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    
    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        
        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                
                epoch_valid_loss += loss.item()

        if (epoch_i+1)%100 == 0:
            print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f} "
                f"Learning Rate {optimizer.param_groups[0]['lr']}")
            with open('logfile.txt','a', encoding='utf-8') as f:
                print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f} "
                f"Learning Rate {optimizer.param_groups[0]['lr']}",file=f)

    # Save the model parameters
    torch.save(net.state_dict(), MODEL_FILE) 
    print("Training finished! Model is saved!")

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

    # Log the error and logfile
    with open(LOG_FILE,'a', encoding='utf-8') as f:
        f.write(f"Relative Error: {Error_re_avg:.8f}%   "
        f"RMS Error: {Error_re_rms:.8f}%   "
        f"MAX Error: {Error_re_max:.8f}%\n")
    
    np.savetxt(ERROR_FILE, Error_re, delimiter=',') 
   
if __name__ == "__main__":
    main()