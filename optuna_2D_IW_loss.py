# Import necessary packages
import optuna
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define model structures and functions
class Net(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, hidden_layers):
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

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    train_layer = 6 #! adjusted in each trainning
    cols_to_drop = df.iloc[9+train_layer][df.iloc[9+train_layer] == 0].index # delect the row where the element in line x is zero
    df = df.drop(columns=cols_to_drop)
    # df.to_csv("processed data.csv", index=False, header=False)
    data_length = 1_000 #! 50_000
   
    # pre-process
    inputs = df.iloc[:10, 0:data_length].values
    inputs[:2] = inputs[:2]/10
    # heigth_windows = np.maximum(inputs[2]+inputs[11], inputs[3]+inputs[12]) #calculate the height of window
    # inputs = inputs[:-2]
    # inputs = np.vstack([inputs, heigth_windows])
    
    outputs = df.iloc[10:22, 0:data_length].values
    outputs = outputs[train_layer-1:train_layer] # train specific layer
    # outputs = np.sum(outputs, axis = 0).reshape(1,-1) # train the loss of a whole section
    # outputs[outputs == 0] = 1 # outputs = np.where(outputs <= 0, 1e-10, outputs) 

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
    inputs = np.where(diff == 1, 1, inputs - inputs_min / diff)
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
def objective(trial):

    # Hyperparameters
    NUM_EPOCH = 10 #! 600
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [128, 256])
    LR_INI = trial.suggest_float("LR_INI", 1e-5, 1e-2, log=True) #! 1e-4
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5

    # Neural Network Structure
    input_size = 10
    output_size = 1
    hidden_size = trial.suggest_int("hidden_size", 10, 100, log=True) #! 300
    hidden_layers = 3

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Load and spit dataset
    dataset, test_outputs_max , test_outputs_min = get_dataset('trainset_5w_IW.csv') #! Change to 10w or 5w datasheet when placed in Snellius 
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
    net = Net(input_size, output_size, hidden_size, hidden_layers).to(device)

    # Setup optimizer
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

    # Log the number of parameters
    with open('optuna_logfile.txt','a', encoding='utf-8') as f:
        f.write(f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f}   "
        f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}   "
        f"LR_INI {LR_INI:.5f}   "
        f"BATCH_SIZE {BATCH_SIZE}   "
        f"hidden_size {hidden_size}\n")

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

    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)

    with open('optuna_logfile.txt','a', encoding='utf-8') as f:
        f.write(f"Relative Error: {Error_re_avg:.5f}%   "
        f"RMS Error: {Error_re_rms:.5f}%   "
        f"MAX Error: {Error_re_max:.5f}% \n")
    
    # print(f"Relative Error: {Error_re_avg:.8f}%")
    # print(f"RMS Error: {Error_re_rms:.8f}%")
    # print(f"MAX Error: {Error_re_max:.8f}%")

    return epoch_valid_loss / len(valid_dataset) * 1e5

# Config the model training
def main():

    # clear output logfile
    with open('optuna_logfile.txt','w', encoding='utf-8') as _:
        pass

    # Create Optuna study object
    study = optuna.create_study(direction="minimize")

    # Hyperparameter optimization
    study.optimize(objective, n_trials=30) #! 100

    # Output perfect results
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"        {key}: {value}")

if __name__ == "__main__":
    main()