# Import necessary packages
import optuna
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
    
    # preprocess
    inputs = df.iloc[:13, 0:].values
    inputs[:2, 0:] = df.iloc[:2, 0:].values/10
    outputs = df.iloc[13:25, 0:].values
    outputs[:12, 0:] = outputs[:12, 0:]
    outputs = np.where(outputs <= 0, 1e-10, outputs) 
    outputs[outputs == 0] = 1
    weight = np.ones(inputs.shape[1]) # Could be adjusted in the boundry condition

    # log tranfer
    inputs = np.log10(inputs)
    outputs = np.log10(outputs)

    # normalization
    inputs_max = np.max(inputs, axis=1)
    inputs_min = np.min(inputs, axis=1)
    outputs_max = np.max(outputs, axis=1)
    outputs_min = np.min(outputs, axis=1)
    inputs = (inputs - inputs_min[:, np.newaxis]) / (inputs_max - inputs_min)[:, np.newaxis]
    outputs = (outputs - outputs_min[:, np.newaxis]) / (outputs_max - outputs_min)[:, np.newaxis]

    # tensor transfer
    inputs = inputs.T
    outputs = outputs.T
    weight = weight.T
    outputs_max = outputs_max.T
    outputs_min = outputs_min.T

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    output_tensor = torch.tensor(outputs, dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32)
   
    return torch.utils.data.TensorDataset(input_tensor, output_tensor, weight)

# Config the model training
def objective(trial):

    # Hyperparameters
    NUM_EPOCH = 20 #! 2000
    BATCH_SIZE = 256
    LR_INI = trial.suggest_float("LR_INI", 1e-5, 1e-2, log=True) #! 1e-4
    WEIGHT_DECAY = trial.suggest_float("WEIGHT_DECAY", 1e-8, 1e-5, log=True) #! 1e-7
    DECAY_EPOCH = 100
    DECAY_RATIO = trial.suggest_float("DECAY_RATIO", 0.5, 0.95, log=True) #! 0.95

    # Neural Network Structure
    input_size = 13
    output_size = 12
    hidden_size = trial.suggest_int("hidden_size", 200, 500, log=True) #! 300
    hidden_layers = trial.suggest_int("hidden_layers", 3, 6, log=True) #! 4

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
    dataset = get_dataset('testset_1w_IW.csv') #! Change to 10w datasheet when placed in Snellius 
    train_size = int(0.75 * len(dataset)) 
    valid_size = int(0.25 * len(dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    if torch.cuda.is_available():
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # Setup network
    net = Net(input_size, output_size, hidden_size, hidden_layers).to(device)

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI, weight_decay=WEIGHT_DECAY)
    
    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for inputs, labels, batch_weights in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            weighted_loss = torch.mean(loss * batch_weights.to(device))
            weighted_loss.backward()
            optimizer.step()

            epoch_train_loss += weighted_loss.item()

        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels, batch_weights in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                weighted_loss = torch.mean(loss * batch_weights.to(device))

                epoch_valid_loss += weighted_loss.item()

    # print(f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
    # f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}   "
    # f"LR_INI {LR_INI}   "
    # f"WEIGHT_DECAY {WEIGHT_DECAY}   "
    # f"DECAY_RATIO {DECAY_RATIO} "
    # f"hidden_size {hidden_size} "
    # f"hidden_layers {hidden_layers} ")

    # Log the number of parameters
    with open('optuna_logfile.txt','a', encoding='utf-8') as f:
        f.write(f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f}   "
        f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}   "
        f"LR_INI {LR_INI}   "
        f"WEIGHT_DECAY {WEIGHT_DECAY}   "
        f"DECAY_RATIO {DECAY_RATIO} "
        f"hidden_size {hidden_size} "
        f"hidden_layers {hidden_layers}\n")

    return epoch_valid_loss / len(valid_dataset) * 1e5

# Config the model training
def main():

    # clear output logfile
    with open('optuna_logfile.txt','w', encoding='utf-8') as f:
        pass

    # Create Optuna study object
    study = optuna.create_study(direction="minimize")

    # Hyperparameter optimization
    study.optimize(objective, n_trials=10) #! 100

    # 输出最佳超参数配置
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"        {key}: {value}")

if __name__ == "__main__":
    main()