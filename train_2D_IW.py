# Import necessary packages
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters
NUM_EPOCH = 20 #! 2000
BATCH_SIZE = 400
LR_INI = 1e-4
WEIGHT_DECAY = 1e-8
DECAY_EPOCH = 30
DECAY_RATIO = 0.95

# Neural Network Structure
input_size = 13
output_size = 24
hidden_size = 100 #! 1000
hidden_layers = 2 #! 6

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

# Load the datasheet
def get_dataset(adr):
    df = pd.read_csv(adr, header=None)
    inputs = df.iloc[:13, 0:].values
    targets = df.iloc[13:, 0:].values
    targets[:6, 0:] = targets[:6, 0:]*1e4+1
    targets[6:, 0:] = targets[6:, 0:]*1e6+1
    weight = np.ones(inputs.shape[1]) # Could be adjusted in the boundry condition
    inputs = inputs.T
    targets = targets.T
    weight = weight.T

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    out_tensor = torch.tensor(targets, dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32)

    input_tensor = torch.log10(input_tensor)
    out_tensor = torch.log10(out_tensor)
    
    return torch.utils.data.TensorDataset(input_tensor, out_tensor, weight)

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
    dataset = get_dataset('testset_1w_IW.csv') #! Change to 10w datasheet when placed in Snellius 
    train_size = int(0.75 * len(dataset)) 
    valid_size = int(0.25 * len(dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    test_dataset = get_dataset('testset_1w_IW.csv')
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
    with open('logfile.txt','w') as f:
        f.write(f"Number of parameters: {count_parameters(net)}\n")

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

        if (epoch_i+1)%100 == 0:
            print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f} "
                f"Learning Rate {optimizer.param_groups[0]['lr']}")
            with open('logfile.txt','a') as f:
                print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f} "
                f"Learning Rate {optimizer.param_groups[0]['lr']}",file=f)

    # Save the model parameters
    torch.save(net.state_dict(), "Model_2D_IW.pth")
    print("Training finished! Model is saved!")

    # Evaluation
    net.eval()
    x_meas = []
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, batch_weights in test_loader:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))
            x_meas.append(inputs)

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset) * 1e5:.5f}")  # f denotes formatting string

    yy_pred = 10**(y_pred.cpu().numpy()) # tensor is transferred to numpy
    yy_meas = 10**(y_meas.cpu().numpy())
    yy_pred = (yy_pred - 1) / 1e4
    yy_meas = (yy_meas - 1) / 1e6
    
    # Relative Error
    Error_re = np.zeros_like(yy_meas)
    for i in range(yy_meas.shape[0]):
        for j in range(yy_meas.shape[1]):
            if yy_meas[i, j] == 0:
                Error_re[i, j] = 0
            else:
                Error_re[i, j] = abs(yy_pred[i, j] - yy_meas[i, j]) / abs(yy_meas[i, j]) * 100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    print(f"Relative Error: {Error_re_avg:.8f}%")
    print(f"RMS Error: {Error_re_rms:.8f}%")
    print(f"MAX Error: {Error_re_max:.8f}%")
   
    # Visualization
    Error_Rac_Ls = 0
    Error_Rac_Lp = 0
    Error_Llk_Ls = 0
    Error_Llk_Lp = 0
    
    colors = plt.cm.viridis(np.linspace(0, 1, Error_re.shape[1]))
    binwidth = 1e4

    #TODO Could change to "bins=np.arange(0, Error_re[:,i].max() + binwidth, binwidth)" when the erro is less than 10%
    plt.figure(figsize=(8, 5))
    for i in range (int(Error_re.shape[1]/4)):
        plt.hist(Error_re[:,i], bins=np.arange(0, Error_re[:,i].max() + binwidth, binwidth), density=True, alpha=0.6, color=colors[i], edgecolor='black') # density = (number / total number) / interval width
        Error_Rac_Ls += np.sum(Error_re[:,i] > 5)
    plt.title('Rac Error Distribution in Inner Winding')
    plt.xlabel('Error(%)')
    plt.ylabel('Distribution')
    plt.legend(labels=['inner_layer_1','inner_layer_2','inner_layer_3','inner_layer_4','inner_layer_5','inner_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Rac_Ls.png',dpi=600)

    plt.figure(figsize=(8, 5))
    for i in range (int(Error_re.shape[1]/4), int(2*Error_re.shape[1]/4)):
        plt.hist(Error_re[:,i], bins=np.arange(Error_re[:,i].min(), Error_re[:,i].max() + binwidth, binwidth), density=True, alpha=0.6, color=colors[i], edgecolor='black') 
        Error_Rac_Lp += np.sum(Error_re[:,i] > 5)
    plt.title('Rac Error Distribution in Outer Winding')
    plt.xlabel('Error(%)')
    plt.ylabel('Distribution')
    plt.legend(labels=['outer_layer_1','outer_layer_2','outer_layer_3','outer_layer_4','outer_layer_5','outer_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Rac_Lp.png',dpi=600)

    plt.figure(figsize=(8, 5))
    for i in range (int(2*Error_re.shape[1]/4), int(3*Error_re.shape[1]/4)):
        plt.hist(Error_re[:,i], bins=np.arange(Error_re[:,i].min(), Error_re[:,i].max() + binwidth, binwidth), density=True, alpha=0.6, color=colors[i], edgecolor='black') 
        Error_Llk_Ls += np.sum(Error_re[:,i] > 5)
    plt.title('Llk Error Distribution in Inner Winding')
    plt.xlabel('Error(%)')
    plt.ylabel('Distribution')
    plt.legend(labels=['outer_layer_1','outer_layer_2','outer_layer_3','outer_layer_4','outer_layer_5','outer_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Llk_Ls.png',dpi=600)
        
    plt.figure(figsize=(8, 5))
    for i in range (int(3*Error_re.shape[1]/4), int(Error_re.shape[1])):
        plt.hist(Error_re[:,i], bins=np.arange(Error_re[:,i].min(), Error_re[:,i].max() + binwidth, binwidth), density=True, alpha=0.6, color=colors[i], edgecolor='black') 
        Error_Llk_Lp += np.sum(Error_re[:,i] > 5)
    plt.title('Llk Error Distribution in Outer Winding')
    plt.xlabel('Error(%)')
    plt.ylabel('Distribution')
    plt.legend(labels=['outer_layer_1','outer_layer_2','outer_layer_3','outer_layer_4','outer_layer_5','outer_layer_6'])
    plt.grid()
    plt.savefig('figs/Fig_Llk_Lp.png',dpi=600)

    print(f"Number of Rac errors greater than 5% in inner winding: {Error_Rac_Ls}")
    print(f"Number of Rac errors greater than 5% in outer winding: {Error_Rac_Lp}")
    print(f"Number of Llk errors greater than 5% in inner winding: {Error_Llk_Ls}")
    print(f"Number of Llk errors greater than 5% in outer winding: {Error_Llk_Lp}")

    # plt.show()

if __name__ == "__main__":
    main()