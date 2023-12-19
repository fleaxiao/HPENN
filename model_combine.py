# Import necessary packages
import torch
import torch.nn as nn

# Neural Network Structure
input_size = 8
output_size = 1
hidden_layers = 4 

# Define model structures and functions
class Net(nn.Module):
    
    def __init__(self, hidden_size):
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
        return output1, output2, output3, output4, output5, output6

def main():

    # Check whether GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Now this program runs on cuda")
    else:
        device = torch.device("cpu")
        print("Now this program runs on cpu")
    
    # #! IW_Ls
    # # Load model
    # hidden_size = 100
    # net1 = Net(hidden_size).to(device)
    # net1.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_1.pth', map_location = torch.device('cpu')))
    # hidden_size = 100
    # net2 = Net(hidden_size).to(device)
    # net2.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_2.pth', map_location = torch.device('cpu')))
    # hidden_size = 100
    # net3 = Net(hidden_size).to(device)
    # net3.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_3.pth', map_location = torch.device('cpu')))
    # hidden_size = 134
    # net4 = Net(hidden_size).to(device)
    # net4.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_4.pth', map_location = torch.device('cpu')))
    # hidden_size = 127
    # net5 = Net(hidden_size).to(device)
    # net5.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_5.pth', map_location = torch.device('cpu')))
    # hidden_size = 117
    # net6 = Net(hidden_size).to(device)
    # net6.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_6.pth', map_location = torch.device('cpu')))

    # # Creat combined model
    # combined_model = CombinedModel(net1, net2, net3, net4, net5, net6)
    # torch.save(combined_model.state_dict(), 'results_loss/Model_2D_IW_Ls.pth')

    # #! IW_Lp
    # # Load model
    # hidden_size = 100
    # net1 = Net(hidden_size).to(device)
    # net1.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_7.pth', map_location = torch.device('cpu')))
    # hidden_size = 143
    # net2 = Net(hidden_size).to(device)
    # net2.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_8.pth', map_location = torch.device('cpu')))
    # hidden_size = 134
    # net3 = Net(hidden_size).to(device)
    # net3.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_9.pth', map_location = torch.device('cpu')))
    # hidden_size = 143
    # net4 = Net(hidden_size).to(device)
    # net4.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_10.pth', map_location = torch.device('cpu')))
    # hidden_size = 140
    # net5 = Net(hidden_size).to(device)
    # net5.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_11.pth', map_location = torch.device('cpu')))
    # hidden_size = 143
    # net6 = Net(hidden_size).to(device)
    # net6.load_state_dict(torch.load('results_loss/results_IW/Model_2D_IW_12.pth', map_location = torch.device('cpu')))

    # # Creat combined model
    # combined_model = CombinedModel(net1, net2, net3, net4, net5, net6)
    # torch.save(combined_model.state_dict(), 'results_loss/Model_2D_IW_Lp.pth')

    #! OW_Ls
    # Load model
    hidden_size = 100
    net1 = Net(hidden_size).to(device)
    net1.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_1.pth', map_location = torch.device('cpu')))
    hidden_size = 100
    net2 = Net(hidden_size).to(device)
    net2.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_2.pth', map_location = torch.device('cpu')))
    hidden_size = 100
    net3 = Net(hidden_size).to(device)
    net3.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_3.pth', map_location = torch.device('cpu')))
    hidden_size = 100
    net4 = Net(hidden_size).to(device)
    net4.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_4.pth', map_location = torch.device('cpu')))
    hidden_size = 128
    net5 = Net(hidden_size).to(device)
    net5.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_5.pth', map_location = torch.device('cpu')))
    hidden_size = 139
    net6 = Net(hidden_size).to(device)
    net6.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_6.pth', map_location = torch.device('cpu')))

    # Creat combined model
    combined_model = CombinedModel(net1, net2, net3, net4, net5, net6)
    torch.save(combined_model.state_dict(), 'results_loss/Model_2D_OW_Ls.pth')

    #! OW_Ls
    # Load model
    hidden_size = 81
    net1 = Net(hidden_size).to(device)
    net1.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_7.pth', map_location = torch.device('cpu')))
    hidden_size = 109
    net2 = Net(hidden_size).to(device)
    net2.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_8.pth', map_location = torch.device('cpu')))
    hidden_size = 123
    net3 = Net(hidden_size).to(device)
    net3.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_9.pth', map_location = torch.device('cpu')))
    hidden_size = 81
    net4 = Net(hidden_size).to(device)
    net4.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_10.pth', map_location = torch.device('cpu')))
    hidden_size = 81
    net5 = Net(hidden_size).to(device)
    net5.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_11.pth', map_location = torch.device('cpu')))
    hidden_size = 125
    net6 = Net(hidden_size).to(device)
    net6.load_state_dict(torch.load('results_loss/results_OW/Model_2D_OW_12.pth', map_location = torch.device('cpu')))

    # Creat combined model
    combined_model = CombinedModel(net1, net2, net3, net4, net5, net6)
    torch.save(combined_model.state_dict(), 'results_loss/Model_2D_OW_Lp.pth')


if __name__ == "__main__":
    main()