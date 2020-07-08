import torch
import torch.nn as nn

class MLP(nn.Module):
    
    #n_hiddens is list with len num_layers 
    #specifying layer width per layer
    # if list is empty, no hiddens
    def __init__(self, n_inputs, n_hiddens):
        super(MLP, self).__init__()
        
        if len(n_hiddens) == 0:
            linears = [nn.Linear(n_inputs, 1)]
        else:
            linears = [nn.Linear(n_inputs, n_hiddens[0])]
            for l in range(len(n_hiddens)-1):
                linears.append(nn.Linear(n_hiddens[l], n_hiddens[l+1]))
            linears.append(nn.Linear(n_hiddens[-1], 1))
            
        self.linears = nn.ModuleList(linears)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        
        for linear in self.linears[:-1]:
            x = self.relu(linear(x))
        out = self.linears[-1](x)
        
        return out
