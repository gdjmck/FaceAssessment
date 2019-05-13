import torch
import numpy as np

def hinge_loss(output, target):
    '''
        全部区间用0.05做slack variable
    '''
    L1_abs = torch.abs(target - output)
    return torch.mean(torch.max(L1_abs, torch.FloatTensor([0.05])))

def accuracy(output, target):
    return 1 if torch.abs(output - target).item() < 0.1 else 0