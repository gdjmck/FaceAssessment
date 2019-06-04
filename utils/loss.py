import torch
import numpy as np

def hinge_loss(output, target):
    '''
        全部区间用0.05做slack variable
    '''
    L1_abs = torch.abs(target - output)
    return torch.mean(torch.max(L1_abs, torch.FloatTensor([0.05])))

def accuracy(output, target):
    # print('output:', output, '\ttarget:', target)
    acc = (torch.abs(output - target) < 0.1).sum(dtype=torch.float) / target.numel()
    # print('acc:', acc, 'number of elements', target.numel())
    return acc

bce_loss = torch.nn.BCELoss()