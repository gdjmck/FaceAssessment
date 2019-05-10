import torch

def hinge_loss(output, target):
    '''
        全部区间用0.05做slack variable
    '''
    L1_abs = torch.FloatTensor(target - output)
    loss_mask = L1_abs > 0.05

    return torch.mean(torch.mul(L1_abs, loss_mask))