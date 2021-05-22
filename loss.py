import torch
import torch.nn as nn

def discr_bce_loss(real_probs, fake_probs, criterion = nn.BCEWithLogitsLoss()):
    #inputs: 2D tensors
    #returns: float
    real_loss = criterion(real_probs, torch.ones_like(real_probs))
    fake_loss = criterion(fake_probs, torch.zeros_like(fake_probs))
    return ((real_loss + fake_loss) / 2).item()

def gen_bce_loss(fake_probs, criterion = nn.BCEWithLogitsLoss()):
    # input: 2D tensor
    # returns: float
    loss = criterion(fake_probs, torch.ones_like(fake_probs))
    # in guide there were also L1 loss here, link:
    #https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/train.py
    return loss.item()

def test():
    real_probs = torch.randn((3,5))
    fake_probs = torch.randn((3,5))
    l = discr_bce_loss(real_probs, fake_probs)
    l_g = gen_bce_loss(real_probs)
    print(type(l))
    print(type(l_g))

