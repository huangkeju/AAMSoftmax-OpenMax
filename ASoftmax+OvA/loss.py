import torch
import torch.nn.functional as F

def Entropy(inputs):
    batch = inputs.shape[0]
    loss = -torch.sum(torch.mul(torch.log(inputs), inputs)) / batch
    return loss

def BCEOvA(inputs, targets):
    loss = 0.
    for i in range(inputs.shape[-1]):
        loss += F.binary_cross_entropy_with_logits(inputs[:,i], torch.eq(targets, i).float())
    return loss