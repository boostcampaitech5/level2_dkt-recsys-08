import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def BCE_sigmoid_loss(target, pred):
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss = loss(pred, target.float())
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss
