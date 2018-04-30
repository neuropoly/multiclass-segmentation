import torch


def dice(pred, gt):
    eps = 0.0000000001
    return -(2*(pred.data.contiguous().view(-1)*gt.view(-1)).sum()+eps)/(torch.sum(pred)+torch.sum(gt)+eps)

def dice_loss(pred, gts):
    loss = 0.
    for i in range(len(gts)):
        loss = loss+dice(pred[::,i,::,::], gts[i])
    return loss