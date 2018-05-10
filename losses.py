import torch


def dice_loss(pred, gts):
    eps = 0.0000000001
    loss = 1.
    intersections = []
    unions = []
    weights = []

    for i in range(len(gts)):
        weights.append(1/(torch.sum(gts[i]))**2+eps)
        intersections.append((pred[::,i,::,::].data.contiguous().view(-1)*gts[i].view(-1)).sum())
        unions.append(torch.sum(pred[::,i,::,::])+torch.sum(gts[i]))

    loss = loss-2*sum([w*i for w,i in zip(weights, intersections)])/(sum([w*u for w,u in zip(weights, unions)])+eps)
    return loss

