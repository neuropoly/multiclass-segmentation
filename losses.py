import torch
import numpy as np



class Dice(object):
    """Dice loss.
    Args:
        smooth (float): value to smooth the dice (and prevent division by 0)
        square (bool): to use the squares of the cardinals at denominator or not
    """
    def __init__(self, smooth=0.001):
        self.smooth = smooth

    def __call__(self, output, gts):
        num = -2*(output * gts).sum()
        den1 = output.pow(2).sum()
        den2 = gts.pow(2).sum()
        loss = (num+self.smooth)/(den1+den2+self.smooth)

        return loss


class CrossEntropy(object):
    """Cross entropy loss.
    """

    def __call__(self, output, gts):

        target = gts[:,0,:,:].clone().zero_()
        for i in range(1, gts.size()[1]):
            target += i*gts[:,i,:,:]

        loss_function = torch.nn.CrossEntropyLoss()

        return loss_function(output, target.long())
