import torch
import numpy as np



class Dice(object):
    """Dice loss.
    Args:
        smooth (float): value to smooth the dice (and prevent division by 0)
        square (bool): to use the squares of the cardinals at denominator or not
    """
    def __init__(self, smooth=0.001, square=False):
        self.smooth = smooth
        self.power = 1
        if square:
            self.power=2

    def __call__(self, output, gts):
        num = -2*(output * gts).sum()
        den1 = output.pow(self.power).sum()
        den2 = gts.pow(self.power).sum()
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
        target = target.reshape((target.shape[0], target.shape[2], target.shape[3]))

        return loss_function(output, target.long())


