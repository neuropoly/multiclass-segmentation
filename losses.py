import torch
import numpy as np



class Dice(object):

    def __init__(self, smooth=0.001, square=False, weights=None):
        self.smooth = smooth
        self.power = 1
        if square:
            self.power=2

    def __call__(self, output, gts):
        target = output.clone().zero_()
        target = target + torch.cat(gts, 1)
        num = -2*(output * target).sum()
        den1 = output.pow(self.power).sum()
        den2 = target.pow(self.power).sum()
        loss = (num+self.smooth)/(den1+den2+self.smooth)

        return loss


class CrossEntropy(object):

    def __call__(self, output, gts):
        
        target = gts[0].clone().zero_()
        for i in range(1, len(gts)):
            target += i*gts[i]
        loss_function = torch.nn.CrossEntropyLoss()
        target = target.reshape((target.shape[0], target.shape[2], target.shape[3]))

        return loss_function(output, target.long())


