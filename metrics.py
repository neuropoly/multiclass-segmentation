import numpy as np

def numeric_score(pred, gts):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    np_pred = pred.numpy()
    np_gts = [gts[:,i,:,:].numpy() for i in range(gts.size()[1])]
    FP = []
    FN = []
    TP = []
    TN = []
    for i in range(len(np_gts)):
        FP.append(np.float(np.sum((np_pred == i) & (np_gts[i] == 0))))
        FN.append(np.float(np.sum((np_pred != i) & (np_gts[i] == 1))))
        TP.append(np.float(np.sum((np_pred == i) & (np_gts[i] == 1))))
        TN.append(np.float(np.sum((np_pred != i) & (np_gts[i] == 0))))
    return FP, FN, TP, TN


def precision_score(FP, FN, TP, TN):
    # PPV
    precision = []
    for i in range(len(FP)):
        if (TP[i] + FP[i]) <= 0.0:
            precision.append(0.0)
        else:
            precision.append(np.divide(TP[i], TP[i] + FP[i])* 100.0)
    return precision


def recall_score(FP, FN, TP, TN):
    # TPR, sensitivity
    TPR = []
    for i in range(len(FP)):
        if (TP[i] + FN[i]) <= 0.0:
            TPR.append(0.0)
        else:
            TPR.append(np.divide(TP[i], TP[i] + FN[i]) * 100.0)
    return TPR


def specificity_score(FP, FN, TP, TN):
    TNR = []
    for i in range(len(FP)):
        if (TN[i] + FP[i]) <= 0.0:
            TNR.append(0.0)
        else:
            TNR.append(np.divide(TN[i], TN[i] + FP[i]) * 100.0)
    return TNR 


def intersection_over_union(FP, FN, TP, TN):
    IOU = []
    for i in range(len(FP)):
        if (TP[i] + FP[i] + FN[i]) <= 0.0:
            IOU.append(0.0)
        else:
            IOU.append(TP[i] / (TP[i] + FP[i] + FN[i]) * 100.0)
    return IOU


def accuracy_score(FP, FN, TP, TN):
    accuracy = []
    for i in range(len(FP)):
        N = FP[i] + FN[i] + TP[i] + TN[i]
        accuracy.append(np.divide(TP[i] + TN[i], N) * 100.0)
    return accuracy


def dice_score(pred, gts):
    dice = []
    np_pred = pred.numpy()[:,0,:,:]
    np_gts = [gts[:,i,:,:].numpy() for i in range(gts.size()[1])]

    for i in range(len(np_gts)):
        intersection = ((np_pred==i)*np_gts[i]).sum()
        card_sum = (np_pred==i).sum()+np_gts[i].sum()
        dice.append(2*intersection/card_sum)
    return dice


def jaccard_score(pred, gts):
    jaccard = []
    for i in range(gts.size()[1]):
        intersection = ((pred==i)*gts[:,i,:,:]).sum()
        union = (pred==i).sum()+gts[:,i,:,:].sum()-intersection
        jaccard.append(float(intersection)/union)
    return jaccard
