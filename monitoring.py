from metrics import *
import torchvision.utils as vutils
import torch



def write_metrics(writer, predictions, gts, loss, epoch, tag):
    """
    Write scalar metrics to tensorboard

    :param writer: SummaryWriter object to write on
    :param predictions: tensor containing predictions
    :param gts: array of tensors containing ground truth
    :param loss: tensor containing the loss value
    :param epoch: int, number of the iteration
    :param tag: string to specify which dataset is used (e.g. "training" or "validation")
    """
    FP, FN, TP, TN = numeric_score(predictions, gts)
    precision = precision_score(FP, FN, TP, TN)
    recall = recall_score(FP, FN, TP, TN)
    specificity = specificity_score(FP, FN, TP, TN)
    iou = intersection_over_union(FP, FN, TP, TN)
    accuracy = accuracy_score(FP, FN, TP, TN)
    dice = dice_score(predictions, gts)

    writer.add_scalar("loss_"+tag, loss, epoch)
    for i in range(len(precision)):
        writer.add_scalar("precision_"+str(i)+"_"+tag, precision[i], epoch)
        writer.add_scalar("recall_"+str(i)+"_"+tag, recall[i], epoch)
        writer.add_scalar("specificity_"+str(i)+"_"+tag, specificity[i], epoch)
        writer.add_scalar("intersection_over_union_"+str(i)+"_"+tag, iou[i], epoch)
        writer.add_scalar("accuracy_"+str(i)+"_"+tag, accuracy[i], epoch)
        writer.add_scalar("dice_"+str(i)+"_"+tag, dice[i], epoch)


def write_images(writer, input, output, predictions, gts, epoch, tag):
    """
    Write images to tensorboard

    :param writer: SummaryWriter object to write on
    :param input: tensor containing input values
    :param output: tensor containing output values
    :param predictions: tensor containing predictions
    :param gts: array of tensors containing ground truth
    :param epoch: int, number of the iteration
    :param tag: string to specify which dataset is used (e.g. "training" or "validation")
    """
    for i in range(input.size()[0]):
        input_image = vutils.make_grid(input[i,:,:].clone().detach().to(dtype=torch.float32),
                                       normalize=True, scale_each=True)
        writer.add_image('Input channel '+str(i)+' '+tag, input_image, epoch)
    for i in range(gts.size()[0]):
        output_image = vutils.make_grid(output[i,:,:], normalize=True)
        writer.add_image('Output class '+str(i)+' '+tag, output_image, epoch)
        pred_image = vutils.make_grid(255*(predictions==i), normalize=False)
        writer.add_image('Prediction class '+str(i)+' '+tag, pred_image, epoch)
        gt_image = vutils.make_grid(gts[i,:,:], normalize=True)
        writer.add_image('GT class '+str(i)+' '+tag, gt_image, epoch)
