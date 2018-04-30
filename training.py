import torch
from dataset import *
import transforms
import json
from torchvision import transforms as torch_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys
from models import UNet
import losses
from metrics import *

## LOAD HYPERPARAMETERS ##

path_to_json = '/Users/frpau_local/Documents/nih/data/luisa_with_gt/parameters.json'

if len(sys.argv)>1:
	path_to_json = sys.argv[1]

parameters = json.load(open(path_to_json))


## CREATE DATASETS ##

toTensor = transforms.ToTensor()
toPIL = transforms.ToPIL()
randomVFlip = transforms.RandomVerticalFlip()
randomResizedCrop = transforms.RandomResizedCrop(parameters["transforms"]["crop_size"], scale=parameters["transforms"]["scale_range"], ratio=parameters["transforms"]["ratio_range"])
randomRotation = transforms.RandomRotation(parameters["transforms"]["max_angle"])
elasticTransform = transforms.ElasticTransform(parameters["transforms"]["alpha_range"], parameters["transforms"]["sigma_range"], parameters["transforms"]["elastic_rate"])

composed = torch_transforms.Compose([toPIL,randomVFlip,randomRotation,randomResizedCrop, elasticTransform, toTensor])

training_dataset = MRI2DSegDataset("/Users/frpau_local/Documents/nih/data/luisa_with_gt/filenames_csf_gm_nawm_training.txt", transform = composed)
validation_dataset = MRI2DSegDataset("/Users/frpau_local/Documents/nih/data/luisa_with_gt/filenames_csf_gm_nawm_validation.txt", transform = toTensor)

training_dataloader = DataLoader(training_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True, num_workers=4)

## CREATE NET ##

net = net = UNet(drop_rate=parameters["net"]["drop_rate"], bn_momentum=parameters["net"]["bn_momentum"])


## DEFINE LOSS AND OPTIMIZER ##

if parameters["training"]["optimizer"]=="sgd":
    optimizer = optim.SGD(net.parameters(), lr=parameters["training"]['learning_rate'], momentum=parameters["training"]['sgd_momentum'])

if parameters["training"]["loss_function"]=="dice":
    loss_function = losses.dice_loss


## TRAINING ##

writer = SummaryWriter()
# add hyperparameters to description
writer.add_text("hyperparameters", json.dumps(parameters))
last_run_dir = os.listdir("./runs")[-1]

best_loss = 0.
batch_nb = len(training_dataloader)

for epoch in tqdm(range(parameters["training"]["nb_epochs"])):
    
    loss_sum = 0.
    
    for i_batch, sample_batched in enumerate(training_dataloader):
        output =  net(sample_batched['input'])
        gts = [get_bg_gt(sample_batched['gt'])]+sample_batched['gt']
        loss = loss_function(output, gts)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()/batch_nb

    predictions = torch.argmax(output, 1, keepdim=True)
        
    # Visualization
    n_iter = epoch

    # loss
    writer.add_scalar("loss_"+parameters["training"]["loss_function"]+"_training", loss_sum, n_iter)

    # metrics
    bg_gt = get_bg_gt(sample_batched['gt'])
    FP, FN, TP, TN = numeric_score(predictions, [bg_gt]+sample_batched['gt'])
    precision = precision_score(FP, FN, TP, TN)
    recall = recall_score(FP, FN, TP, TN)
    specificity = specificity_score(FP, FN, TP, TN)
    iou = intersection_over_union(FP, FN, TP, TN)
    accuracy = accuracy_score(FP, FN, TP, TN)
    for i in range(len(precision)):
        writer.add_scalar("precision_"+str(i)+"_training", precision[i], n_iter)
        writer.add_scalar("recall_"+str(i)+"_training", recall[i], n_iter)
        writer.add_scalar("specificity_"+str(i)+"_training", specificity[i], n_iter)
        writer.add_scalar("intersection_over_union_"+str(i)+"_training", iou[i], n_iter)
        writer.add_scalar("accuracy_"+str(i)+"_training", accuracy[i], n_iter)

    #images
    input_image = vutils.make_grid(sample_batched['input'][0]/torch.max(sample_batched['input'][0]), normalize=True, scale_each=True)
    writer.add_image('Input image training', input_image, n_iter)
    output_bg = vutils.make_grid(output[0,0,::,::], normalize=True, scale_each=True)
    pred_bg = vutils.make_grid(predictions[0,0,::,::]==0, normalize=True, scale_each=True)
    writer.add_image('Output background training', output_bg, n_iter)
    writer.add_image('Prediction background training', pred_bg, n_iter)
    for i in range(len(sample_batched['gt'])):
        output_image = vutils.make_grid(output[0,i+1,::,::], normalize=True, scale_each=True)
        pred_image = vutils.make_grid(predictions[0,0,::,::]==i+1, normalize=True, scale_each=True)
        writer.add_image('Output class '+str(i+1)+' training', output_image, n_iter)
        writer.add_image('Prediction class '+str(i+1)+' training', pred_image, n_iter)
        gt_image = vutils.make_grid(sample_batched['gt'][i][0,::,::], normalize=True, scale_each=True)
        writer.add_image('gt class '+str(i+1)+' training', gt_image, n_iter)
    bg_gt_image = vutils.make_grid(bg_gt[0,::,::], normalize=True, scale_each=True)
    writer.add_image('gt background training', bg_gt_image, n_iter)

    # net parameters histograms
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)


    # Validation   
    loss_sum = 0.

    for i_batch, sample_batched in enumerate(validation_dataloader):
        output =  net(sample_batched['input'])
        gts = [get_bg_gt(sample_batched['gt'])]+sample_batched['gt']
        loss = loss_function(output, gts)
        loss_sum += loss.item()/len(validation_dataloader)

    predictions = torch.argmax(output, 1, keepdim=True)

    # loss
    writer.add_scalar("loss_"+parameters["training"]["loss_function"]+"_validation", loss_sum, n_iter)

    if loss_sum < best_loss:
        torch.save(net, "./runs/"+last_run_dir+"/best_model.pt")

    # metrics
    bg_gt = get_bg_gt(sample_batched['gt'])
    FP, FN, TP, TN = numeric_score(predictions, [bg_gt]+sample_batched['gt'])
    precision = precision_score(FP, FN, TP, TN)
    recall = recall_score(FP, FN, TP, TN)
    specificity = specificity_score(FP, FN, TP, TN)
    iou = intersection_over_union(FP, FN, TP, TN)
    accuracy = accuracy_score(FP, FN, TP, TN)
    for i in range(len(precision)):
        writer.add_scalar("precision_"+str(i)+"_validation", precision[i], n_iter)
        writer.add_scalar("recall_"+str(i)+"_validation", recall[i], n_iter)
        writer.add_scalar("specificity_"+str(i)+"_validation", specificity[i], n_iter)
        writer.add_scalar("intersection_over_union_"+str(i)+"_validation", iou[i], n_iter)
        writer.add_scalar("accuracy_"+str(i)+"_validation", accuracy[i], n_iter)

    #images
    input_image = vutils.make_grid(sample_batched['input'][0]/torch.max(sample_batched['input'][0]), normalize=True, scale_each=True)
    writer.add_image('Input image validation', input_image, n_iter)
    output_bg = vutils.make_grid(output[0,0,::,::], normalize=True, scale_each=True)
    pred_bg = vutils.make_grid(predictions[0,0,::,::]==0, normalize=True, scale_each=True)
    writer.add_image('Output background validation', output_bg, n_iter)
    writer.add_image('Prediction background validation', pred_bg, n_iter)
    for i in range(len(sample_batched['gt'])):
        output_image = vutils.make_grid(output[0,i+1,::,::], normalize=True, scale_each=True)
        pred_image = vutils.make_grid(predictions[0,0,::,::]==i+1, normalize=True, scale_each=True)
        writer.add_image('Output class '+str(i+1)+' validation', output_image, n_iter)
        writer.add_image('Prediction class '+str(i+1)+' validation', pred_image, n_iter)
        gt_image = vutils.make_grid(sample_batched['gt'][i][0,::,::], normalize=True, scale_each=True)
        writer.add_image('gt class '+str(i+1)+' validation', gt_image, n_iter)
    bg_gt_image = vutils.make_grid(bg_gt[0,::,::], normalize=True, scale_each=True)
    writer.add_image('gt background validation', bg_gt_image, n_iter)

                
writer.export_scalars_to_json("./runs/"+last_run_dir+"/all_scalars.json")
writer.close()

last_run_dir = os.listdir("./runs")[-1]
torch.save(net, "./runs/"+last_run_dir+"/final_model.pt")

print "training complete, model saved at ./runs/"+last_run_dir+"/model.pt"