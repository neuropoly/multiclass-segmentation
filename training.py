import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to the patient directory (which should contain parameters.json, filenames_training.txt and filenames_validation.txt files)")
parser.add_argument("--GPU", help="define the number of the GPU to use", type=int)
args = parser.parse_args()

patient_directory = args.path

gpu_number = '1' # number of the GPU to use
if args.GPU:
    gpu_number = str(args.GPU)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number 

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
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
from models import UNet
import losses
from metrics import *




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

    writer.add_scalar("loss_"+tag, loss, epoch)
    for i in range(len(precision)):
        writer.add_scalar("precision_"+str(i)+"_"+tag, precision[i], epoch)
        writer.add_scalar("recall_"+str(i)+"_"+tag, recall[i], epoch)
        writer.add_scalar("specificity_"+str(i)+"_"+tag, specificity[i], epoch)
        writer.add_scalar("intersection_over_union_"+str(i)+"_"+tag, iou[i], epoch)
        writer.add_scalar("accuracy_"+str(i)+"_"+tag, accuracy[i], epoch)


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
    input_max = max(torch.max(input), 0.00000001)
    input_image = vutils.make_grid(input/input_max, normalize=True)
    writer.add_image('Input '+tag, input_image, epoch)
    for i in range(len(gts)):
        output_image = vutils.make_grid(output[i,::,::], normalize=True)
        writer.add_image('Output class '+str(i)+' '+tag, output_image, epoch)
        pred_image = vutils.make_grid(predictions==i, normalize=False)
        writer.add_image('Prediction class '+str(i)+' '+tag, pred_image, epoch)
        gt_image = vutils.make_grid(gts[i], normalize=True)
        writer.add_image('GT class '+str(i)+' '+tag, gt_image, epoch)



## LOAD HYPERPARAMETERS FROM JSON FILE ##

path_to_json = patient_directory+'/parameters.json'

parameters = json.load(open(path_to_json))

if not "optimizer" in parameters["training"].keys():
    parameters["training"]["optimizer"]="adam"
if not "learning_rate" in parameters["training"].keys():
    parameters["training"]["learning_rate"]=0.00001

# TODO: define default values for parameters



## DEFINE DEVICE ##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print "working on {}".format(device)
if torch.cuda.is_available():
    print "using GPU number {}".format(gpu_number)



## CREATE DATASETS ##

# defining transormations
toTensor = transforms.ToTensor()
toPIL = transforms.ToPIL()
randomVFlip = transforms.RandomVerticalFlip()
randomResizedCrop = transforms.RandomResizedCrop(parameters["transforms"]["crop_size"], scale=parameters["transforms"]["scale_range"], ratio=parameters["transforms"]["ratio_range"])
randomRotation = transforms.RandomRotation(parameters["transforms"]["max_angle"])
elasticTransform = transforms.ElasticTransform(parameters["transforms"]["alpha_range"], parameters["transforms"]["sigma_range"], parameters["transforms"]["elastic_rate"])
centerCrop = transforms.CenterCrop2D(parameters["transforms"]["crop_size"])

# creating composed transformation
# Composed transformations should always contain toPIL as first transformations (since other transforamtions are made to work on PIL images) and toTensor as last transforamtion (since the network is excpecting tensors as input). 
composed = torch_transforms.Compose([toPIL,randomVFlip,randomRotation,randomResizedCrop, elasticTransform, toTensor])
crop_val = torch_transforms.Compose([toPIL, centerCrop, toTensor])

# creating datasets
# Datasets should be created with at least a toTensor transformation or a composed transformation with toTensor as last transformation since the network is excpecting tensors as input.
training_dataset = MRI2DSegDataset(patient_directory+"/filenames_training.txt", transform = composed)
validation_dataset = MRI2DSegDataset(patient_directory+"/filenames_validation.txt", transform = crop_val)

# creating data loaders
training_dataloader = DataLoader(training_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True)



## CREATE NET ##

net = UNet(drop_rate=parameters["net"]["drop_rate"], bn_momentum=parameters["net"]["bn_momentum"])

# To use multiple GPUs :
#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  net = nn.DataParallel(net)

net = net.to(device)



## DEFINE LOSS, OPTIMIZER AND LR SCHEDULE ##

if parameters["training"]["optimizer"]=="sgd":
    if not "sgd_momentum" in parameters["training"]:
        parameters["training"]['sgd_momentum']=0.9
    optimizer = optim.SGD(net.parameters(), lr=parameters["training"]['learning_rate'], momentum=parameters["training"]['sgd_momentum'])
elif parameters["training"]["optimizer"]=="adam":
    optimizer = optim.Adam(net.parameters(), lr=parameters["training"]['learning_rate'])

if parameters["training"]["loss_function"]=="dice":
    loss_function = losses.dice_loss

if parameters["training"]["lr_schedule"]=="cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, parameters["training"]["nb_epochs"])
elif parameters["training"]["lr_schedule"]=="poly":
    if not "poly_schedule_p" in parameters["training"]:
        parameters["training"]['poly_schedule_p']=0.9
    lr_lambda = lambda epoch: (1-epoch/parameters["training"]["nb_epochs"])**parameters["training"]["poly_schedule_p"]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
else:
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)



## TRAINING ##

writer = SummaryWriter()
writer.add_text("hyperparameters", json.dumps(parameters)) # add hyperparameters to description
last_run_dir = sorted(os.listdir("./runs"))[-1] # get the name of the directory of the current run (to save the model in that directory)

best_loss = 0.
batch_length = len(training_dataloader)

for epoch in tqdm(range(parameters["training"]["nb_epochs"])):
    
    loss_sum = 0.
    scheduler.step()
    net.train()

    writer.add_scalar("learning_rate", scheduler.get_lr()[0], epoch)
    
    for i_batch, sample_batched in enumerate(training_dataloader):
        optimizer.zero_grad()
        input = sample_batched['input'].to(device)
        output =  net(input)
        gts = [get_bg_gt(sample_batched['gt'])]+sample_batched['gt'] # make an array of ground truths (with the computed background gt mask)
        loss = loss_function(output, [gt.to(device) for gt in gts])
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()/batch_length

    predictions = torch.argmax(output, 1, keepdim=True).to("cpu") # get predicted class for each pixel (on cpu to compute metrics)

    
    # metrics
    write_metrics(writer, predictions, gts, loss_sum, epoch, "training")

    # images
    input_for_image = sample_batched['input'][0]
    output_for_image = output[0,::,::,::]
    pred_for_image = predictions[0,0,::,::]
    gts_for_image = [gt[0,::,::] for gt in gts]

    write_images(writer, input_for_image, output_for_image, pred_for_image, gts_for_image, epoch, "training")

    
    if "write_param_histograms" in parameters["training"].keys() and parameters["training"]["write_param_histograms"]:
        # write net parameters histograms (make the training significantly slower)
        for name, param in net.named_parameters():
           writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


    ## Validation ##  

    loss_sum = 0.
    net.eval()

    for i_batch, sample_batched in enumerate(validation_dataloader):
        output =  net(sample_batched['input'].to(device))
        gts = [get_bg_gt(sample_batched['gt'])]+sample_batched['gt']
        loss = loss_function(output, [gt.to(device) for gt in gts])
        loss_sum += loss.item()/len(validation_dataloader)

    predictions = torch.argmax(output, 1, keepdim=True).to("cpu")


    if loss_sum < best_loss:
        torch.save(net, "./runs/"+last_run_dir+"/best_model.pt")

    # metrics
    write_metrics(writer, predictions, gts, loss_sum, epoch, "validation")

    #images
    input_for_image = sample_batched['input'][0]
    output_for_image = output[0,::,::,::]
    pred_for_image = predictions[0,0,::,::]
    gts_for_image = [gt[0,::,::] for gt in gts]

    write_images(writer, input_for_image, output_for_image, pred_for_image, gts_for_image, epoch, "validation")


                
writer.export_scalars_to_json("./runs/"+last_run_dir+"/all_scalars.json")
writer.close()

torch.save(net, "./runs/"+last_run_dir+"/final_model.pt")

print "training complete, model saved at ./runs/"+last_run_dir+"/final_model.pt"
