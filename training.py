import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to the patient directory (which should contain parameters.json, filenames_training.txt and filenames_validation.txt files)")
parser.add_argument("--GPU", help="define the number of the GPU to use", type=int)
args = parser.parse_args()

patient_directory = args.path
if patient_directory[-1]=="/":
    patient_directory = patient_directory[:-1]

gpu_number = '7' # number of the GPU to use
if args.GPU:
    gpu_number = str(args.GPU)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number 

import torch
from dataset import *
import transforms
import json
from torchvision import transforms as torch_transforms
from tensorboardX import SummaryWriter
#import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import UNet
import losses
import monitoring




## LOAD HYPERPARAMETERS FROM JSON FILE ##

path_to_json = patient_directory+'/parameters.json'

parameters = json.load(open(path_to_json))


## DEFINE DEVICE ##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print "working on {}".format(device)
if torch.cuda.is_available():
    print "using GPU number {}".format(gpu_number)



## CREATE DATASETS ##

# defining transormations
randomVFlip = transforms.RandomVerticalFlip()
randomResizedCrop = transforms.RandomResizedCrop(parameters["input"]["matrix_size"], scale=parameters["transforms"]["scale_range"], ratio=parameters["transforms"]["ratio_range"], dtype=parameters['input']['data_type'])
randomRotation = transforms.RandomRotation(parameters["transforms"]["max_angle"])
elasticTransform = transforms.ElasticTransform(alpha_range=parameters["transforms"]["alpha_range"], sigma_range=parameters["transforms"]["sigma_range"], p=parameters["transforms"]["elastic_rate"], dtype=parameters['input']['data_type'])
channelShift = transforms.ChannelShift(parameters["transforms"]["channel_shift_range"], dtype=parameters['input']['data_type'])
centerCrop = transforms.CenterCrop2D(parameters["input"]["matrix_size"])

# creating composed transformation
composed = torch_transforms.Compose([randomVFlip,randomRotation,randomResizedCrop, elasticTransform])

# creating datasets
training_dataset = MRI2DSegDataset(patient_directory+"/filenames_training.txt", matrix_size=parameters["input"]["matrix_size"], orientation=parameters["input"]["orientation"], resolution=parameters["input"]["resolution"], transform = composed)
validation_dataset = MRI2DSegDataset(patient_directory+"/filenames_validation.txt", matrix_size=parameters["input"]["matrix_size"], orientation=parameters["input"]["orientation"], resolution=parameters["input"]["resolution"])

# creating data loaders
training_dataloader = DataLoader(training_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=parameters["training"]["batch_size"], shuffle=True, drop_last=False)



## CREATE NET ##

net = UNet(nb_input_channels=1, class_names=training_dataset.class_names, drop_rate=parameters["net"]["drop_rate"], bn_momentum=parameters["net"]["bn_momentum"], mean=training_dataset.mean, std=training_dataset.std, orientation=parameters["input"]["orientation"], resolution=parameters["input"]["resolution"], matrix_size=parameters["input"]["matrix_size"])

# To use multiple GPUs :
#if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
#  net = nn.DataParallel(net)

net = net.to(device)



## DEFINE LOSS, OPTIMIZER AND LR SCHEDULE ##

# OPTIMIZER
if parameters["training"]["optimizer"]=="sgd":
    if not "sgd_momentum" in parameters["training"]:
        parameters["training"]['sgd_momentum']=0.9
    optimizer = optim.SGD(net.parameters(), lr=parameters["training"]['learning_rate'], momentum=parameters["training"]['sgd_momentum'])
elif parameters["training"]["optimizer"]=="adam":
    optimizer = optim.Adam(net.parameters(), lr=parameters["training"]['learning_rate'])

# LOSS
if parameters["training"]["loss_function"]=="dice":

    if not "dice_square" in parameters["training"]:
        parameters["training"]['dice_square']=True

    if not "dice_smooth" in parameters["training"]:
        parameters["training"]['dice_smooth']=0.001

    loss_function = losses.Dice(square=parameters["training"]['dice_square'], smooth=parameters["training"]['dice_smooth'])
    
elif parameters["training"]["loss_function"]=="crossentropy":

    loss_function = losses.CrossEntropy()

# LR SCHEDULE
if parameters["training"]["lr_schedule"]=="cosine":

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, parameters["training"]["nb_epochs"])

elif parameters["training"]["lr_schedule"]=="poly":

    if not "poly_schedule_p" in parameters["training"]:
        parameters["training"]['poly_schedule_p']=0.9

    lr_lambda = lambda epoch: (1-float(epoch)/parameters["training"]["nb_epochs"])**parameters["training"]["poly_schedule_p"]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

else:
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)



## TRAINING ##

writer = SummaryWriter()
writer.add_text("hyperparameters", json.dumps(parameters)) # add the hyperparameters to the description
log_dir = writer.file_writer.get_logdir() # get the name of the directory of the current run (to save the model in that directory)


best_loss = float("inf")
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
        #gts = [get_bg_gt(sample_batched['gt'])]+sample_batched['gt'] # make an array of ground truths (with the computed background gt mask)
        gts = sample_batched['gt']
        loss = loss_function(output, gts.to(device))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()/batch_length

    predictions = torch.argmax(output, 1, keepdim=True).to("cpu") # get predicted class for each pixel (on cpu to compute metrics)

    
    # metrics
    monitoring.write_metrics(writer, predictions, gts, loss_sum, epoch, "training")

    # images
    input_for_image = sample_batched['input'][0]
    output_for_image = output[0,:,:,:]
    pred_for_image = predictions[0,0,:,:]
    gts_for_image = gts[0]

    monitoring.write_images(writer, input_for_image, output_for_image, pred_for_image, gts_for_image, epoch, "training")

    
    if "write_param_histograms" in parameters["training"].keys() and parameters["training"]["write_param_histograms"]:
        # write net parameters histograms (make the training significantly slower)
        for name, param in net.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


    ## Validation ##  

    loss_sum = 0.
    net.eval()

    for i_batch, sample_batched in enumerate(validation_dataloader):
        output =  net(sample_batched['input'].to(device))
        gts = sample_batched['gt']
        loss = loss_function(output, gts.to(device))
        loss_sum += loss.item()/len(validation_dataloader)

    predictions = torch.argmax(output, 1, keepdim=True).to("cpu")


    if loss_sum < best_loss:
        torch.save(net, "./"+log_dir+"/best_model.pt")

    # metrics
    monitoring.write_metrics(writer, predictions, gts, loss_sum, epoch, "validation")

    #images
    input_for_image = sample_batched['input'][0]
    output_for_image = output[0,:,:,:]
    pred_for_image = predictions[0,0,:,:]
    gts_for_image = gts[0]

    monitoring.write_images(writer, input_for_image, output_for_image, pred_for_image, gts_for_image, epoch, "validation")


                
writer.export_scalars_to_json("./"+log_dir+"/all_scalars.json")
writer.close()

torch.save(net, "./"+log_dir+"/final_model.pt")

print "training complete, model saved at ./"+log_dir+"/final_model.pt"
