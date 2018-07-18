sct_scripts = "/Users/frpau_local/sct_3.1.1/scripts"
sct_dir = "/Users/frpau_local/sct_3.1.1/python/lib/python2.7/site-packages/spinalcordtoolbox"

import os
import sys
sys.path.append(sct_dir)
sys.path.append(sct_scripts)
from msct_image import Image
import numpy as np
import torch
from resample.nipy_resample import resample_image
from sct_image import set_orientation



def segment(network_path, input_path, output_path, net_name):
    network = torch.load(network_path, map_location='cpu')
    network.eval()

    image = Image(input_path)
    
    # orientation
    orientation = image.orientation
    if orientation != network.orientation:
        image = set_orientation(image, network.orientation)

    # resolution
    res_w, res_h = list(np.around(image.dim[4:6], 2))
    res_str = str(res_w)+"x"+str(res_h)
    if res_str != network.resolution:
        image = resample_image(image, network.resolution, 'mm', 'linear', verbose=0)

    # matrix size
    w, h = image.dim[0:2]
    new_w, new_h = network.matrix_size
    w1 = (w-new_w)/2
    w2 = new_w+w1
    h1 = (h-new_h)/2
    h2 = new_h+h1
    input = np.moveaxis(image.data, 2, 0) # use z dim as batchsize
    input = input[:,w1:w2,h1:h2] # crop images
    if len(input.shape)==3:
        input = input.astype('float32').reshape(input.shape[0], 1, input.shape[1], input.shape[2]) # add 1 channel dim
    else:
        input = np.moveaxis(input, 3,1)
    input = torch.Tensor(input)

    output = network(input)

    if output.size()[1]==1:
        predictions = output.detach().numpy()>0.5
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[2], predictions.shape[3])
    else:
        predictions = torch.argmax(output, 1, keepdim=False).numpy()

    class_names = network.class_names
    
    # matrix size
    predictions = np.moveaxis(predictions, 0, 2)
    predictions_uncropped = np.zeros((w, h, predictions.shape[2]))
    predictions_uncropped[w1:w2,h1:h2,:] = predictions
    image.data = predictions_uncropped

    # resolution
    if res_str != network.resolution:
        image = resample_image(image, res_str, 'mm', 'linear', verbose=0)

    # orientation
    if orientation != network.orientation:
        image = set_orientation(image, orientation)

    pred = image.data

    for i in range(len(class_names)):
        image.data = pred==i+1
        file_name = output_path[:-7]+"_"+net_name+"_"+class_names[i]+"_seg.nii.gz"
        image.setFileName(file_name)
        image.save()


    






