import os
import sys
import nibabel as nib
from nibabel import Nifti1Image
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to the network model to use (.pt file).", required=True)
parser.add_argument("-i", "--input", help="Path to the input NifTi file.", required=True)
parser.add_argument("-o", "--output", help="Path for the output NifTi file, "\
                    "subscripts with class names will be added at the end.")
parser.add_argument("-t", "--tag", help="Tag to add in the output file name.")
args = parser.parse_args()


def segment(network_path, input_path, output_path, tag=""):
    network = torch.load(network_path, map_location='cpu')
    network.eval()

    image = nib.load(input_path)

    output_path_head, output_path_tail = os.path.split(output_path)
    output_path_head = output_path_head+"/"+output_path_tail.split(".")[0]
    output_path_tail = output_path_tail.replace(output_path_tail.split(".")[0],"")

    if tag:
    	tag = "_"+tag

    # orientation
    # nib.aff2axcodes(image.affine)
    #orientation = image.orientation
    #if orientation != network.orientation:
    #    raise RuntimeError('The orientation of the input must be : '+network.orientation)

    # resolution
    #res_w, res_h = list(np.around(image.dim[4:6], 2))
    #res_str = str(res_w)+"x"+str(res_h)
    #if res_str != network.resolution:
        #raise RuntimeError('The resolution of the input must be : '+network.resolution)

    # matrix size
    w, h = image.shape[0:2]
    new_w, new_h = network.matrix_size
    w1 = (w-new_w)/2
    w2 = new_w+w1
    h1 = (h-new_h)/2
    h2 = new_h+h1
    input = np.moveaxis(image.get_data(), 2, 0) # use z dim as batchsize
    input = input[:,w1:w2,h1:h2] # crop images
    if len(input.shape)==3:
        input = input.astype('float32').reshape(input.shape[0], 1, input.shape[1],
                                                input.shape[2]) # add 1 channel dim
    else:
        input = np.moveaxis(input, 3,1)
    input = torch.Tensor(input)

    output = network(input)

    if output.size()[1]==1:
        predictions = output.detach().numpy()>0.5
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[2],
                                          predictions.shape[3])
    else:
        predictions = torch.argmax(output, 1, keepdim=False).numpy()

    class_names = network.class_names

    # matrix size
    predictions = np.moveaxis(predictions, 0, 2)
    predictions_uncropped = np.zeros((w, h, predictions.shape[2]))
    predictions_uncropped[w1:w2,h1:h2,:] = predictions

    # resolution
    #if res_str != network.resolution:
    #    image = resample_image(image, res_str, 'mm', 'linear', verbose=0)

    # orientation
    #if orientation != network.orientation:
    #    image = set_orientation(image, orientation)

    #pred = image.get_data()

    for i in range(len(class_names)):
        image_seg = Nifti1Image(predictions_uncropped==i+1, None, image.header.copy())
        file_name = output_path_head+tag+"_"+class_names[i]+"_seg"+output_path_tail
        nib.save(image_seg, file_name)
        print "Segmentation of {} saved at {}".format(class_names[i], file_name)


if args.output:
	output_path = args.output
else:
	output_path = args.input
tag = ""
if args.tag:
    tag = args.tag

segment(args.model, args.input, output_path, tag)
