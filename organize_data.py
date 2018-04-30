import os
from msct_image import Image
import numpy as np

"""This script copies the data from original folders (source_dir) and pasts it with required organization in the subject_dir. 
It depends on the original data organization and naming.
"""

subject_dir = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/"

source_dir = "/Users/frpau_local/Documents/nih/original_data/240418/"

region_dirs = [name for name in os.listdir(source_dir) if not "." in name]

## CREATE TP FOLDERS 

# for i in range(10) :
# 	os.system("mkdir "+subject_dir+"tp_"+str(i))

tp_dirs = [name for name in os.listdir(subject_dir) if not "." in name]

# print tp_dirs


## CREATE CERV, THOR, LUMB FOLDERS AND UNPACK 4D RAW FILES

# for dir in tp_dirs:
# 	os.system("mkdir "+subject_dir+dir+"/t2s_cerv")
# 	os.system("mkdir "+subject_dir+dir+"/t2s_thor")
# 	os.system("mkdir "+subject_dir+dir+"/t2s_lumb")

# for dir in region_dirs:
# 	images = [name for name in os.listdir(source_dir+dir) if ".nii" in name]

# 	for image in images:
# 		if "no_N4" in image:
# 			os.system("sct_image -i "+source_dir+dir+"/"+image+" -split t")


## PUT RAW DATA IN CORRESPONDING FOLDER

# for tp_dir in tp_dirs:
# 	number = tp_dir[-1]
# 	for region_dir in region_dirs:
# 		for image in os.listdir(source_dir+region_dir):
# 			if "T000"+number in image:
# 				if "lower" in region_dir:
# 					os.system("cp "+source_dir+region_dir+"/"+image+" "+subject_dir+"/tp_"+number+"/t2s_lumb/t2s_lumb.nii")
# 				elif "middle" in region_dir:
# 					os.system("cp "+source_dir+region_dir+"/"+image+" "+subject_dir+"/tp_"+number+"/t2s_thor/t2s_thor.nii")
# 				 elif "upper" in region_dir:
# 				 	os.system("cp "+source_dir+region_dir+"/"+image+" "+subject_dir+"/tp_"+number+"/t2s_cerv/t2s_cerv.nii")


## COPY THOR FOCAL GT

# for image in os.listdir(source_dir+"middle_segment"):
# 	if "focal" in image:
# 		number = image[-5]
# 		os.system("cp "+source_dir+"middle_segment/"+image+" "+subject_dir+"tp_"+number+"/t2s_thor/t2s_thor_focal_manual.nii")


## COPY CERV FOCAL GT

# for image in os.listdir(source_dir+"upper_segment"):
# 	if "focal" in image:
# 		number = image[-11]
# 		os.system("cp "+source_dir+"upper_segment/"+image+" "+subject_dir+"tp_"+number+"/t2s_cerv/t2s_cerv_focal_manual.nii")


## COMPRESS ALL FILES

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		for image in os.listdir(subject_dir+tp_dir+"/"+dir):
# 			if not ".gz" in image:
# 				os.system("gzip "+subject_dir+tp_dir+"/"+dir+"/"+image)


## MOVE LUMB FILES BECAUSE TP2 IS MISSING

# for i in range(2,9)[::-1]:
# 	os.system("mv "+subject_dir+"tp_"+str(i)+"/t2s_lumb/t2s_lumb.nii.gz "+subject_dir+"tp_"+str(i+1)+"/t2s_lumb/t2s_lumb.nii.gz") 


## REORIENT GT FROM CORRUPT RAS TO RAI

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		for im_name in [ image for image in os.listdir(subject_dir+tp_dir+"/"+dir) if ".nii" in image]:
# 			image = Image(subject_dir+tp_dir+"/"+dir+"/"+im_name)
# 			if image.orientation == "RAS":
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient RAI")
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient-data LAS")
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient-data LAI")


# TEMPORALLY SPLIT LUMB FILES

# for tp_dir in tp_dirs:
# 	image_name = subject_dir+tp_dir+"/t2s_lumb/t2s_lumb.nii.gz"
# 	os.system("sct_image -i "+image_name+" -split t")


## TURN GT TO BINARY MASKS

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		for im_name in [ image for image in os.listdir(subject_dir+tp_dir+"/"+dir) if "manual.nii" in image]:
# 			print im_name
# 			image = Image(subject_dir+tp_dir+"/"+dir+"/"+im_name)
# 			image.data[image.data<1]=0
# 			image.data[image.data>=1]=1
# 			image.save()


## REMOVE EMPTY SLICES

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		gt_3D = []
# 		for im_name in [ image for image in os.listdir(subject_dir+tp_dir+"/"+dir) if ".nii" in image]:
# 			if not "manual" in im_name:
# 				input_3D = Image(subject_dir+tp_dir+"/"+dir+"/"+im_name)
# 			else : 
# 				gt_3D.append(Image(subject_dir+tp_dir+"/"+dir+"/"+im_name))
# 		slices_to_delete = []
# 		for i in range(input_3D.data.shape[2]):
# 			# print np.max(input_3D.data[::,::,i])
# 			if np.max(input_3D.data[::,::,i])<=200:
# 				print np.max(input_3D.data[::,::,i])
# 				slices_to_delete.append(i)
# 		if slices_to_delete:
# 			print tp_dir
# 			input_3D.data = np.delete(input_3D.data, slices_to_delete, 2)
# 			input_3D.save()
# 			for gt in gt_3D:
# 				gt.data = np.delete(gt.data, slices_to_delete, 2)
# 				gt.save()


## CREATE TRAINING AND VALIDATION FILES

# def create_train_val(region, TP):
# 	gts = ["", "_csf_manual", "_gm_manual", "_nawm_manual"]
# 	for gt in gts:
# 		image = Image(subject_dir+"tp_"+TP+"/t2s_"+region+"/t2s_"+region+gt+".nii.gz")
# 		validation_slices = [0,4,9,14,19]
# 		training_slices = [i for i in range(image.dim[2]) if i not in validation_slices]

# 		image.data = np.delete(image.data, training_slices, 2)
# 		image.setFileName(subject_dir+"tp_"+TP+"/t2s_"+region+"/t2s_"+region+gt+"_validation.nii.gz")
# 		image.save()

# 		image = Image(subject_dir+"tp_"+TP+"/t2s_"+region+"/t2s_"+region+gt+".nii.gz")
# 		image.data = np.delete(image.data, validation_slices, 2)
# 		image.setFileName(subject_dir+"tp_"+TP+"/t2s_"+region+"/t2s_"+region+gt+"_training.nii.gz")
# 		image.save()

# create_train_val("cerv", "0")
# create_train_val("thor", "0")
# create_train_val("lumb", "1")
