import os
from msct_image import Image

"""This script copies the data from original folders (source_dir) and pasts it with required organization in the subject_dir. 
It depends on the original data organization and naming.
"""

subject_dir = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/"

source_dir = "/Users/frpau_local/Documents/nih/original_data/marmoset_SC_forCharley/"

region_dirs = [name for name in os.listdir(source_dir) if not "." in name]

## CREATE TIME FOLDERS 

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
# 		if "4D" in image:
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
# 				elif "upper" in region_dir:
# 					os.system("cp "+source_dir+region_dir+"/"+image+" "+subject_dir+"/tp_"+number+"/t2s_cerv/t2s_cerv.nii")


## COPY THOR FOCAL GT

# for image in os.listdir(source_dir+"middle_segment"):
# 	if "focal" in image:
# 		number = image[14]
# 		os.system("cp "+source_dir+"middle_segment/"+image+" "+subject_dir+"tp_"+number+"/t2s_thor/t2s_thor_focal_manual.nii")


## COMPRESS ALL FILES

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		for image in os.listdir(subject_dir+tp_dir+"/"+dir):
# 			if not ".gz" in image:
# 				os.system("gzip "+subject_dir+tp_dir+"/"+dir+"/"+image)




## REORIENT GT FROM CORRUPT RAI TO RAS

# for tp_dir in tp_dirs:
# 	for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
# 		for im_name in [ image for image in os.listdir(subject_dir+tp_dir+"/"+dir) if ".nii" in image]:
# 			image = Image(subject_dir+tp_dir+"/"+dir+"/"+im_name)
# 			if image.orientation == "RAI":
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient RAS")
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient-data LAI")
# 				os.system("sct_image -i "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" -setorient-data LAS")


## TEMPORALLY SPLIT LUMB FILES

for tp_dir in tp_dirs:
	image_name = subject_dir+tp_dir+"/t2s_lumb/t2s_lumb.nii.gz"
	os.system("sct_image -i "+image_name+" -split t")
