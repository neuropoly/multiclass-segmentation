import os

""" This script creates the filenames txt file contaning the paths to the files according to the following structure on each line:
input <path to the input file> <gt class name> <path to the gt file for this class> <gt second class name> <path to the gt file for this second class>

The data are retrieved from the subject_dir, the file is written to the output_path.
"""

subject_dir = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/"
tp_dirs = [name for name in os.listdir(subject_dir) if not "." in name]

output_path = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/filenames.txt"

with open(output_path, 'w') as filenames:

	for tp_dir in tp_dirs:
		for dir in [ dir for dir in os.listdir(subject_dir+tp_dir) if not ".DS_Store" in dir]:
			line = ""
			for im_name in [image for image in os.listdir(subject_dir+tp_dir+"/"+dir) if ".nii" in image]:
				if not "manual" in im_name:
					line+="input "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" "
				else:
					line+=im_name.split('_')[-2]+" "+subject_dir+tp_dir+"/"+dir+"/"+im_name+" "
			if line:
				line+="\n"
			filenames.write(line)

