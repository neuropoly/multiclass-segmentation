import numpy as np
from msct_image import Image
from scipy import ndimage
from scipy.spatial.distance import cdist
import os




def delete_false_positives(img_path, class_names):
	images = []
	for class_name in class_names:
		images.append(Image(img_path+class_name+"_seg.nii.gz"))
	img = sum([image.data for image in images])

	for i in range(img.shape[2]):
		labeled, nr_objects = ndimage.label(img[:,:,i])

		if nr_objects>1:
			sizes = []
			for k in range(1,nr_objects+1):
				sizes.append(np.sum(labeled==k))
			good_class = np.argmax(sizes)+1
			for k in range(1, nr_objects+1):
				if k != good_class:
					for image in images:
						image.data[:,:,i] = image.data[:,:,i] - (labeled == k)*image.data[:,:,i]
					print img_path, i

	for image in images:
		image.save()



