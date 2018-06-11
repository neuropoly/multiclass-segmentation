# nih segmentaion

<h4 style="color:red"> Still under development </h4>

This repo contains tools to train segmentation networks using Pytorch.
An example of use is visible in the **training.py** file. 

- The **dataset.py** file contains classes used to create datasets that inherit the Pytorch dataset class and thus can be used to make dataloader objects. 
They are initialized with txt files containing the paths to the nifti files for the inputs and the ground truths. The data is stored as numpy arrays but returned as torch tensors. They can be provided with transformations for data augmentation. 
All files used for training (and validation) must share the same orientation, resolution and pixel/voxel sizes.


- The **transfroms.py** file contains classes used to create transformations that can be provided to a dataset for data augmentation. The transformations are functions that take as argument and return PIL images. 

- The **models.py** file contains classes that implement network architectures.

- The **losses.py** file contains calsses that implement different loss functions adapted to segmentation problems.

- The **metrics.py** file contains functions that compute different metrics to monitor the results of the model. 

- The **monitoring** file contains functions to write metrics and images to the tensorboard dashboard.


The notebooks folder contains jupyter notebooks that show how to use the classes and functions. 
