## Ths file contains the paths to all the files necessary to load the data and train the model

# paths to the txt files containing the paths to the nifti files (input and gt)
training_data = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/filenames_training.txt"
validation_data = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/filenames_validation.txt"

# path to the json file containing the hyper-parameters
parameters = "/Users/frpau_local/Documents/nih/data/luisa_with_gt/parameters.json" 

# paths to sct directories (used to change input resolution or orientation in pre_processing)
sct_scripts = "/Users/frpau_local/sct_3.1.1/scripts"
sct_dir = "/Users/frpau_local/sct_3.1.1/python/lib/python2.7/site-packages/spinalcordtoolbox"
