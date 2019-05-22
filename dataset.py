import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import math
import nibabel as nib



class MRI2DSegDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

        The paths to the nifti files must be contained in a txt file following
        the structure (for an example with 2 classes):

        input <path to input 1> class_1 <path to class 1 gt mask of input 1> class_2 <path to class 2 gt mask of input 1>
        input <path to input 2> class_1 <path to class 1 gt mask of input 2> class_2 <path to class 2 gt mask of input 2>

        class_1 and class_2 can be any string (with no space) that will be used
        as class names.
        For multi-class segmentation, there is no need to provide the background
        mask, it will be computed as the complementary of all other masks. Each
        segmentation class ground truth mus be in different 1 channel file.
        The inputs can be volumes of multichannel 2D images.

    :param txt_path_file: the path to a txt file containing the list of paths to
    input data files and gt masks.
    :param matrix_size: size of the slices (tuple of two integers). If the model
    contains p operations of pooling, the sizes should be multiples of 2^p.
    :param orientation: string describing the orientation to use, e.g. "RAI".
    :param resolution: string describing the resolution to use e.g. "0.15x0.15".
    :param data_type: data type to use for the tensors, e.g. "float32".
    :param transform: transformation to apply for data augmentation.
    The transformation should take as argument and return a PIL image.
    """
    def __init__(self, txt_path_file, matrix_size, orientation, resolution,
                 data_type="float32", transform=None):
        self.filenames = []
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        self.class_names = []
        self.read_filenames(txt_path_file)
        self.data_type = data_type
        self.transform = transform
        self.handlers = []
        self.mean = 0.
        self.std = 0.

        self._load_files()

        # compute std of the whole dataset (for input normalization in network)
        for seg_item in self.handlers:
            self.std += np.mean((seg_item['input']-self.mean)**2)/len(self.handlers)
        self.std = math.sqrt(self.std)


    def __len__(self):
        return len(self.handlers)


    def __getitem__(self, index):
        sample = self.handlers[index]
        sample = self.to_PIL(sample)

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        sample = self.to_tensor(sample)

        if len(sample['gt'])>1:    # if it is a multiclass problem
            # make sure gt masks are not overlapping due to transformations
            sample['gt'] = make_masks_exclusive(sample['gt'])
            sample['gt'] = self.add_background_gt(sample['gt'])

        return sample


    def _load_files(self):
        for input_filename, gt_dict in self.filenames:

            # load input
            input_image = nib.load(input_filename)
            #input_image = check_orientation(input_image, self.orientation)
            #input_image = check_resolution(input_image, self.resolution)

            # get class names
            gt_class_names = sorted(gt_dict.keys())
            if not self.class_names:
                self.class_names = gt_class_names
            #sanity check for consistent classes
            elif self.class_names != gt_class_names:
                raise RuntimeError('Inconsistent classes in gt files.')

            # load gts
            gt_nps = []
            for gt_class in gt_class_names:
                gt_image = nib.load(gt_dict[gt_class])
                #gt_image = check_orientation(gt_image, self.orientation)
                #gt_image = check_resolution(gt_image, self.resolution)
                gt_nps.append(gt_image.get_data().astype(self.data_type))

            # compute min and max width and height to crop the arrays to the wanted size
            w, h = input_image.shape[0:2]
            new_w, new_h = self.matrix_size
            if w<new_w or h<new_h:
                print w, h
                raise RuntimeError('Image smaller than required size : {}x{}, '\
                'please provide images of equal or greater size.'.format(new_w, new_h))
            w1 = (w-new_w)/2
            w2 = new_w+w1
            h1 = (h-new_h)/2
            h2 = new_h+h1

            # iterating over the z axis to get each 2D slice
            for i in range(input_image.shape[2]):
                input_slice = input_image.get_data()[w1:w2,h1:h2,i,...].astype(self.data_type)
                if len(input_slice.shape)==2:
                    # if there is only one channel in input, add the channel dimension as first dimension
                    input_slice = np.reshape(input_slice, (1,)+input_slice.shape)
                else:
                    # if there are multiple channel, move axis to have the channel dimension as first dimension
                    input_slice = np.moveaxis(input_slice, 2, 0)
                gt_slices = [gt[w1:w2,h1:h2,i] for gt in gt_nps]

                # compute mean of all the input slices (on 1st channel only, for input normalization in network)
                self.mean += np.mean(input_slice[0,:,:])/(input_image.shape[2]*len(self.filenames))

                #sanity check for no overlap in gt masks
                if np.max(sum(gt_slices))>1:
                    raise RuntimeError('Ground truth masks overlapping in {}.'.format(input_filename))

                seg_item = {"input":input_slice, "gt":np.array(gt_slices)}
                self.handlers.append(seg_item)


    def read_filenames(self, txt_path_file):
        for line in open(txt_path_file, 'r'):
            if "input" in line:
                fnames=[None, {}]
                line = line.split()
                if len(line)%2:
                    raise RuntimeError('Error in data paths text file parsing.')
                for i in range(len(line)/2):
                    try:
                        nib.load(line[2*i+1])
                    except Exception:
                        print line[2*i+1]
                        raise RuntimeError("Invalid path in data paths textt file : {}".format(line[2*i+1]))
                    if(line[2*i]=="input"):
                        fnames[0]=line[2*i+1]
                    else:
                        fnames[1][line[2*i]]=line[2*i+1]
                self.filenames.append((fnames[0], fnames[1]))


    def to_PIL(self, sample):
        # turns a sample of numpy arrays to a sample of PIL images
        sample_pil = {}
        sample_pil['input'] = [Image.fromarray(sample['input'][i]) for i in range(sample['input'].shape[0])]
        sample_pil['gt'] = [Image.fromarray(gt) for gt in sample['gt']]
        return sample_pil


    def to_tensor(self, sample):
        # turns a sample of PIL images to a sample of torch tensors
        np_inputs = [np.array(input, dtype=self.data_type) for input in sample['input']]
        torch_input = torch.stack([torch.tensor(input, dtype=getattr(torch, self.data_type)) for input in np_inputs], dim=0)
        np_gt = [np.array(gt, dtype=self.data_type) for gt in sample['gt']]
        torch_gt = torch.stack([torch.tensor(gt, dtype=getattr(torch, self.data_type)) for gt in np_gt])
        sample_torch = {}
        sample_torch['input'] = torch_input
        sample_torch['gt'] = torch_gt
        return sample_torch


    def add_background_gt(self, gts):
        # create the background mask as complementary to the other gt masks
        gt_size = gts.size()[1:]
        bg_gt = torch.ones(gt_size, dtype=getattr(torch, self.data_type))
        zeros = torch.zeros(gt_size, dtype=getattr(torch, self.data_type))
        for i in range(gts.size()[0]):
            bg_gt = torch.max(bg_gt - gts[i], zeros)
        new_gts = torch.cat((torch.stack([bg_gt]), gts))
        return new_gts



def make_masks_exclusive(gts):
    # make sure gt masks are not overlapping
    indexes = range(len(gts))
    np.random.shuffle(indexes)
    for i in range(len(indexes)):
        for j in range(i):
            gts[indexes[i]][gts[indexes[j]]>=gts[indexes[i]]]=0
    return gts
