import numpy as np
from torch.utils.data import Dataset, DataLoader
from msct_image import Image as msct_Image


class MRI2DSegDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.
    
    :param txt_path_file: the path to a txt file containing the list of paths to input data files and gt masks.
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, txt_path_file, slice_axis=2, cache=True, transform=None):
        self.filenames = []
        self.header = {}
        self.class_names = []
        self.read_filenames(txt_path_file)
        self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.handlers = []
        
        self._load_files()
    
    def __len__(self):
        return len(self.handlers)
    
    def __getitem__(self, index):
        sample = self.handlers[index]
        data_dict = {
            'input': sample[0],
            'gt': [sample[i] for i in range(1, len(sample))]
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict
        
    
    def _load_files(self):
        for input_filename, gt_dict in self.filenames:
            input_3D = msct_Image(input_filename)
            if self.slice_axis == 0:
                resolution = list(np.around(input_3D.dim[5:7], 2))
                matrix_size = input_3D.dim[1:3]
            elif self.slice_axis == 1:
                resolution = list(np.around([input_3D.dim[4], input_3D.dim[6]], 2))
                matrix_size = (input_3D.dim[0], input_3D.dim[2])
            else:
                if self.slice_axis != 2:
                    print "Invalid slice axis given, replaced by default value of 2."
                    self.slice_axis = 2
                resolution = list(np.around(input_3D.dim[4:6], 2))
                matrix_size = input_3D.dim[0:2]
                
            input_header = {"orientation":input_3D.orientation, "resolution":resolution, "matrix_size":matrix_size}
            
            gt_3D = []
            gt_class_names = sorted(gt_dict.keys())
            for gt_class in gt_class_names:
                gt_3D.append(msct_Image(gt_dict[gt_class]))
                  
            if not self.header:
                self.header = input_header
            #sanity check for consistent header
            elif self.header != input_header :
                print self.header
                print input_header
                raise RuntimeError('Inconsistent header in input files.')
                
            if not self.class_names:
                self.class_names = gt_class_names 
            #sanity check for consistent gt classes
            elif self.class_names != gt_class_names:
                raise RuntimeError('Inconsistent classes in gt files.')
                
            for i in range(input_3D.dim[2]):
                if self.slice_axis == 0:
                    input_slice = input_3D.data[i,::,::]
                    gt_slices = [gt.data[i,::,::] for gt in gt_3D]
                elif self.slice_axis == 1:
                    input_slice = input_3D.data[::,i,::]
                    gt_slices = [gt.data[::,i,::] for gt in gt_3D]
                else:
                    input_slice = input_3D.data[::,::,i]
                    gt_slices = [gt.data[::,::,i] for gt in gt_3D]
                seg_item = [input_slice]
                for gt_slice in gt_slices:
                    if gt_slice.shape != input_slice.shape:
                        print "input dimensions : {}".format(input_slice.shape)
                        print "gt dimensions : {}".format(gt_slice.shape)
                        raise RuntimeError('Input and ground truth with different dimensions.')
                    seg_item.append(gt_slice)
                self.handlers.append(np.array(seg_item))
                
    
    def read_filenames(self, txt_path_file):
        for line in open(txt_path_file, 'r'):
            if "input" in line:
                fnames=[None, {}]
                line = line.split()
                if len(line)%2:
                    raise RuntimeError('Error in filenames txt file parsing.')
                for i in range(len(line)/2):
                    try:
                        msct_Image(line[2*i+1])
                    except Exception:
                        raise RuntimeError("Invalid path in filenames txt file.")
                    if(line[2*i]=="input"):
                        fnames[0]=line[2*i+1]
                    else:
                        fnames[1][line[2*i]]=line[2*i+1]
                self.filenames.append((fnames[0], fnames[1]))