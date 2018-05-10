import torch
import random
import math
import numbers
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class ElasticTransform(object):
    def __init__(self, alpha_range, sigma_range, p=0.5):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p
    
    @staticmethod
    def get_params(alpha_range, sigma_range):
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        return alpha, sigma

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def __call__(self, sample):
        if np.random.random() < self.p:
            param_alpha, param_sigma = self.get_params(self.alpha_range, self.sigma_range)
            
            input_data = np.array(sample['input'])
            input_data = self.elastic_transform(input_data, param_alpha, param_sigma)
            input_data = Image.fromarray(input_data, mode='F')
            
            gt_data = sample['gt']
            for i in range(len(gt_data)):
                gt = np.array(gt_data[i])
                gt = self.elastic_transform(gt, param_alpha, param_sigma)
                gt[gt >= 0.5] = 1.0
                gt[gt < 0.5] = 0.0
                gt_data[i] = Image.fromarray(gt, mode='F')

                
            sample['input'] = input_data
            sample['gt'] = gt_data
            
            # for i in range(len(gt_data)) :
            #     gt_data[i].save("gt_"+str(i)+"_elastic.tiff")
        
        return sample

class ToPIL(object):
    def __call__(self, sample):
        sample['input'] = Image.fromarray(np.array(sample['input']), mode='F')
        sample['gt'] = [Image.fromarray(np.array(gt), mode='F') for gt in sample['gt']]
        # for i in range(len(sample['gt'])) :
        #         sample['gt'][i].save("gt_"+str(i)+"_original.tiff")
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        np_input = np.array(sample['input'])
        np_input = np_input.reshape(1, np_input.shape[0], np_input.shape[1])
        np_gt = [np.array(gt) for gt in sample['gt']]
        np_gt = [gt.reshape(1, gt.shape[0], gt.shape[1]) for gt in np_gt]
        sample['input'] = torch.Tensor(np_input)
        sample['gt'] = [torch.Tensor(gt) for gt in np_gt]
        return sample

class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        rdict = {}
        
        input_data = sample['input']
        input_data = F.rotate(input_data, angle, self.resample, self.expand, self.center)
        rdict['input'] = input_data
        
        gt_data = sample['gt']
        gt_data = [F.rotate(gt, angle, self.resample, self.expand, self.center) for gt in gt_data]
        rdict['gt'] = gt_data
        
        # for i in range(len(gt_data)) :
        #     gt_data[i].save("gt_"+str(i)+"_rotation.tiff")
            
        return rdict
    

class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size[0], size[1])
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        i, j, h, w = self.get_params(sample['input'], self.scale, self.ratio)
        rdict = {}
        
        input_data = F.resized_crop(sample['input'], i, j, h, w, self.size, self.interpolation)
        
        gt_data = [F.resized_crop(gt, i, j, h, w, self.size, self.interpolation) for gt in sample['gt']]
        for i in range(len(gt_data)):
            gt = np.array(gt_data[i])
            gt[gt >= 0.5] = 1.0
            gt[gt < 0.5] = 0.0
            gt_data[i] = Image.fromarray(gt, mode='F')

        rdict['input'] = input_data
        rdict['gt'] = gt_data

        # for i in range(len(gt_data)) :
        #     gt_data[i].save("gt_"+str(i)+"_resize.tiff")
        
        return rdict
    
    
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['input'] = F.vflip(sample['input'])
            sample['gt'] = [F.vflip(gt) for gt in sample['gt']]
        return sample


class CenterCrop2D(object):
    """Make a center crop of a specified size.

    :param segmentation: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        sample['input'] = F.center_crop(sample['input'], self.size)
        sample['gt'] = [F.center_crop(gt, self.size) for gt in sample['gt']]
        return sample