import random
import math
import numbers
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image as PIL_Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates




class ElasticTransform(object):
    """Elastic transformation.
    Args:
        alpha_range (tuple): range of alpha value
        sigma_range (tuple): range of sigma value
        p (float): probability of applying the transformation
        dtype (string): data type to use for numpy array
    """
    def __init__(self, alpha_range, sigma_range, dtype, p=0.5):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p
        self.dtype = dtype

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

            input_data = [np.array(input, dtype=self.dtype) for input in sample['input']]
            input_data = [self.elastic_transform(input, param_alpha, param_sigma) for input in input_data]
            input_data = [PIL_Image.fromarray(input) for input in input_data]

            gt_data = sample['gt']
            for i in range(len(gt_data)):
                gt = np.array(gt_data[i], dtype=self.dtype)
                gt = self.elastic_transform(gt, param_alpha, param_sigma)
                gt[gt >= 0.5] = 1.0
                gt[gt < 0.5] = 0.0
                gt_data[i] = PIL_Image.fromarray(gt)


            sample['input'] = input_data
            sample['gt'] = gt_data

        return sample


class RandomRotation(object):
    """Rotation of random angle.
    Args:
        degrees (float or tuple): angle range (if it is a single float a, the range will be [-a,a])
    """
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
        input_data = [F.rotate(input, angle, self.resample, self.expand, self.center) for input in input_data]
        rdict['input'] = input_data

        gt_data = sample['gt']
        gt_data = [F.rotate(gt, angle, self.resample, self.expand, self.center) for gt in gt_data]
        rdict['gt'] = gt_data

        return rdict


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.
    This crop is finally resized to given size.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, dtype, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=PIL_Image.BILINEAR):
        self.size = (size[0], size[1])
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.dtype = dtype

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
        i, j, h, w = self.get_params(sample['input'][0], self.scale, self.ratio)
        rdict = {}

        input_data = [F.resized_crop(input, i, j, h, w, self.size, self.interpolation) for input in sample['input']]

        gt_data = [F.resized_crop(gt, i, j, h, w, self.size, self.interpolation) for gt in sample['gt']]
        for i in range(len(gt_data)):
            gt = np.array(gt_data[i], dtype=self.dtype)
            gt[gt >= 0.5] = 1.0
            gt[gt < 0.5] = 0.0
            gt_data[i] = PIL_Image.fromarray(gt)

        rdict['input'] = input_data
        rdict['gt'] = gt_data

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
            sample['input'] = [F.vflip(input) for input in sample['input']]
            sample['gt'] = [F.vflip(gt) for gt in sample['gt']]
        return sample


class ChannelShift(object):
    """Make a center crop of a specified size.
    Args:
        max_range (int): range of percentage of the maximum pixel value to use as
                         shift value (e.g. if max_range=20, the shift value will be
                         randomly selected between -0.2*max(input) and 0.2*max(input))
        dtype (string): the data type to use while converting to numpy array (e.g. "float32")
    """
    def __init__(self, max_range, dtype):
        self.max_range = max_range
        self.dtype = dtype

    def __call__(self, sample):
        input_np = [np.array(input, dtype=self.dtype) for input in sample['input']]
        shift = random.uniform(-1, 1)*self.max_range/100.*(np.max(input_np))
        input_np = [input + shift for input in input_np]
        sample['input'] = [PIL_Image.fromarray(input) for input in input_np]
        return sample


class CenterCrop2D(object):
    """Make a center crop of a specified size.
    Args:
        size (tuple): expected output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        sample['input'] = F.center_crop(sample['input'], self.size)
        sample['gt'] = [F.center_crop(gt, self.size) for gt in sample['gt']]
        return sample
