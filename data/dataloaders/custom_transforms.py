#
# Code referenced from https://github.com/facebookresearch/astmt

import numpy.random as random
import numpy as np
import torch
import cv2
import math
import data.util.helpers as helpers
import torchvision
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms.functional as F

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False, flagvals=None):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg
        self.flagvals = flagvals

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)
            if self.flagvals is None:
                if ((tmp == 0) | (tmp == 1)).all():
                    flagval = cv2.INTER_NEAREST
                elif 'gt' in elem and self.semseg:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC
            else:
                flagval = self.flagvals[elem]

            if elem == 'normals':
                # Rotate Normals properly
                in_plane = np.arctan2(tmp[:, :, 0], tmp[:, :, 1])
                nrm_0 = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2)
                rot_rad= rot * 2 * math.pi / 360
                tmp[:, :, 0] = np.sin(in_plane + rot_rad) * nrm_0
                tmp[:, :, 1] = np.cos(in_plane + rot_rad) * nrm_0

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())
        for elem in elems:
            if 'meta' in elem or 'bbox' in elem:
                continue

            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])

                    if elem == 'normals':
                        N1, N2, N3 = sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2]
                        Nn = np.sqrt(N1 ** 2 + N2 ** 2 + N3 ** 2) + np.finfo(np.float32).eps
                        sample[elem][:, :, 0], sample[elem][:, :, 1], sample[elem][:, :, 2] = N1/Nn, N2/Nn, N3/Nn
            else:
                del sample[elem]

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomResize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem or 'bbox' in elem:
                continue

            tmp = sample[elem]

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomResize:'+str(self.scales)


class FixedResizeRatio(object):
    """Fixed resize for the image and the ground truth to specified scale.
    Args:
        scales (float): the scale
    """
    def __init__(self, scale=None, flagvals=None):
        self.scale = scale
        self.flagvals = flagvals

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            if elem in self.flagvals:
                if self.flagvals[elem] is None:
                    continue

                tmp = sample[elem]
                tmp = cv2.resize(tmp, None, fx=self.scale, fy=self.scale, interpolation=self.flagvals[elem])

                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'FixedResizeRatio: '+str(self.scale)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

                if elem == 'normals':
                    sample[elem][:, :, 0] *= -1

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            
            elif elem == 'image' or elem == 'strong':
                sample[elem] = self.to_tensor(sample[elem]) # Regular ToTensor operation 
            
            elif elem in ['semseg', 'sal', 'semseg_10']:
                sample[elem] = torch.from_numpy(sample[elem]).long() # Torch Long

            else:
                raise NotImplementedError

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    """Apply normalization to the image """
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean,std)

    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        if 'strong' in sample:
            sample['strong'] = self.normalize(sample['strong'])
        return sample
    
    def __str__(self):
        return self.normalize.__str__()

class ToTensorPIL(object):
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample['image'] = self.to_tensor(sample['image'])
        if 'sal' in sample:
            sample['sal'] = torch.from_numpy(np.array(sample['sal'])).long()
        if 'semseg' in sample:
            sample['semseg'] = torch.from_numpy(np.array(sample['semseg'])).long()

        return sample
    
    def __str__(self):
        return 'ToTensor'

class NPToPIL(object):
    def __call__(self, sample):

        sample['image'] = Image.fromarray(np.uint8(sample['image'])).convert('RGB')
        if 'sal' in sample:
            sample['sal'] = Image.fromarray(np.uint8(sample['sal']))
        if 'semseg' in sample:
            sample['semseg'] = Image.fromarray(np.uint8(sample['semseg']))


        return sample
    
    def __str__(self):
        return 'NPToPIL'


class ColorJitter(object):
    def __init__(self, jitter):
        self.jitter = torchvision.transforms.ColorJitter(jitter[0], jitter[1], jitter[2], jitter[3])

    def __call__(self, sample):
        sample['image'] = self.jitter(sample['image']) 
        return sample

    def __str__(self):
        return 'ColorJitter'


class RandomHorizontalFlipPIL(object):
    def __call__(self, sample):

        if random.random() < 0.5:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            if 'sal' in sample:
                sample['sal'] = sample['sal'].transpose(Image.FLIP_LEFT_RIGHT)
            if 'semseg' in sample:
                sample['semseg'] = sample['semseg'].transpose(Image.FLIP_LEFT_RIGHT)

        return sample
    
    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        img = sample['image']
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            sample['image'] = F.to_grayscale(img, num_output_channels=num_output_channels)
        return sample

    def __str__(self):
        return 'RandomGrayscale'

class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Invalid type size {}'.format(type(size)))
    
        try:
            self.interpolation_img = T.InterpolationMode.BILINEAR
            self.interpolation_sal = T.InterpolationMode.NEAREST
        except:
            self.interpolation_img = Image.BILINEAR
            self.interpolation_sal = Image.NEAREST

        self.resize_img = torchvision.transforms.Resize(self.size, interpolation=self.interpolation_img)
        self.resize_sal = torchvision.transforms.Resize(self.size, interpolation=self.interpolation_sal)

    def __call__(self, sample):
        sample['image'] = self.resize_img(sample['image'])
        
        if 'sal' in sample:
            sample['sal'] = self.resize_sal(sample['sal'])
        if 'semseg' in sample:
            sample['semseg'] = self.resize_sal(sample['semseg'])


        return sample

class RandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super(RandomResizedCrop, self).__init__(size, scale=scale, ratio=ratio)
        try:
            self.interpolation_img = T.InterpolationMode.BILINEAR
            self.interpolation_sal = T.InterpolationMode.NEAREST
        except:
            self.interpolation_img = Image.BILINEAR
            self.interpolation_sal = Image.NEAREST
        
    
    def __call__(self, sample):
        img = sample['image']
        
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation_img)

        if 'sal' in sample:
            sal = sample['sal']
            sample['sal'] = F.resized_crop(sal, i, j, h, w, self.size, self.interpolation_sal)

        if 'semseg' in sample:
            semseg = sample['semseg']
            sample['semseg'] = F.resized_crop(semseg, i, j, h, w, self.size, self.interpolation_sal)

        return sample