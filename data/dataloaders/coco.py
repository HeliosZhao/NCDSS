#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import copy
import os
import cv2
import glob
import tarfile

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms

import data.dataloaders.custom_transforms as custom_tr
from utils.utils import mkdir_if_missing

ALL_FOLDS = {
    'fold0': [c for c in range(1,21)],
    'fold1': [c for c in range(21,41)],
    'fold2': [c for c in range(41,61)],
    'fold3': [c for c in range(61,81)],
}

num_classes = 81


class COCO_All(data.Dataset):


    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None):
        # Set paths
        self.root = root
        valid_splits = ['train', 'val']
        assert (split in valid_splits)
        self.split = split
        

        _image_name = split + '2014'
        _sem_name = 'masks_' + _image_name

        _image_dir = os.path.join(self.root, _image_name)
        _semseg_dir = os.path.join(self.root, _sem_name)


        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))

        img_list = sorted(glob.glob(os.path.join(_image_dir, '*.jpg')))

        self.images = []
        self.semsegs = []
        

        for img_file in img_list:
            # Semantic Segmentation
            _im_name = os.path.basename(img_file)
            _semseg = os.path.join(_semseg_dir, _im_name)
            # img_file.replace(_image_name, _sem_name)
            _semseg = _semseg[:-4] + '.png'
            # print(_semseg)
            if os.path.isfile(_semseg):
                self.semsegs.append(_semseg)
                self.images.append(img_file)

        # self.images = list(set(self.images))
        # self.semsegs = list(set(self.semsegs))

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))

        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES


class COCO_Base(data.Dataset):


    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, download=False, novel_fold='fold0', base_all=True):
        # Set paths
        self.root = root
        valid_splits = ['train', 'val']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        self.base_classes = [c for c in range(num_classes) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(self.base_classes):
            self.classes_to_index[cl_base] = ind

        _image_name = split + '2014'
        _sem_name = 'masks_' + _image_name

        _image_dir = os.path.join(self.root, _image_name)
        _semseg_dir = os.path.join(self.root, _sem_name)


        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))

        split_dir = 'data/data_split/coco'
        split_file = os.path.join(split_dir, novel_fold, self.split, 'base.txt')


        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

        self.images = list(set(self.images))

        for img_file in self.images:
            # Semantic Segmentation
            _im_name = os.path.basename(img_file)
            _semseg = os.path.join(_semseg_dir, _im_name)
            # img_file.replace(_image_name, _sem_name)
            _semseg = _semseg[:-4] + '.png'
            # print(_semseg)
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        # self.images = list(set(self.images))
        # self.semsegs = list(set(self.semsegs))

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))

        ### in base training the novel classes should be view as background
        for ignore_class in self.novel_classes:
            _semseg[_semseg == ignore_class] = 0
        _semseg_relabel = _semseg.copy()
        
        for k,v in self.classes_to_index.items():
            _semseg[_semseg_relabel == k] = v

        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.base_classes



class COCO_NovelSaliency(data.Dataset):


    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='train', transform=None, download=False, novel_fold='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['train', 'val']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        self.base_classes = [c for c in range(num_classes) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(self.base_classes):
            self.classes_to_index[cl_base] = ind

        _image_name = split + '2014'
        _sem_name = 'masks_' + _image_name
        _sal_name = 'saliency_supervised_model'

        _image_dir = os.path.join(self.root, _image_name)
        _semseg_dir = os.path.join(self.root, _sem_name)
        _sal_dir = os.path.join(self.root, _sal_name)

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))

        split_dir = 'data/data_split/coco'
        split_file = os.path.join(split_dir, novel_fold, self.split, 'novel.txt') ## here fold0/train/base.txt and novel.txt has overlap


        self.images = []
        self.semsegs = []
        self.sals = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            _semseg = os.path.join(_semseg_dir, line_name + '.png')
            _sal = os.path.join(_sal_dir, line_name + '.png')

            # print(_semseg)
            if os.path.isfile(_semseg) and os.path.isfile(_sal) and os.path.isfile(_image):
                self.images.append(_image)
                self.semsegs.append(_semseg)
                self.sals.append(_sal)

        assert (len(self.images) == len(self.semsegs) == len(self.sals))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg

        _sal = self._load_sal(index)
        sample['sal'] = _sal

        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))

        ### in base training the novel classes should be view as background
        for ignore_class in self.novel_classes:
            _semseg[_semseg == ignore_class] = 0

        _semseg_relabel = _semseg.copy()
        
        for k,v in self.classes_to_index.items():
            _semseg[_semseg_relabel == k] = v

        return _semseg

    def _load_sal(self, index):
        _sal = np.array(Image.open(self.sals[index]))
        return _sal

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES


class COCO_NovelClustering(data.Dataset):
    
    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]

    def __init__(self, root='',
                 split='train', transform=None, novel_dir='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['train', 'val']
        assert (split in valid_splits)
        self.split = split

        _image_dir = os.path.join(self.root, 'train2014')
        _sal_dir = os.path.join(novel_dir, 'novel_map')


        # Transform
        self.transform = transform
        split_file = os.path.join(novel_dir, 'novel.txt')
        self.images = []
        # self.semsegs = []
        self.sals = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)
            _sal = os.path.join(_sal_dir, line_name + '.png')
            assert os.path.isfile(_sal)
            self.sals.append(_sal)
            
        assert (len(self.images) == len(self.sals))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        _sal = self._load_sal(index)

        sample['sal'] = _sal
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.sals[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_sal(self, index):
        _sal = np.array(Image.open(self.sals[index]))
        return _sal

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

### stage 2 novel class finetuning
class COCO_Basic_Train(data.Dataset):
    
    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='base', transform=None, novel_dir='', novel_fold='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['base', 'novel']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(num_classes) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        if split == 'base':
            # _sem_name = 'SegmentationClassAug'
            _semseg_dir = os.path.join(self.root, 'masks_train2014')
        else:
            # _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(novel_dir, 'pseudo_labels')

        _image_dir = os.path.join(self.root, 'train2014')

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))
        # split_dir = 'data/splits/pascal'
        # split_dir = '../splits/pascal'
        split_file = os.path.join(novel_dir, self.split + '.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)
            _semseg = os.path.join(_semseg_dir, line_name + ".png")
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))        
        ## novel classes are always > 15
        ## this is in ordered
        if self.split == 'base':
            ## unique before relabel
            label = np.unique(np.array(_semseg))
            overlap_labels = np.intersect1d(label, self.novel_classes)
            assert len(overlap_labels) == 0 ## base map should not have overlap with novel class

            _semseg_relabel = copy.deepcopy(_semseg)
            for k,v in self.classes_to_index.items():
                _semseg[_semseg_relabel == k] = v

        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES



### stage 2 novel class finetuning
class COCO_NovelFinetuning_Val(data.Dataset):
    
    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, novel_fold='fold0'):
        # Set paths
        self.root = root
        # valid_splits = ['base', 'novel']
        assert split == 'val'
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(num_classes) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        all_classes = base_classes + self.novel_classes ## index : 0 background 1-15 base class 16-20 novel class
        self.classes_to_index = {}
        for ind, cl_base in enumerate(all_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        _semseg_dir = os.path.join(self.root, 'masks_val2014')


        _image_dir = os.path.join(self.root, 'val2014')

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))
        # split_dir = 'data/splits/pascal'
        # split_dir = '../splits/pascal'
        split_file = os.path.join(self.root, self.split + '2014.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)
            _semseg = os.path.join(_semseg_dir, line_name + ".png")
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))

        _semseg_relabel = copy.deepcopy(_semseg)

        for k,v in self.classes_to_index.items():
            _semseg_relabel[_semseg == k] = v

        return _semseg_relabel

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES


class COCO_EUMS(data.Dataset):
    
    VOC_CATEGORY_NAMES = [str(c) for c in range(num_classes)]

    def __init__(self, root='',
                 split='base', transform=None, novel_dir='', novel_fold='fold0', split_dir=''):
        # Set paths
        self.root = root
        valid_splits = ['base', 'novel']
        if split_dir:
            valid_splits = ['easy', 'hard']
            
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(num_classes) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        if split == 'base':
            _semseg_dir = os.path.join(self.root, 'masks_train2014')
        else:
            # _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(novel_dir, 'pseudo_labels')

        _image_dir = os.path.join(self.root, 'train2014')


        # Transform
        if self.split == 'hard' and isinstance(transform, tuple):
            ## at this time, strong and weak transform
            self.transform, self.strong_transform = transform
            self.nomal_transform = transforms.Compose([custom_tr.ToTensor(),
                                    custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            
        else:
            self.transform = transform
            self.strong_transform = None
            self.nomal_transform = None

        # Splits are pre-cut
        print("Initializing dataloader for COCO 2014 {} set".format(''.join(self.split)))
        # split_dir = 'data/splits/pascal'
        # split_dir = '../splits/pascal'
        if split_dir:
            # split_file = os.path.join(novel_dir, split_dir, self.split + '_split.txt')
            split_file = os.path.join(split_dir, self.split + '_split.txt')
        else:
            split_file = os.path.join(novel_dir, self.split + '.txt')

        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)
            _semseg = os.path.join(_semseg_dir, line_name + ".png")
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        # self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        if self.strong_transform is not None:
            sample['strong'] = self.strong_transform(Image.fromarray(np.uint8(sample['image'])).convert('RGB'))
        
        if self.nomal_transform is not None:
            sample = self.nomal_transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index]))        
        ## novel classes are always > 15
        ## this is in ordered
        if self.split == 'base':
            ## unique before relabel
            label = np.unique(np.array(_semseg))
            overlap_labels = np.intersect1d(label, self.novel_classes)
            assert len(overlap_labels) == 0 ## base map should not have overlap with novel class

            _semseg_relabel = copy.deepcopy(_semseg)
            for k,v in self.classes_to_index.items():
                _semseg[_semseg_relabel == k] = v

        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))
    def __str__(self):
        return 'COCO2014(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

