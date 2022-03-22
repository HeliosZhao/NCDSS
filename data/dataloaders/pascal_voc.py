#

import copy
import os
import cv2
import tarfile

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms

from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
import data.dataloaders.custom_transforms as custom_tr
from utils.utils import mkdir_if_missing

ALL_FOLDS = {
    'fold0': [c for c in range(1,6)],
    'fold1': [c for c in range(6,11)],
    'fold2': [c for c in range(11,16)],
    'fold3': [c for c in range(16,21)],
    'fold4': [8,10,12,13,17]
}


class VOC12(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, root='',
                 split='val', transform=None, download=False, ignore_classes=[]):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert(split in valid_splits)
        self.split = split
         
        if split == 'trainaug':
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert(len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

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

        for ignore_class in self.ignore_classes:
            _semseg[_semseg == ignore_class] = 255
        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')



class VOC12_Base(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, download=False, novel_fold='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(21) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind

        if split == 'trainaug':
            _sem_name = 'SegmentationClassAug'
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        split_dir = 'data/data_split'
        if self.split == 'trainaug':
            split_file = os.path.join(split_dir, 'base_train_' + novel_fold + '.txt')

        elif self.split == 'val':
            split_file = os.path.join(split_dir, 'base_val_' + novel_fold + '.txt')


        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            line_name = line.split('__')[0]
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

        self.images = list(set(self.images))

        for img_file in self.images:
            # Semantic Segmentation
            _semseg = img_file.replace('images', _sem_name)
            _semseg = _semseg[:-4] + '.png'
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        # self.images = list(set(self.images))
        # self.semsegs = list(set(self.semsegs))

        assert (len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))


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
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


class VOC12_NovelSaliency(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, download=False, novel_fold='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(21) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind

        if split == 'trainaug':
            _sem_name = 'SegmentationClassAug'
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')
        _sal_dir = os.path.join(self.root, 'saliency_supervised_model')

        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} Novel set".format(''.join(novel_fold)))

        split_dir = 'data/data_split'
        split_file = os.path.join(split_dir, novel_fold, 'novel.txt')
        self.images = []
        self.semsegs = []
        self.sals = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            # line_name = line.split('__')[0]
            line_name = line
            _image = os.path.join(_image_dir, line_name + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

        self.images = list(set(self.images))

        remove_files = []

        for img_file in self.images:
            # Semantic Segmentation
            _semseg = img_file.replace('images', _sem_name)
            _semseg = _semseg[:-4] + '.png'
            _sal = _semseg.replace(_sem_name, 'saliency_supervised_model')
            if os.path.isfile(_semseg) and os.path.isfile(_sal):
            # assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)
                self.sals.append(_sal)
            else:
                remove_files.append(img_file)

        for rf in remove_files:
            self.images.remove(rf)

        assert (len(self.images) == len(self.semsegs) == len(self.sals))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

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

        for k,v in self.classes_to_index.items():
            _semseg[_semseg == k] = v
        return _semseg

    def _load_sal(self, index):
        _sal = np.array(Image.open(self.sals[index]))
        return _sal

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


class VOC12_NovelClustering(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, download=False, novel_dir='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert (split in valid_splits)
        self.split = split

        _image_dir = os.path.join(self.root, 'images')
        _sal_dir = os.path.join(novel_dir, 'novel_map')

        # Download
        if download:
            self._download()

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
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

### stage 2 novel class finetuning
class VOC12_Basic_Train(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='base', transform=None, download=False, novel_dir='', novel_fold='fold0'):
        # Set paths
        self.root = root
        valid_splits = ['base', 'novel']
        assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(21) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        if split == 'base':
            _sem_name = 'SegmentationClassAug'
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            # _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(novel_dir, 'pseudo_labels')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
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
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

### stage 2 novel class finetuning
class VOC12_NovelFinetuing_Val(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='val', transform=None, download=False, novel_dir='', novel_fold='fold0'):
        # Set paths
        self.root = root
        # valid_splits = ['base', 'novel']
        # assert (split in valid_splits)
        self.split = split
        
        assert (novel_fold in ALL_FOLDS.keys())
        self.novel_fold = novel_fold  ## fold name
        self.novel_classes = ALL_FOLDS[novel_fold] ## the five class index
        base_classes = [c for c in range(21) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        all_classes = base_classes + self.novel_classes ## index : 0 background 1-15 base class 16-20 novel class
        self.classes_to_index = {}
        for ind, cl_base in enumerate(all_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')


        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

        # Transform
        self.transform = transform

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        #
        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
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

        # 

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
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

class VOC12_EUMS(data.Dataset):
    
    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'

    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # ALL_FOLDS = ['fold0', 'fold1', 'fold2', 'fold3']

    def __init__(self, root='',
                 split='base', transform=None, download=False, novel_dir='', novel_fold='fold0', split_dir=''):
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
        base_classes = [c for c in range(21) if c not in self.novel_classes]  ## from 0 -- 20 , except for the novel classes; 0 denotes bg
        self.classes_to_index = {}
        for ind, cl_base in enumerate(base_classes):
            self.classes_to_index[cl_base] = ind
        ## novel class is from 16 to 20

        if split == 'base':
            _sem_name = 'SegmentationClassAug'
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            # _sem_name = 'SegmentationClass'
            _semseg_dir = os.path.join(novel_dir, 'pseudo_labels')

        _image_dir = os.path.join(self.root, 'images')


        # Download
        if download:
            self._download()

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
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        # 
        if split_dir:
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

        # 

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
        return 'VOC12(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


