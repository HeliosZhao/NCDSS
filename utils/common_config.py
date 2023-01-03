

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToPILImage
import data.dataloaders.custom_transforms as custom_tr
from utils.collate import collate_custom


def get_model(p):
    # Get backbone
    if p['backbone'] == 'resnet18':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet18'](pretrained=True)
        backbone_channels = 512
    
    elif p['backbone'] == 'resnet50':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet50'](pretrained=p['backbone_imagenet'])
        backbone_channels = 2048
    
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['backbone_kwargs']['dilated']:
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)
    
    # Get head
    if p['head'] == 'deeplab':
        if not p['kmeans_eval']:
            nc = p['num_classes'] + int(p['has_bg'])
        else:
            nc = p['model_kwargs']['ndim']

        from models.deeplab import DeepLabHead
        head = DeepLabHead(backbone_channels, nc)

    elif p['head'] == 'dim_reduction':
        nc = p['num_classes'] + int(p['has_bg'])
        import torch.nn as nn
        head = nn.Conv2d(backbone_channels, nc, 1)

    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    if p['kmeans_eval']:
        from models.models import BackBoneModel
        import torch.nn as nn
        if p['backbone_imagenet']:
            model = BackBoneModel(backbone)
    else:
        from models.models import SimpleSegmentationModel
        model = SimpleSegmentationModel(backbone, head)
    

    return model


def get_con_pre_model(p):
    # Get backbone
    if p['backbone'] == 'resnet18':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet18'](pretrained=True)
        backbone_channels = 512
    
    elif p['backbone'] == 'resnet50':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet50'](pretrained=p['backbone_imagenet'])
        backbone_channels = 2048
    
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['backbone_kwargs']['dilated']:
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)
    
    # Get head
    if p['head'] == 'deeplab':
        nc = p['num_classes'] + int(p['has_bg'])

        from models.deeplab import DeepLabHead
        head = DeepLabHead(backbone_channels, nc)

    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    # if p['kmeans_eval']:
    from models.models import ContrastivePredictionModel
    import torch.nn as nn
    model = ContrastivePredictionModel(backbone, head)

    return model


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(root=p['data_root'], split=p['train_db_kwargs']['split'], transform=transform)

    elif p['train_db_name'] == 'BASE':
        from data.dataloaders.pascal_voc import VOC12_Base

        dataset = VOC12_Base(root=p['data_root'], split=p['train_db_kwargs']['split'], transform=transform, novel_fold=p['fold'])

    elif p['train_db_name'] == 'COCO_BASE':
        from data.dataloaders.coco import COCO_Base

        dataset = COCO_Base(root=p['data_root'], split=p['train_db_kwargs']['split'], transform=transform, novel_fold=p['fold'])
    
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(root=p['data_root'], split='val', transform=transform)  

    elif p['val_db_name'] == 'BASE':
        from data.dataloaders.pascal_voc import VOC12_Base
        dataset = VOC12_Base(root=p['data_root'], split='val', transform=transform, novel_fold=p['fold'])        

    elif p['val_db_name'] == 'COCO_BASE':
        from data.dataloaders.coco import COCO_Base
        dataset = COCO_Base(root=p['data_root'], split='val', transform=transform, novel_fold=p['fold'])        

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['train_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['val_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=False, shuffle=False)


def get_train_transformations(augmentation_strategy='pascal'):
    return transforms.Compose([custom_tr.RandomHorizontalFlip(),
                                   custom_tr.ScaleNRotate(rots=(-5,5), scales=(.75,1.25),
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                   custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 'semseg': tuple((512,512))},
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                   custom_tr.ToTensor(),
                                    custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def get_strong_transformations(augmentation_strategy='pascal'):
    ## color jitter + cutout [optional]
    return transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)])

def get_weak_transformations(augmentation_strategy='pascal'):
    return transforms.Compose([custom_tr.RandomHorizontalFlip(),
                                   custom_tr.ScaleNRotate(rots=(-5,5), scales=(.75,1.25),
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                   custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 'semseg': tuple((512,512))},
                                    flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC})])


def get_val_transformations():
    return transforms.Compose([custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                        'semseg': tuple((512,512))},
                                            flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST}),
                                custom_tr.ToTensor(),
                                custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def get_val_transformations_wsal():
    return transforms.Compose([custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                        'semseg': tuple((512,512)),
                                                        'sal': tuple((512,512))},
                                            flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST, 'sal': cv2.INTER_NEAREST}),
                                custom_tr.ToTensor(),
                                custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 0.01

    return lr