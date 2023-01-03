

import argparse
import cv2
import os
import sys
from numpy import triu_indices_from
import torch
import random
import numpy as np

from utils.config import create_config
from utils.common_config import get_train_transformations, get_strong_transformations, get_weak_transformations, \
                                get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, adjust_learning_rate
from utils.train_utils import *
from utils.evaluate_utils import eval_segmentation_full_classes_online
from utils.entropy_ranking_utils import *

from data.dataloaders.pascal_voc import VOC12_EUMS, VOC12_NovelFinetuning_Val
from data.dataloaders.coco import COCO_EUMS, COCO_NovelFinetuning_Val

from models.teacher_student import TeacherStudentModel
from termcolor import colored
from utils.logger import Logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parser
parser = argparse.ArgumentParser(description='Finetune EUMS Framework')
parser.add_argument('--config', type=str,
                    help='Config file for the experiment')
parser.add_argument('--fold', type=str, default='fold0',
                    help='Split fold for novel class')
parser.add_argument('--output-dir', type=str, default='',
                    help='Output dir, if indicate, use it; else, use the time')
parser.add_argument('--novel-dir', type=str, default='',
                    help='novel dir storing the pseudo labels')
parser.add_argument('--nclusters', type=int, default=5,
                    help='Number of novel clusters')
parser.add_argument('--threshold', type=float, default=None,
                    help='threshold for online pseudolabels')
                    
parser.add_argument('--seed', type=int, default=242133,
                    help='Set seed for reproducibility')

parser.add_argument('--data-root', type=str, default=None,
                    help='Dataset root')

parser.add_argument('--split-dir', type=str, default='',
                    help='Split dir for txt, empty for not use easy split')
parser.add_argument('--eval-online', type=str2bool, default='yes',
                    help='eval online for pascal')

args = parser.parse_args()

def set_seed(random_seed):
    print(colored('setting random seed to {}'.format(random_seed), 'green'))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def main():
    cv2.setNumThreads(1)
    # Retrieve config file
    p = create_config(args)
    set_seed(args.seed)

    print(colored('Set CuDNN benchmark for accelerate', 'blue')) 
    torch.backends.cudnn.benchmark = True

    sys.stdout = Logger(p['log_file'])
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = TeacherStudentModel(p)
    # print(model)
    model = model.cuda()
    
    backbone_params, decoder_params = model._parameter_groups(p['ft_layer'])

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion.cuda()
    print(criterion)

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    print(colored('backbone params {}'.format(len(backbone_params))))
    print(colored('decoder params {}'.format(len(decoder_params))))
    parameters = [
        {'params': decoder_params},
        {'params': backbone_params, 'lr': 0.01 * p['optimizer_kwargs']['lr']}
    ]
    optimizer = get_optimizer(p, parameters)
    print(optimizer)

    ## set backbone freeze bn
    assert p['freeze_batchnorm'] == 'backbone'
    model._freeze_backbone_bn()
    
    ## teacher student use different intialization
    model._initialize_params()

    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations()       
    strong_transforms = get_strong_transformations()
    weak_transforms = get_weak_transformations()
    val_transforms = get_val_transformations()

    if p['dataset'] == 'COCO':
        train_dataset_func = COCO_EUMS
        val_dataset_func = COCO_NovelFinetuning_Val
    else:
        train_dataset_func = VOC12_EUMS
        val_dataset_func = VOC12_NovelFinetuning_Val

    base_dataset = train_dataset_func(root=p['data_root'], split='base', transform=train_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'])
    easy_dataset = train_dataset_func(root=p['data_root'], split='easy', transform=train_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'], split_dir=os.path.join(args.novel_dir, args.split_dir))
    hard_dataset = train_dataset_func(root=p['data_root'], split='hard', transform=(weak_transforms,strong_transforms), novel_dir=args.novel_dir, novel_fold=p['fold'], split_dir=os.path.join(args.novel_dir, args.split_dir))
    val_dataset = val_dataset_func(root=p['data_root'], split='val', transform=val_transforms, novel_fold=p['fold'])

    easy_dataloader = get_train_dataloader(p, easy_dataset)
    base_dataloader = get_train_dataloader(p, base_dataset)
    hard_dataloader = get_train_dataloader(p, hard_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)  
    
    print(colored('Base Train samples %d - Novel Easy samples %d - Novel Hard samples %d - Val samples %d' %(len(base_dataset), len(easy_dataset), len(hard_dataset), len(val_dataset)), 'yellow'))

    start_epoch = 0
    best_epoch = 0
    best_iou = 0
    model = model.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']-1), 'yellow'))
        print(colored('-'*10, 'yellow'))

        if epoch in p['entropy']['reassign_epoch']:
            print(colored('Reassign Easy Split at Epoch {} --> Add part of them to hard split'.format(epoch), 'blue'))
            cur_easy_dataset = train_dataset_func(root=p['data_root'], split='easy', transform=val_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'], split_dir=os.path.join(args.novel_dir, args.split_dir))
            cur_easy_dataloader = get_val_dataloader(p, cur_easy_dataset)

            reassign_data_split(p, cur_easy_dataloader, model.model_q)

            ### reassign hard loader and easy loader
            ## hard loader
            hard_new_dataset = train_dataset_func(root=p['data_root'], split='hard', transform=(weak_transforms,strong_transforms), novel_dir=args.novel_dir, novel_fold=p['fold'], split_dir=p['output_dir'])
            hard_dataset = torch.utils.data.ConcatDataset([hard_dataset, hard_new_dataset])
            hard_dataloader = get_train_dataloader(p, hard_dataset)

            easy_dataset = train_dataset_func(root=p['data_root'], split='easy', transform=train_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'], split_dir=p['output_dir'])
            easy_dataloader = get_train_dataloader(p, easy_dataset)
            print(colored('Base Train samples %d - Novel Easy samples %d - Novel Hard samples %d - Val samples %d' %(len(base_dataset), len(easy_dataset), len(hard_dataset), len(val_dataset)), 'yellow'))


        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train_eums(p, base_dataloader, easy_dataloader, hard_dataloader, model, criterion, optimizer, epoch, freeze_batchnorm=p['freeze_batchnorm'])

        # Evaluate online -> This will use batched eval where every image is resized to the same resolution.
        if args.eval_online:
            print('Evaluate ...')
            eval_val = eval_segmentation_full_classes_online(p, val_dataloader, model.model_q)
            
            if eval_val['mIoU'] > best_iou:
                print('Found new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
                best_iou = eval_val['mIoU']
                best_epoch = epoch
                torch.save(model.state_dict(), p['best_model'])
            
            else:
                print('No new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
                print('Last best model was found in epoch %d' %(best_epoch))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_iou': best_iou}, 
                    p['checkpoint'])

    ## eval last model
    eval_val = eval_segmentation_full_classes_online(p, val_dataloader, model)
    print('Final Model at Epoch {} \t mIoU: {:.2f}'.format(p['epochs'], 100*eval_val['mIoU']) )
    
if __name__ == "__main__":
    main()
