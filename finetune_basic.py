#

import argparse
import cv2
import os
import sys
from numpy import triu_indices_from
import torch
import random
import numpy as np

from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate
from utils.train_utils import train_basic
from utils.evaluate_utils import eval_segmentation_full_classes_online
from data.dataloaders.pascal_voc import VOC12_Basic_Train, VOC12_NovelFinetuing_Val
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
parser = argparse.ArgumentParser(description='Finetune Basic Model')
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
                    
parser.add_argument('--seed', type=int, default=242133,
                    help='Set seed for reproducibility')

parser.add_argument('--data-root', type=str, default=None,
                    help='Dataset root')

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
    model = get_model(p)
    # print(model)
    model = model.cuda()
    classifier_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        for ft_layer in p['ft_layer']:
            if name.startswith(ft_layer):

                param.requires_grad = True
                if name.startswith('decoder'):
                    print('Add {} in classifier params'.format(name))
                    classifier_params.append(param)
                else:
                    print('Add {} in decoder params'.format(name))
                    decoder_params.append(param)
                
                break
            else:
                param.requires_grad = False
    

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion.cuda()
    print(criterion)

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    print(colored('classifier params {}'.format(len(classifier_params))))
    print(colored('Other params {}'.format(len(decoder_params))))
    parameters = [
        {'params': classifier_params},
        {'params': decoder_params, 'lr': 0.01 * p['optimizer_kwargs']['lr']}
    ]
    optimizer = get_optimizer(p, parameters)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations()       
    base_dataset = VOC12_Basic_Train(root=p['data_root'], split='base', transform=train_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'])
    base_dataloader = get_train_dataloader(p, base_dataset)

    novel_dataset = VOC12_Basic_Train(root=p['data_root'], split='novel', transform=train_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'])
    novel_dataloader = get_train_dataloader(p, novel_dataset)


    val_transforms = get_val_transformations()
    val_dataset = VOC12_NovelFinetuing_Val(root=p['data_root'], split='val', transform=val_transforms, novel_fold=p['fold'])
    val_dataloader = get_val_dataloader(p, val_dataset)  
    
    
    print(colored('Base Train samples %d - Novel Train samples %d - Val samples %d' %(len(base_dataset), len(novel_dataset), len(val_dataset)), 'yellow'))


    print(colored('Train from Scratch', 'blue'))
    start_epoch = 0
    best_epoch = 0
    best_iou = 0
    model = model.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        eval_train = train_basic(p, base_dataloader, novel_dataloader, model, criterion, optimizer, epoch, freeze_batchnorm=p['freeze_batchnorm'])

        # Evaluate online -> This will use batched eval where every image is resized to the same resolution.
        print('Evaluate ...')
        eval_val = eval_segmentation_full_classes_online(p, val_dataloader, model)
        
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


if __name__ == "__main__":
    main()
