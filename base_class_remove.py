#

import argparse
import cv2
import os
import torch

from utils.config import create_config
from utils.common_config import get_val_transformations_wsal, get_val_dataloader, get_model
from utils.evaluate_utils import base_class_remove_save
from termcolor import colored

from data.dataloaders.pascal_voc import VOC12_NovelSaliency
         
# Parser
parser = argparse.ArgumentParser(description='Base Class Remove')

parser.add_argument('--config',
                    help='Config file for the experiment')

parser.add_argument('--fold', type=str, default='fold0',
                    help='Split fold for novel class')
parser.add_argument('--output-dir', type=str, default='',
                    help='Output dir, if indicate, use it; else, use the time')

parser.add_argument('--ckpt', 
                    help='Model state dict to test')
parser.add_argument('-t', '--threshold', type=float, default=0.9, 
                    help='Model state dict to test')

parser.add_argument('--data-dir', type=str, default='./VOCSegmentation',
                    help='Config file for the experiment')

args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    # print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    val_transforms = get_val_transformations_wsal()
    # val_dataset = get_val_dataset(p, val_transforms)
    val_dataset = VOC12_NovelSaliency(root=args.data_dir, split='trainaug', transform=val_transforms, novel_fold=p['fold']) 
    # true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Val samples %d' %(len(val_dataset)), 'yellow'))

    # Evaluate best model at the end
    print(colored('Evaluating model at {}'.format(args.ckpt), 'blue'))
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    base_class_remove_save(p, val_dataloader, model, threshold=args.threshold)

if __name__ == "__main__":
    main()
