#

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from utils.config import create_config
from utils.common_config import get_val_dataloader, get_model
from utils.kmeans_utils import save_kmeans_embeddings_novel
import data.dataloaders.custom_transforms as custom_tr
from termcolor import colored
import torchvision.transforms as transforms
from termcolor import colored

from data.dataloaders.pascal_voc import VOC12_NovelClustering
from data.dataloaders.coco import COCO_NovelClustering


# Parser
parser = argparse.ArgumentParser(description='Kmeans for clustering pseudo-labels')
parser.add_argument('--config', type=str, required=True,
                    help='Config file for the experiment')
parser.add_argument('--novel-dir', type=str, required=True,
                    help='Config file for the experiment')
parser.add_argument('--output-dir', type=str,
                    help='Config file for the experiment')
parser.add_argument('--fold', type=str, default='fold0',
                    help='Split fold for novel class')
parser.add_argument('--nclusters', default=5, type=int,
                    help='number of seeds during kmeans')
parser.add_argument('--seed', default=242133, type=int,
                    help='number of seeds during kmeans')
parser.add_argument('--data-dir', type=str, default='./VOCSegmentation',
                    help='Config file for the experiment')


args = parser.parse_args()
args.output_dir = args.novel_dir

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print(model)
    model = model.cuda()
    

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    val_transforms = transforms.Compose([custom_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                        'sal': tuple((512,512))},
                                            flagvals={'image': cv2.INTER_CUBIC, 'sal': cv2.INTER_NEAREST}),
                                custom_tr.ToTensor(),
                                custom_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    print(val_transforms)
    if p['dataset'] == 'COCO':
        val_dataset = COCO_NovelClustering(root=args.data_dir, split='val', transform=val_transforms, novel_dir=args.novel_dir)
    else:
        val_dataset = VOC12_NovelClustering(root=args.data_dir, split='val', transform=val_transforms, novel_dir=args.novel_dir)
        
    val_dataloader = get_val_dataloader(p, val_dataset)

    # Kmeans Clustering
    save_kmeans_embeddings_novel(p, val_dataloader, model, n_clusters=args.nclusters, seed=args.seed)


    

if __name__ == "__main__":
    main()
