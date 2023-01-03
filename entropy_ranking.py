#

import argparse
import cv2
import os
import sys
import torch
import random
import numpy as np

from utils.config import create_config
from utils.common_config import get_val_transformations, get_val_dataloader, get_model

from data.dataloaders.pascal_voc import VOC12_Basic_Train
from data.dataloaders.coco import COCO_Basic_Train

from termcolor import colored
from utils.logger import Logger
from tqdm import tqdm
import torch.nn.functional as F

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
parser = argparse.ArgumentParser(description='Fully-supervised segmentation - Finetune linear layer')
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
parser.add_argument('--ckpt', 
                    help='Model state dict to test')

parser.add_argument('--split-dir', type=str, default='',
                    help='Split dir for txt, empty for not use easy split')
parser.add_argument('--ent', type=float, default=0.67,
                    help='entropy ranking proportion')


args = parser.parse_args()

def set_seed(random_seed):
    print(colored('setting random seed to {}'.format(random_seed), 'green'))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def cluster_subdomain(entropy_list, save_dir, lambda1):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]

    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

    with open(os.path.join(save_dir,'easy_split.txt'),'w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open(os.path.join(save_dir,'hard_split.txt'),'w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

    return copy_list

def main():
    cv2.setNumThreads(1)
    print(args)
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
    val_transforms = get_val_transformations()

    if p['dataset'] == 'COCO':
        novel_dataset = COCO_Basic_Train(root=p['data_root'], split='novel', transform=val_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'])
        base_cls_num = 60
    else:
        novel_dataset = VOC12_Basic_Train(root=p['data_root'], split='novel', transform=val_transforms, novel_dir=args.novel_dir, novel_fold=p['fold'])
        base_cls_num = 15

    val_dataloader = get_val_dataloader(p, novel_dataset)  

    ## load best model from ckpt

    print(colored('Train from Scratch', 'blue'))
    model = model.cuda()
    model.load_state_dict(torch.load(args.ckpt), strict=False)
    model.eval()

    save_dir = os.path.join(args.novel_dir, args.split_dir)
    os.makedirs(save_dir, exist_ok=True)
    entropy_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            images = batch['image'].cuda(non_blocking=True)
            targets = batch['semseg'].cuda(non_blocking=True)
            meta = batch['meta']
            
            output_tuple = model(images)
            if isinstance(output_tuple, tuple):
                output, _ = output_tuple
            else:
                output = output_tuple

            normalizor = 1
            pred_trg_entropy = prob_2_entropy(F.softmax(output, dim=1))  ## b,c,h,w
            b,c,h,w = pred_trg_entropy.size()
            novel_map = (targets > base_cls_num).reshape(b,-1).unsqueeze(1) # b,h,w --> b,hw --> b,1,hw
            novel_entropy = (pred_trg_entropy.reshape(b,c,-1) * novel_map).sum(-1) / novel_map.sum(-1)
            for jj in range(output.shape[0]):
                entropy_list.append((meta['image'][jj], novel_entropy[jj].mean().item() * normalizor))
        # colorize_save(pred_trg_main, name[0])

    # split the enntropy_list into 
    cluster_subdomain(entropy_list, save_dir, args.ent)

if __name__ == "__main__":
    main()
