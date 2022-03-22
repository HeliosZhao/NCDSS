#

import argparse
import cv2
import os
import torch

from utils.config import create_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader, get_model
from utils.evaluate_utils import eval_segmentation_supervised_offline, eval_segmentation_full_classes_offline
from termcolor import colored
from models.teacher_student import TeacherStudentModel
from data.dataloaders.pascal_voc import VOC12_NovelFinetuing_Val, VOC12


# Parser
parser = argparse.ArgumentParser(description='Evaluate segmantion model')
parser.add_argument('--config',
                    help='Config file for the experiment')
parser.add_argument('--ckpt', 
                    help='Model state dict to test')

parser.add_argument('--fold', type=str, default='fold0',
                    help='Split fold for novel class')
parser.add_argument('--output-dir', type=str, default='',
                    help='Output dir, if indicate, use it; else, use the time')
parser.add_argument('--novel-dir', type=str, default='',
                    help='novel dir storing the pseudo labels')
parser.add_argument('--nclusters', type=int, default=5,
                    help='Number of novel clusters')

parser.add_argument('--data-root', type=str, default=None,
                    help='Dataset root')


args = parser.parse_args()

def main():
    cv2.setNumThreads(1)

    # Retrieve config file
    p = create_config(args)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    checkpoint = torch.load(args.ckpt, map_location='cpu')

    model = TeacherStudentModel(p)
    model.load_state_dict(checkpoint, strict=False)
    model = model.cuda()
    eval_model = model.model_q

    

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    val_transforms = get_val_transformations()
    # val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    # val_dataloader = get_val_dataloader(p, val_dataset)

    # val_transforms = get_val_transformations()
    val_dataset = VOC12_NovelFinetuing_Val(root=p['data_root'], split='val', transform=val_transforms, novel_fold=p['fold'])
    val_dataloader = get_val_dataloader(p, val_dataset)  

    print(colored('Val samples %d' %(len(val_dataset)), 'yellow'))

    # Evaluate best model at the end
    print(colored('Evaluating model at {}'.format(args.ckpt), 'blue'))
    eval_segmentation_full_classes_offline(p, val_dataloader, eval_model)
    eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)

if __name__ == "__main__":
    main()
