import os
from PIL import Image
import argparse
import numpy as np
from pascal_palette import PASCAL_PALETTE_NOVEL
from tqdm import tqdm

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(PASCAL_PALETTE_NOVEL)

    return new_mask

ALL_FOLDS = {
    'fold0': [c for c in range(1,6)],
    'fold1': [c for c in range(6,11)],
    'fold2': [c for c in range(11,16)],
    'fold3': [c for c in range(16,21)],
}

# Parser
parser = argparse.ArgumentParser(description='Fully-supervised segmentation')
parser.add_argument('--novel-dir', type=str, required=True,
                    help='Config file for the experiment')
parser.add_argument('--fold', type=str, default='fold0',
                    help='Config file for the experiment')
parser.add_argument('--nclusters', type=int, default=5,
                    help='Number of novel clusters')

args = parser.parse_args()

ndir = args.novel_dir
base_map = os.path.join(args.novel_dir, 'high_base_pred')
novel_map = os.path.join(args.novel_dir, 'embeddings')
pseudo_dir = os.path.join(args.novel_dir, 'pseudo_labels')
os.makedirs(pseudo_dir, exist_ok=True)

novel_classes = [c for c in range(16,16+args.nclusters)]

with open(os.path.join(ndir, 'novel.txt'), 'r') as nf:
    data_list = nf.readlines()

for line in tqdm(data_list):
    name = line.strip()
    base_pred = np.array(Image.open(os.path.join(base_map, name+'.png')))
    novel_pred = np.array(Image.open(os.path.join(novel_map, name+'.png')))

    ## set the novel label to suitable set part
    novel_label = np.unique(novel_pred)
    novel_label = novel_label[novel_label!=0]
    novel_relabel = novel_pred.copy()

    for i in novel_label:
        novel_relabel[novel_pred==i] = novel_classes[i-1]
    bmask = base_pred > 0

    ## fuse base and novel class
    pseudo = base_pred + novel_relabel * (~bmask)
    # pseudo = base_pred + novel_pred * (~bmask)
    label_kinds = np.unique(pseudo)
    # print(label_kinds)
    if (label_kinds > 15+args.nclusters).any():
        
        if (label_kinds[:-1] > 15).any():
            raise 'Error there should be only one target label'

        raise 'Error the label should not be larger than 20'
    
    pseudo_label = colorize_mask(pseudo)
    pseudo_label.save(os.path.join(pseudo_dir, name+'.png'))

        