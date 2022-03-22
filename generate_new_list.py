import os
import numpy as np
from PIL import Image
import argparse
from glob import glob

parser = argparse.ArgumentParser(description='Generate new list')
parser.add_argument('--novel-dir', required=True,
                    help='Directory of predicted map')
parser.add_argument('--data-dir', type=str, default='./VOCSegmentation',
                    help='Config file for the experiment')
parser.add_argument('--fold', type=str, default='fold0',
                    help='Split fold for novel class')

parser.add_argument('-t', '--threshold', type=int, default=1500, 
                    help='Threshold for filter out images')
args = parser.parse_args()

novel_map_list = sorted(glob(os.path.join(args.novel_dir, 'novel_map', '*.png')))

remaining_list = []

for file in novel_map_list:
    im = np.array(Image.open(file))
    saliency_pixels = len(im.nonzero()[0])
    if saliency_pixels > args.threshold:
        remaining_list.append(os.path.basename(file)[:-4])

novel_file_name = os.path.join(args.novel_dir, 'novel.txt')
with open(novel_file_name, 'w') as novelf:
    for name in remaining_list:
        novelf.write(name + '\n')


split_dir = 'data/data_split'
split_file = os.path.join(split_dir, args.fold, 'base.txt')
with open(split_file, "r") as f:
    all_base_lines = f.read().splitlines()

split_dir = 'data/data_split'
split_file = os.path.join(split_dir, args.fold, 'novel.txt')
with open(split_file, "r") as f:
    all_novel_lines = f.read().splitlines()



non_overlap_lines = list(set(all_base_lines) - set(all_novel_lines))
base_file_name = os.path.join(args.novel_dir, 'base.txt')
with open(base_file_name, 'w') as basef:
    for name in non_overlap_lines:
        basef.write(name + '\n')

