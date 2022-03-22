
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import numpy as np
from termcolor import colored

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def cluster_subdomain(entropy_list, save_dir, lambda1):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]

    if lambda1 < 1:
        easy_num = int(len(entropy_rank) * lambda1)
    else:
        easy_num = lambda1

    easy_split = entropy_rank[ : easy_num]
    hard_split = entropy_rank[easy_num: ]

    with open(os.path.join(save_dir,'easy_split.txt'),'w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open(os.path.join(save_dir,'hard_split.txt'),'w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

    print(colored('Write new data split list to {}'.format(save_dir), 'yellow'))
    
    return copy_list


@torch.no_grad()
def reassign_data_split(p, cur_loader, model, save_dir_=None, split_proportion=None):

    model.eval()

    save_dir = save_dir_ if save_dir_ else p['output_dir']

    os.makedirs(save_dir, exist_ok=True)
    entropy_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(cur_loader)):
            images = batch['image'].cuda(non_blocking=True)
            targets = batch['semseg'].cuda(non_blocking=True)
            meta = batch['meta']
            
            output_tuple = model(images)
            if isinstance(output_tuple, tuple):
                output, _ = output_tuple
            else:
                output = output_tuple

            # if args.normalize == True:
            #     normalizor = (11-len(find_rare_class(pred_trg_main))) / 11.0 + 0.5
            # else:
            #     normalizor = 1
            normalizor = 1
            pred_trg_entropy = prob_2_entropy(F.softmax(output, dim=1))  ## b,c,h,w
            b,c,h,w = pred_trg_entropy.size()
            novel_map = (targets > 15).reshape(b,-1).unsqueeze(1) # b,h,w --> b,hw --> b,1,hw
            novel_entropy = (pred_trg_entropy.reshape(b,c,-1) * novel_map).sum(-1) / novel_map.sum(-1)
            for jj in range(output.shape[0]):
                entropy_list.append((meta['image'][jj], novel_entropy[jj].mean().item() * normalizor))
        # colorize_save(pred_trg_main, name[0])

    # split the enntropy_list into 
    split = split_proportion if split_proportion else p['entropy']['split_proportion']
    cluster_subdomain(entropy_list, save_dir, split)

