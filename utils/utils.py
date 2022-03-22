#

import os
import torch
import numpy as np
import errno
from PIL import Image

from pascal_palette import PASCAL_PALETTE_NOVEL
import torch.nn as nn

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(PASCAL_PALETTE_NOVEL)

    return new_mask

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class SemsegMeter(object):
    def __init__(self, num_classes, class_names, has_bg=True, ignore_index=255):
        self.num_classes = num_classes + int(has_bg)
        self.class_names = class_names
        if len(class_names) < self.num_classes:
            self.class_names.extend(['novel-'+str(i) for i in range(self.num_classes - len(class_names))])
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        assert(ignore_index == 255)
        self.ignore_index = ignore_index

    def update(self, pred, gt):
        valid = (gt != self.ignore_index)

        for i_part in range(0, self.num_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
            
    def return_score(self, verbose=True, return_jac=False):
        jac = [0] * self.num_classes
        for i_part in range(self.num_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        if verbose:

            print('Evaluation of semantic segmentation ')
            print('mIoU is %.2f' %(100*eval_result['mIoU']))
            for i_part in range(self.num_classes):
                print('IoU class %s is %.2f' %(self.class_names[i_part], 100*jac[i_part]))
                
        return eval_result


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def freeze_layers(model):
    # Freeze block 1+2 layers in the backbone
    model.backbone.conv1.eval()
    model.backbone.bn1.eval()
    model.backbone.layer1.eval()
    model.backbone.layer2.eval()



def _init_weight(layer, conv_init=nn.init.kaiming_normal_, **kwargs):
    for name, m in layer.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
