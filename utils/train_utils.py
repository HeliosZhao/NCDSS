

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeter, ProgressMeter
from utils.utils import SemsegMeter, freeze_layers
import time
from utils.ramps import sigmoid_rampup
from termcolor import colored
import numpy as np


def train_base_classes(p, train_loader, model, criterion, optimizer, epoch, freeze_batchnorm='none'):
    """ Train a segmentation model in a fully-supervised manner """
    losses = AverageMeter('Loss', ':.2f')
    times = AverageMeter('Time', ':6.2f')
    semseg_meter = SemsegMeter(p['num_classes'], train_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    progress = ProgressMeter(len(train_loader),
        [losses, times],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    if freeze_batchnorm == 'none':
        print('BatchNorm tracks running stats - model put to train mode.')
        pass
    
    elif freeze_batchnorm == 'backbone':
        print('Freeze BatchNorm in the backbone - backbone put to eval mode.')
        model.backbone.eval() # Put encoder to eval

    elif freeze_batchnorm == 'first_layers':
        print('Freeze BatchNorm in the backbone - backbone put to eval mode.')
        freeze_layers(model)

    elif freeze_batchnorm == 'all': # Put complete model to eval
        print('Freeze BatchNorm - model put to eval mode.')
        model.eval()

    else:
        raise ValueError('Invalid value freeze batchnorm {}'.format(freeze_batchnorm))

    for i, batch in enumerate(train_loader):

        start_time = time.time()

        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        
        
        losses.update(loss.item())
        semseg_meter.update(torch.argmax(output, dim=1), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        times.update(end_time-start_time)

        if i % 25 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results



def train_basic(p, base_loader, novel_loader, model, criterion, optimizer, epoch, freeze_batchnorm='none'):
    """ Train a segmentation model in a fully-supervised manner """
    losses_base = AverageMeter('Loss_base', ':.2f')
    losses_novel = AverageMeter('Loss_novel', ':.2f')
    times = AverageMeter('Time', ':.2f')
    semseg_meter = SemsegMeter(p['num_classes'], novel_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    loader_len = min(len(base_loader), len(novel_loader))
    progress = ProgressMeter(loader_len,
        [losses_base, losses_novel, times],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    if freeze_batchnorm == 'none':
        print('BatchNorm tracks running stats - model put to train mode.')
        pass
    
    elif freeze_batchnorm == 'backbone':
        print('Freeze BatchNorm in the backbone - backbone put to eval mode.')
        model.backbone.eval() # Put encoder to eval

    elif freeze_batchnorm == 'first_layers':
        print('Freeze BatchNorm in the backbone - backbone put to eval mode.')
        freeze_layers(model)

    elif freeze_batchnorm == 'all': # Put complete model to eval
        print('Freeze BatchNorm - model put to eval mode.')
        model.eval()

    else:
        raise ValueError('Invalid value freeze batchnorm {}'.format(freeze_batchnorm))

    for i, (batch, batch_novel) in enumerate(zip(base_loader, novel_loader)):

        start_time = time.time()

        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output = model(images)
        loss_base = criterion(output, targets)
        
        images_novel = batch_novel['image'].cuda(non_blocking=True)
        targets_novel = batch_novel['semseg'].cuda(non_blocking=True)
        names = batch_novel['meta']['image_file']

        output_novel = model(images_novel)
        loss_novel = criterion(output_novel, targets_novel)
        loss = (loss_base + loss_novel) / 2

        losses_base.update(loss_base.item())
        losses_novel.update(loss_novel.item())
        prediction = torch.argmax(output_novel, dim=1)

        semseg_meter.update(prediction, targets_novel)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        times.update(end_time-start_time)

        # targets, prediction_relabel = targets_novel.data.cpu().numpy(), prediction.data.cpu().numpy()
        # for fname, gt, pred in zip(names, targets, prediction_relabel):
        #     name = fname.split('/')[-1][:-4]
        #     gt = colorize_mask(gt)
        #     gt.save(os.path.join(p['output_dir'], 'train_gt', name+'.png'))
        #     pred = colorize_mask(pred)
        #     pred.save(os.path.join(p['output_dir'], 'train_pre', name+'.png'))

        if i % 25 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results

def train_eums(p, base_loader, easy_loader, hard_loader, model, criterion, optimizer, epoch, freeze_batchnorm='none'):
    """ Train a segmentation model in a fully-supervised manner """
    losses_base = AverageMeter('Loss_base', ':.2f')
    losses_novel = AverageMeter('Loss_novel', ':.2f')
    losses_consis = AverageMeter('Loss_consis', ':.4f')
    times = AverageMeter('Time', ':.2f')
    semseg_meter = SemsegMeter(p['num_classes'], easy_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    loader_len = min(len(base_loader), len(easy_loader), len(hard_loader))
    progress = ProgressMeter(loader_len,
        [losses_base, losses_novel, losses_consis, times],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    if p['freeze_batchnorm'] == 'backbone':
        print('------> Freeze backbone batchnorm <--------')
        model._freeze_backbone_bn()


    w = p['rampup_coefficient'] * sigmoid_rampup(epoch, p['rampup_length'])
    print(colored('Consistency weight is : {:.3f}'.format(w), 'blue'))

    for i, (batch, batch_novel, batch_hard) in enumerate(zip(base_loader, easy_loader, hard_loader)):

        start_time = time.time()


        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
      
        images_novel = batch_novel['image'].cuda(non_blocking=True)
        targets_novel = batch_novel['semseg'].cuda(non_blocking=True)
        names = batch_novel['meta']['image_file']

        images_hard = batch_hard['image'].cuda(non_blocking=True)
        images_hard_strong = batch_hard['strong'].cuda(non_blocking=True)
        output_dict = model(im_base=images, im_easy=images_novel, im_hard=images_hard, im_hard_strong=images_hard_strong)

        loss_base = criterion(output_dict['base'], targets)

        loss_novel = criterion(output_dict['easy'], targets_novel)
        loss = (loss_base + loss_novel) / 2

        max_val, max_ind = F.softmax(output_dict['weak'], dim=1).max(dim=1)
        # print(max_val)
        # print('threshold', p['threshold'])
        max_ind[max_val < p['threshold']] = 255
        consis_loss = criterion(output_dict['strong'], max_ind.long())
        ## can use the ramp but the mean teacher directly use 100
        loss += w * consis_loss

        losses_base.update(loss_base.item())
        losses_novel.update(loss_novel.item())
        losses_consis.update(consis_loss.item())

        prediction = torch.argmax(output_dict['easy'], dim=1)

        semseg_meter.update(prediction, targets_novel)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        times.update(end_time-start_time)

        if i % 25 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results




