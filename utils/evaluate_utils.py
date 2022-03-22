#

import os
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.utils import SemsegMeter
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm




N_JOBS = 16


def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac

def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc'):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
    results = Parallel(n_jobs=16, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def _majority_vote(flat_preds, flat_targets, preds_k, targets_k, metric='acc'):
    iou_mat = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    results = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(results, axis=1)
    match = np.array(list(zip(range(preds_k), results)))
    return match


"""
    Semantic segmentation evaluation
"""
@torch.no_grad()
def eval_segmentation_supervised_online(p, val_loader, model, verbose=True):
    """ Evaluate a segmentation network 
        The evaluation is performed online, without storing the results.
        
        Important: All images are assumed to be rescaled to the same resolution.
        As a consequence, the results might not exactly match with the true evaluation script
        if every image had a different size. 

        Alternative: Use store_results_to_disk and then evaluate with eval_segmentation_supervised_offline.
    """
    semseg_meter = SemsegMeter(p['num_classes'], val_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    model.eval()

    for i, batch in enumerate(tqdm(val_loader)):
        # pbar.update(1)
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output = model(images)
        semseg_meter.update(torch.argmax(output, dim=1), targets)

    eval_results = semseg_meter.return_score(verbose = True)
    return eval_results

@torch.no_grad()
def eval_segmentation_supervised_offline(p, val_dataset, verbose=True):
    """ Evaluate stored predictions from a segmentation network.
        The semantic masks from the validation dataset are not supposed to change. 
    """
    n_classes = 21

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
   
    for i, sample in enumerate(val_dataset):
        if i % 250 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(val_dataset)))
        
        # Load result
        filename = os.path.join(p['save_dir'], sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.uint8)

        gt = sample['semseg']
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction ..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
        
    if verbose:
        print('Evaluation of semantic segmentation ')
        print('mIoU is %.2f' %(100*eval_result['mIoU']))
        class_names = val_dataset.get_class_names()
        for i_part in range(n_classes):
            # print('IoU class %s is %.2f' %(class_names[i_part], 100*jac[i_part]))
            print('%.2f' %(100*jac[i_part]))

    return eval_result

@torch.no_grad()
def eval_segmentation_full_classes_online(p, val_loader, model, verbose=True):
    """ Evaluate a segmentation network 
        The evaluation is performed online, without storing the results.
        
        Important: All images are assumed to be rescaled to the same resolution.
        As a consequence, the results might not exactly match with the true evaluation script
        if every image had a different size. 

        Alternative: Use store_results_to_disk and then evaluate with eval_segmentation_supervised_offline.
    """

    semseg_meter = SemsegMeter(20, val_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    model.eval()
    all_pixels = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda()  #.to(model.device)
    all_gt = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda() ###.to(model.device)
    all_gt_novel = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda() ###.to(model.device)
    offset_ = 0
    gt_offset_sum = 0
    pred_offset_sum = 0

    print('Start gathering novel class information ......')
    for i, batch in tqdm(enumerate(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output_tuple = model(images)
        if isinstance(output_tuple, tuple):
            output, _ = output_tuple
        else:
            output = output_tuple

        novel_map = (targets > 15).long()
        novel_map[targets==255] = 0
        gt_valid = novel_map.sum()
        all_gt_novel[gt_offset_sum:gt_offset_sum+gt_valid,] = targets[novel_map==1]
        gt_offset_sum += gt_valid

        prediction = torch.argmax(output, dim=1)
        
        prediction_map = (prediction > 15).long()
        pred_offset_sum += prediction_map.sum()
        true_positive_novel = ((prediction * novel_map) > 15).long() ## both gt and pred are of the novel class
        n_valid = true_positive_novel.sum()
        all_gt[offset_:offset_+n_valid,] = targets[true_positive_novel==1]
        all_pixels[offset_:offset_+n_valid,] = prediction[true_positive_novel==1]

        offset_ += n_valid
    
    all_pixels = all_pixels[:offset_,]
    all_gt = all_gt[:offset_,]

    all_pixels -= 16 ## convert to 0-4
    all_gt -= 16

    all_pixels = all_pixels.data.cpu().numpy()
    all_gt = all_gt.data.cpu().numpy()
    
    print('Start hungarian match ......')
    if p['nclusters'] == 5:
        match = _hungarian_match(all_pixels, all_gt, preds_k=5, targets_k=5, metric='iou')
    elif p['nclusters'] > 5:
        match = _majority_vote(all_pixels, all_gt, preds_k=p['nclusters'], targets_k=5, metric='iou')
    else:
        raise NotImplementedError
    print('Hungarian match : ', match)
    del all_pixels, all_gt

    print('Evaluation with new label ......')
    for i, batch in tqdm(enumerate(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output_tuple = model(images)
        if isinstance(output_tuple, tuple):
            output, _ = output_tuple
        else:
            output = output_tuple
        prediction = torch.argmax(output, dim=1) ## B,H,W

        prediction_relabel = prediction.clone()
        for pred_i, target_i in match:
            prediction_relabel[prediction==(pred_i+16)] = target_i+16
        
        semseg_meter.update(prediction_relabel, targets)

    eval_results = semseg_meter.return_score(verbose = True, return_jac=True)
    novel_and_bg = eval_results['jaccards_all_categs'][-5:] 
    eval_results['mIoU'] = np.mean(novel_and_bg)
    return eval_results


@torch.no_grad()
def base_class_remove_save(p, val_loader, model, threshold=0.9):
    print('Save results to disk ...')
    model.eval()


    dir_names = ['full_pred', 'high_base_pred', 'high_base_map', 'novel_map']
    for k in dir_names:
        os.makedirs(os.path.join(p['output_dir'], k), exist_ok=True)

    counter = 0
    for i, batch in enumerate(tqdm(val_loader)):
        output = model(batch['image'].cuda(non_blocking=True))
        output = F.softmax(output, dim=1)
        meta = batch['meta']
        sals = batch['sal'].cuda(non_blocking=True)

        for jj in range(output.shape[0]):
            counter += 1
            image_file = meta['image_file'][jj]

            max_val, max_ind = output[jj].max(dim=0)
            full_pred = max_ind  ## direct pred
            threshold_map = torch.zeros_like(max_val, dtype=bool)
            threshold_map[max_val > threshold] = 1
            high_act_base_pred = (max_ind * threshold_map).long() ## high activation pixels with class
            high_act_base_map = (high_act_base_pred > 0).long() * 255 ## high activation base class pixels map, {0,1}
            novel_map = (~high_act_base_map) * sals[jj] * 255  ### novel mask

            save_dict = {
                'full_pred': full_pred,
                'high_base_pred': high_act_base_pred,
                'high_base_map' : high_act_base_map,
                'novel_map': novel_map
            }

            for k,v in save_dict.items():
                save_file = v.cpu().numpy().astype(np.uint8) 
                result = cv2.resize(save_file, dsize=(int(meta['im_size'][1][jj]), int(meta['im_size'][0][jj])), interpolation=cv2.INTER_NEAREST)
                imageio.imwrite(os.path.join(p['output_dir'], k, meta['image'][jj] + '.png'), result)
   
        if counter % 250 == 0:
            print('Saving results: {} of {} objects'.format(counter, len(val_loader.dataset)))


@torch.no_grad()
def eval_segmentation_full_classes_offline(p, val_loader, model, verbose=True):
    """ Evaluate a segmentation network 
        The evaluation is performed online, without storing the results.
        
        Important: All images are assumed to be rescaled to the same resolution.
        As a consequence, the results might not exactly match with the true evaluation script
        if every image had a different size. 

        Alternative: Use store_results_to_disk and then evaluate with eval_segmentation_supervised_offline.
    """
    semseg_meter = SemsegMeter(20, val_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    model.eval()
    all_pixels = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda()  #.to(model.device)
    all_gt = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda() ###.to(model.device)
    all_gt_novel = torch.zeros((len(val_loader.sampler) * 512 * 512)).cuda() ###.to(model.device)
    offset_ = 0
    gt_offset_sum = 0
    pred_offset_sum = 0

    print('Start gathering novel class information ......')
    for i, batch in enumerate(tqdm(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        output_tuple = model(images)
        if isinstance(output_tuple, tuple):
            output, _ = output_tuple
        else:
            output = output_tuple
        novel_map = (targets > 15).long()
        novel_map[targets==255] = 0
        gt_valid = novel_map.sum()
        all_gt_novel[gt_offset_sum:gt_offset_sum+gt_valid,] = targets[novel_map==1]
        gt_offset_sum += gt_valid

        prediction = torch.argmax(output, dim=1)
        
        prediction_map = (prediction > 15).long()
        pred_offset_sum += prediction_map.sum()
        true_positive_novel = ((prediction * novel_map) > 15).long() ## both gt and pred are of the novel class

        n_valid = true_positive_novel.sum()
        all_gt[offset_:offset_+n_valid,] = targets[true_positive_novel==1]
        all_pixels[offset_:offset_+n_valid,] = prediction[true_positive_novel==1]

        offset_ += n_valid
    
    all_pixels = all_pixels[:offset_,]
    all_gt = all_gt[:offset_,]

    all_pixels -= 16 ## convert to 0-4
    all_gt -= 16

    all_pixels = all_pixels.data.cpu().numpy()
    all_gt = all_gt.data.cpu().numpy()
    
    print('Start hungarian match ......')
    if p['nclusters'] == 5:
        match = _hungarian_match(all_pixels, all_gt, preds_k=5, targets_k=5, metric='iou')
    elif p['nclusters'] > 5:
        match = _majority_vote(all_pixels, all_gt, preds_k=p['nclusters'], targets_k=5, metric='iou')
    else:
        raise NotImplementedError
    print('Hungarian match : ', match)

    del all_pixels, all_gt

    class2index = val_loader.dataset.classes_to_index

    print('Evaluation with new label ......')
    counter = 0
    for i, batch in tqdm(enumerate(val_loader)):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)
        # meta = batch['meta']
        output_tuple = model(images)
        if isinstance(output_tuple, tuple):
            output, _ = output_tuple
        else:
            output = output_tuple
        prediction = torch.argmax(output, dim=1) ## B,H,W
        prediction_relabel = prediction.clone()
        for pred_i, target_i in match:
            prediction_relabel[prediction==(pred_i+16)] = target_i+16
        
        prediction_2class = prediction_relabel.clone()
        ## change the index back to original class order
        for k,v in class2index.items():
            prediction_2class[prediction_relabel==v] = k

        meta = batch['meta']
        for jj in range(prediction_2class.shape[0]):
            counter += 1
            image_file = meta['image_file'][jj]

            pred = prediction_2class[jj].cpu().numpy().astype(np.uint8)

            result = cv2.resize(pred, dsize=(int(meta['im_size'][1][jj]), int(meta['im_size'][0][jj])), 
                                        interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(p['save_dir'], meta['image'][jj] + '.png'), result)
   
        if counter % 250 == 0:
            print('Saving results: {} of {} objects'.format(counter, len(val_loader.dataset)))

