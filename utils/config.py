

import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing
import datetime

def load_config(config_file_exp):
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    for k, v in config.items():
        cfg[k] = v

    return cfg


def create_config(args):
   
    config_file_exp = args.config
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    try:
        cfg['nclusters'] = args.nclusters
    except:
        cfg['nclusters'] = 5
    # print(cfg['nclusters'])

    # Num classes
    if cfg['train_db_name'] == 'VOCSegmentation':
        cfg['num_classes'] = 20
        cfg['has_bg'] = True
    
    elif cfg['train_db_name'] == 'BASE':
        cfg['num_classes'] = 15
        cfg['has_bg'] = True

    elif cfg['train_db_name'] == 'NOVEL_CLUSTERS':
        cfg['num_classes'] = 15 + cfg['nclusters']
        cfg['has_bg'] = True

    elif cfg['train_db_name'] == 'COCO_BASE':
        cfg['num_classes'] = 60
        cfg['has_bg'] = True

    elif cfg['train_db_name'] == 'COCO_NOVEL':
        cfg['num_classes'] = 60 + cfg['nclusters']
        cfg['has_bg'] = True


    else:
        raise ValueError('Invalid train db name {}'.format(cfg['train_db_name']))

    if 'COCO' in cfg['train_db_name']:
        cfg['dataset'] = 'COCO'
    else:
        cfg['dataset'] = 'PASCAL'
        
    if 'threshold' in vars(args):
        if args.threshold is not None:
            cfg['threshold'] = args.threshold

    if 'data_root' in vars(args):
        if args.data_root is not None:
            cfg['data_root'] = args.data_root

    if 'rampup_length' in vars(args):
        cfg['rampup_length'] = args.rampup_length

    if 'rampup_coefficient' in vars(args):
        cfg['rampup_coefficient'] = args.rampup_coefficient


    cfg['fold'] = args.fold ## novel class fold
  
    # Paths 
    root_dir = config['root_dir']
    current_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(root_dir, os.path.basename(config_file_exp).split('.')[0] + current_time)
    
    print('Results save in output dir: ', output_dir)
    mkdir_if_missing(output_dir)
    
    cfg['output_dir'] = output_dir
    cfg['checkpoint'] = os.path.join(cfg['output_dir'], 'checkpoint.pth.tar')
    cfg['best_model'] = os.path.join(cfg['output_dir'], 'best_model.pth.tar')
    cfg['save_dir'] = os.path.join(cfg['output_dir'], 'predictions')
    
    mkdir_if_missing(cfg['save_dir'])
    cfg['log_file'] = os.path.join(cfg['output_dir'], 'logger.txt')

    # Special directories for K-Means -> Which happens off-line
    cfg['embedding_dir'] = os.path.join(cfg['output_dir'], 'embeddings')
    cfg['sal_dir'] = os.path.join(cfg['output_dir'], 'saliency')
    mkdir_if_missing(cfg['embedding_dir'])
    mkdir_if_missing(cfg['sal_dir'])

    # Special directories for retrieval
    cfg['retrieval_dir'] = os.path.join(cfg['output_dir'], 'retrieval')
    mkdir_if_missing(cfg['retrieval_dir'])
    
    if 'backbone_imagenet' not in cfg:
        cfg['backbone_imagenet'] = False 

    if 'kmeans_eval' not in cfg.keys():
        cfg['kmeans_eval'] = False

    return cfg 
