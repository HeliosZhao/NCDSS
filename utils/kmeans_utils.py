
import os
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from termcolor import colored
from tqdm import tqdm
import imageio


N_JOBS = 16 # set to number of threads

@torch.no_grad()
def save_kmeans_embeddings_novel(p, val_loader, model, n_clusters=5, seed=242133):
    import torch.nn as nn
    print('Save embeddings to disk ...')
    model.eval()
    ptr = 0

    proto_dir = os.path.join(p['output_dir'], 'prototypes')
    os.makedirs(proto_dir, exist_ok=True)

    if p['backbone_imagenet']:
        prototype_dim = 2048
        if p['backbone'] == 'resnet18':
            prototype_dim = 512
    else:
        prototype_dim = 32

    all_prototypes = torch.zeros((len(val_loader.sampler), prototype_dim)).cuda()
    all_sals = torch.zeros((len(val_loader.sampler), 512, 512)).cuda()
    names = []
    img_sizes = []
    for i, batch in enumerate(tqdm(val_loader)):
        sal = batch['sal'].cuda(non_blocking=True)
        output, _ = model(batch['image'].cuda(non_blocking=True))
        meta = batch['meta']

        # compute prototypes
        bs, dim, _, _ = output.shape
        output = output.reshape(bs, dim, -1) # B x dim x H.W
        sal_proto = sal.reshape(bs, -1, 1).type(output.dtype) # B x H.W x 1
        prototypes = torch.bmm(output, sal_proto*(sal_proto>0.5).float()).squeeze(-1) # B x dim
        prototypes = nn.functional.normalize(prototypes, dim=1)        
        all_prototypes[ptr: ptr + bs] = prototypes
        all_sals[ptr: ptr + bs, :, :] = (sal > 0.5).float()
        ptr += bs
        for i in range(len(meta['image'])):
            name = meta['image'][i]
            names.append(name)
            im_size = (int(meta['im_size'][0][i].data), int(meta['im_size'][1][i].data))
            img_sizes.append(im_size)

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    # perform kmeans
    all_prototypes = all_prototypes.cpu().numpy()
    all_sals = all_sals.cpu().numpy()
    # n_clusters = n_clusters - 1
    print('Kmeans clustering to {} clusters'.format(n_clusters))
    
    print(colored('Starting kmeans with scikit', 'green'))
    pca = PCA(n_components = 32, whiten = True)
    all_prototypes = pca.fit_transform(all_prototypes)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_prototypes) # 0-4

    # save predictions
    for i, fname, pred, im_size in zip(range(len(val_loader.sampler)), names, prediction_kmeans, img_sizes):
        prediction = all_sals[i].copy() ## shape --> 512 * 512
        prediction[prediction == 1] = pred + 1
        prediction = prediction.reshape(512,512)
        prediction = cv2.resize(prediction.astype(np.uint8) , dsize=(int(im_size[1]), int(im_size[0])), 
                                        interpolation=cv2.INTER_NEAREST)

        imageio.imwrite(os.path.join(p['embedding_dir'], fname + '.png'), prediction)
        proto = all_prototypes[i].copy()
        np.save(os.path.join(proto_dir, fname+'.npy'), proto)
        if i % 300 == 0:
            print('Saving results: {} of {} objects'.format(i, len(val_loader.dataset)))
