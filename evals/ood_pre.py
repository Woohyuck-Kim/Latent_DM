
import torch
import numpy as np
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import pickle
import tensorflow as tf

import datasets
import models_1.transform_layers as TL
import sampling
from utils_1.utils import set_random_seed, normalize
from evals.evals import get_auroc, get_auprc, get_fpr80
import sde_lib
import likelihood
from residual_mlp import Residual_MLP_Small
from models.ema import ExponentialMovingAverage
from utils import save_checkpoint, restore_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    model.eval()
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################
    score_model = Residual_MLP_Small()
    score_model = torch.nn.DataParallel(score_model)
    score_model = score_model.to(torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))

    optimizer = torch.optim.Adam(score_model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=0.9999)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    sde = sde_lib.subVPSDE(beta_min=0.1, beta_max=20, N=1000)
    sampling_eps = 1e-3

    likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x: (x + 1.) / 2.,rtol=1e-3, atol=1e-3)

    state = restore_checkpoint('./checkpoints/checkpoint_ResMLP_new_10.pth', state,
                               device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    ema.copy_to(score_model.parameters())
    score_model.eval()
    model.eval()

    # Build data iterators

    std = torch.load('std_new.pt').cuda()
    mean = torch.load('mean_new.pt').cuda()
    # Create data normalizer and its inverse
    scaler = datasets.ResMLP_scaler(std, mean)
    inverse_scaler = datasets.ResMLP_inverse_scaler(std, mean)
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################

    print('Pre-compute global statistics...')

    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    with open('feats_train.pkl', 'wb') as f:
        pickle.dump(feats_train, f)
    
    with open('feats_train.pkl', 'rb') as f:
        feats_train = pickle.load(f)

    P.axis = []
    for f in feats_train['feature'].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))

    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    f_sim = [f.mean(dim=1) for f in feats_train['feature'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    # Generate random samples.

    sampling_shape = (128, 1, 512, 1)
    sampling_fn = sampling.get_sampling_fn_no_config(sde, sampling_shape, inverse_scaler, sampling_eps, mean_bias=0,
                                           std_bias=0)

    num_sampling_rounds = 5000 // 128 + 1

    for r in range(num_sampling_rounds):
        if r == 0:
            samples, n = sampling_fn(score_model)
            samples = samples.detach().cpu().numpy()
            samples_np = samples

        samples, n = sampling_fn(score_model)
        samples = samples.detach().cpu().numpy()
        samples_np = np.concatenate((samples_np, samples), axis=0)

    samples_tensor = torch.FloatTensor(samples_np).cuda()
    samples_tensor = samples_tensor.squeeze(dim=1).squeeze(dim=-1)

    weight_sim = []
    weight_shi = []

    for shi in range(P.K_shift):

        KNNs = []

        for i in range(int(len(f_sim[0])/128)+1):
            if i==0:
                input = f_sim[shi][:(i+1)*128].cuda()
            elif i==int(len(f_sim)/128):
                input = f_sim[shi][i*128:].cuda()
            else:
                input = f_sim[shi][i*128:(i+1)*128].cuda()

            KNN=KNN_OOD(feature_batch=input, samples_tensor=samples_tensor, KNN_k=5)
            #KNNs.extend( (torch.FloatTensor(KNN).cuda() * torch.norm(input, dim=1).cuda()).detach().cpu().numpy() )
            KNNs.extend((10.0*torch.FloatTensor(KNN).cuda() + torch.norm(input, dim=1).cuda()).detach().cpu().numpy())

       # sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        #weight_sim.append(1 / sim_norm.mean().item())
        weight_sim.append(1 / abs(np.array(KNNs).mean().item()))
        weight_shi.append(1 / shi_mean.mean().item())

    #with open('weight_sim.pkl', 'wb') as f:
        #pickle.dump(weight_sim, f)

   # with open('weight_sim.pkl', 'rb') as f:
       # weight_sim = pickle.load(f)

    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    print('Pre-compute features...')

    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    #with open('feats_id.pkl', 'wb') as f:
     #   pickle.dump(feats_id, f)

    #with open('feats_id.pkl', 'rb') as f:
        #feats_id = pickle.load(f)

    feats_ood = dict()

    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores... (score: {ood_score})')
    scores_id = get_scores(P, feats_id, ood_score, score_model, likelihood_fn, scaler, samples_tensor).numpy()

    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []
    auprc = []
    fpr80 = []
    for ood, feats in feats_ood.items():
        scores_ood[ood] = get_scores(P, feats, ood_score, score_model, likelihood_fn, scaler, samples_tensor).numpy()
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        AUPRC = get_auprc(scores_id, scores_ood[ood])
        auprc.append(AUPRC)
        FPR80 = get_fpr80(scores_id, scores_ood[ood])
        fpr80.append(FPR80)
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

    #with open('scores_ood.pkl', 'wb') as f:
        #pickle.dump(scores_ood, f)

    print("AUPRC: ")
    for i in range(len(auprc)):
        print(str(auprc[i]))
    print("FPR80: ")
    for i in range(len(fpr80)):
        print(str(fpr80[i]))

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict

def KNN_OOD(feature_batch, samples_tensor, KNN_k=5):

   # cos_dist = 1.0 - torch.inner(torch.nn.functional.normalize(samples_tensor),
                                #torch.nn.functional.normalize(feature_batch))
    cos_dist = 1.0 + torch.inner(torch.nn.functional.normalize(samples_tensor),
                                torch.nn.functional.normalize(feature_batch))
   # cos_dist, _ = torch.sort(cos_dist)
    cos_dist, _ = torch.sort(cos_dist, descending=True)
    #knn_dist = torch.mean(cos_dist[:KNN_k, :], dim=0)
    knn_dist = cos_dist[KNN_k, :]

   # return -1.0 * knn_dist.detach().cpu().numpy()
    return knn_dist.detach().cpu().numpy()

def get_scores(P, feats_dict, ood_score, model, likelihood_fn, scaler, samples_tensor):
    # convert to gpu tensor
    feats_sim = feats_dict['feature'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)
    model.eval()

    # compute scores
    scores = []
    batch = 64
    for i in range(int(len(feats_sim)/batch)+1):

        if i == 0:
            f_sim = feats_sim[:(i+1) * batch]
            f_shi = feats_shi[:(i+1) * batch]
        elif i == int(len(feats_sim)/batch):
            f_sim = feats_sim[i * batch:]
            f_shi = feats_shi[i * batch:]
        else:
            f_sim = feats_sim[i * batch:(i + 1) * batch]
            f_shi = feats_shi[i * batch:(i + 1) * batch]

        f_sim = [f.mean(dim=1, keepdim=True) for f in f_sim.chunk(P.K_shift, dim=1)]  # list of (batch, 1, d)
        f_shi = [f.mean(dim=1, keepdim=True) for f in f_shi.chunk(P.K_shift, dim=1)]  # list of (batch, 1, 4)
        for shi in range(P.K_shift):
            KNN = KNN_OOD(feature_batch=f_sim[shi].squeeze(dim=1), samples_tensor=samples_tensor, KNN_k=5)
            #bpd = likelihood_fn(model, scaler(f_sim[shi].squeeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=-1))[0]).unsqueeze(dim=-1)
            #score = ((((f_sim[shi] * P.axis[shi].unsqueeze(dim=0)).sum(dim=2) / torch.norm(f_sim[shi], dim=2)) * bpd).max(dim=1)[0] * torch.FloatTensor([P.weight_sim[shi]]).cuda()).detach().cpu().numpy()
            if shi==0:
                #score = ((torch.FloatTensor(KNN).cuda()*torch.norm(f_sim[shi].squeeze(dim=1), dim=1).cuda()) * torch.FloatTensor([P.weight_sim[shi]]).cuda()).detach().cpu().numpy()
                score = ((10.0*torch.FloatTensor(KNN).cuda()+torch.norm(f_sim[shi].squeeze(dim=1), dim=1).cuda()) * torch.FloatTensor([P.weight_sim[shi]]).cuda()).detach().cpu().numpy()
            else:
                #score += ((torch.FloatTensor(KNN).cuda()*torch.norm(f_sim[shi].squeeze(dim=1), dim=1).cuda()) * torch.FloatTensor([P.weight_sim[shi]]).cuda()).detach().cpu().numpy()
                score += ((10.0*torch.FloatTensor(KNN).cuda()+torch.norm(f_sim[shi].squeeze(dim=1), dim=1).cuda()) * torch.FloatTensor([P.weight_sim[shi]]).cuda()).detach().cpu().numpy()
            score += (f_shi[shi][:,0, shi]* torch.FloatTensor([P.weight_shi[shi]]).cuda()).detach().cpu().numpy()
        score = score / P.K_shift
        scores.extend(score)

    scores = torch.tensor(scores)
    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift', 'feature')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            #path = prefix + f'_{data_name}_{layer}.pth'
            #torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift', 'feature')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                #x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
                x_t = torch.cat([P.shift_trans(x, k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            #x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

