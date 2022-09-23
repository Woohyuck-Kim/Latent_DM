# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """


import os
import time
import models_1.transform_layers as TL
import pickle
import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import sde_lib
import torch
from torch.utils import tensorboard
import torchvision

from utils import save_checkpoint, restore_checkpoint
from residual_mlp import Residual_MLP
from torch.utils.data import DataLoader
import models_1.classifier as C




def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = Residual_MLP(in_dim=config.training.feature_dim, hidden_dim=config.training.hidden_dim)
  score_model = torch.nn.DataParallel(score_model)
  score_model = score_model.to(config.device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  #state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Import contrastive learning encoder
  model = C.get_classifier('resnet18').to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))
  model = C.get_shift_classifer(model, 4).to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))
  checkpoint = torch.load('./resnet18/cifar10_unlabeled.model')
  model.load_state_dict(checkpoint)
  model.eval()

  # Expected feature dimension: (Batch, feature size)

  def to_feature(batch, model=model):
    _, output = model(batch.cuda(), simclr=True, feature=True)
    feature = output['feature']
    return feature

  assert to_feature(torch.randn(config.training.batch_size,config.data.num_channels,
                                config.data.image_size, config.data.image_size)).shape == (config.training.batch_size, config.training.feature_dim)
  # Build data iterators

  train_set = torchvision.datasets.CIFAR10('~/data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]))

  test_set = torchvision.datasets.CIFAR10('~/data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))

  train_loader = DataLoader(train_set, shuffle=True, batch_size=config.training.batch_size)
  test_loader = DataLoader(test_set, shuffle=True, batch_size=config.training.batch_size)

  # Compute sample mean/std for data normalization
  flag = 0
  iter = 3
  for i in range(iter):
    print("Iteration "+str(i))

    for x, y in enumerate(train_loader):
      if flag==0 and i==0:
        batch = y[0]
        #batch = transform(batch)
        feature = to_feature(batch).detach().cpu().numpy()
        features = feature
        flag=1
      else:
        batch = y[0]
       # batch = transform(batch)
        feature = to_feature(batch).detach().cpu().numpy()
        features = np.concatenate((features,feature), axis=0)

  train_size = features.shape[0]/iter
  std_new, mean_new = torch.std_mean(torch.FloatTensor(features), dim=0)
  std_new = std_new.cuda()
  mean_new = mean_new.cuda()

  # Create data normalizer and its inverse
  scaler = datasets.ResMLP_scaler(std_new, mean_new)
  inverse_scaler = datasets.ResMLP_inverse_scaler(std_new, mean_new)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))


  step = 0
  loss_train = []
  loss_eval = []
  model.eval()

  for i in range(int(num_train_steps/(train_size/config.training.batch_size))):
    for x, y in enumerate(train_loader):
      batch = y[0]
      feature = to_feature(batch)
      feature = scaler(feature)
      # Reshape to (Batch, 1, feature size, 1)
      feature = feature.unsqueeze(dim=1).unsqueeze(dim=-1)
      # Execute one training step
      loss = train_step_fn(state, feature)
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
        loss_train.append(loss)
        writer.add_scalar("training_loss", loss, step)

        # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        for x, y in enumerate(test_loader):
          batch = y[0]
          feature = to_feature(batch)
          feature = scaler(feature)
          feature = feature.unsqueeze(dim=1).unsqueeze(dim=-1)
            # Execute one training step
          eval_loss = eval_step_fn(state, feature)
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
          loss_eval.append(loss)
          writer.add_scalar("eval_loss", eval_loss.item(), step)
          break

        # Save a checkpoint periodically and generate samples if needed
      if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
          # Save the checkpoint.
        save_step = step // config.training.snapshot_freq
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_ResMLP_new_{save_step}.pth'), state)

      step = step+1



def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder

  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data iterators

  train_set = torchvision.datasets.CIFAR10('~/data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]))

  test_set = torchvision.datasets.CIFAR10('~/data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))

  ood_loaders = []
  ood_names = ['SVHN', 'CIFAR100']

  ood_set = torchvision.datasets.SVHN('~/data', split='test', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
  ood_loader = DataLoader(ood_set, shuffle=True, batch_size=config.training.batch_size)
  ood_loaders.append(ood_loader)

  ood_set = torchvision.datasets.CIFAR100('~/data', train=False, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
  ood_loader = DataLoader(ood_set, shuffle=True, batch_size=config.training.batch_size)
  ood_loaders.append(ood_loader)

  train_loader = DataLoader(train_set, shuffle=True, batch_size=config.training.batch_size)
  test_loader = DataLoader(test_set, shuffle=True, batch_size=config.training.batch_size)

  # Initialize model
  score_model = Residual_MLP(in_dim=config.training.feature_dim, hidden_dim=config.training.hidden_dim)
  score_model = torch.nn.DataParallel(score_model)
  score_model = score_model.to(config.device)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Import contrastive learning encoder
  model = C.get_classifier('resnet18').to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))
  model = C.get_shift_classifer(model, 4).to(torch.device(f"cuda" if torch.cuda.is_available() else "cpu"))
  checkpoint = torch.load('./resnet18/cifar10_unlabeled.model')
  model.load_state_dict(checkpoint)

  # Expected feature dimension: (Batch, feature size)

  def to_feature(batch, model=model):
    _, output = model(batch.cuda(), simclr=True, feature=True)
    feature = output['feature']
    return feature

  # Compute sample mean/std for data normalization
  flag = 0
  iter = 3
  for i in range(iter):
    print("Iteration "+str(i))

    for x, y in enumerate(train_loader):
      if flag==0 and i==0:
        batch = y[0]
        #batch = transform(batch)
        feature = to_feature(batch).detach().cpu().numpy()
        features = feature
        flag=1
      else:
        batch = y[0]
       # batch = transform(batch)
        feature = to_feature(batch).detach().cpu().numpy()
        features = np.concatenate((features,feature), axis=0)

  std_new, mean_new = torch.std_mean(torch.FloatTensor(features), dim=0)
  std = std_new.cuda()
  mean = mean_new.cuda()

  # Create data normalizer and its inverse
  scaler = datasets.ResMLP_scaler(std, mean)
  inverse_scaler = datasets.ResMLP_inverse_scaler(std, mean)

  # Align augmentation
  color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
  color_gray = TL.RandomColorGrayLayer(p=0.2)
  resize_crop = TL.RandomResizedCropLayer(scale=(0.54 ,1.0), size=(32, 32, 3))

  transform = torch.nn.Sequential(
      color_jitter,
      color_gray,
      resize_crop)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x: x)
  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:

    sampling_shape = (config.eval.batch_size,
                      1,
                      config.training.feature_dim, 1)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps ,scale=0 , train_feature=torch.randn(1,1,128,1),mean_bias=0, std_bias=0)


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_ResMLP_new_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
        time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_ResMLP_new_{ckpt}.pth')

    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)

    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)

      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)


    # Compute log-likelihoods (bits/dim) if enabled

    if config.eval.enable_bpd:

      bpds = []
      batch_id = 0

      for x, y in enumerate(test_loader):
        batch = y[0]
        batch = transform(batch)
        feature = to_feature(batch)
        feature = scaler(feature).unsqueeze(dim=1).unsqueeze(dim=-1)
        bpd = likelihood_fn(score_model, feature)[0]
        bpd = bpd.detach().cpu().numpy().reshape(-1)
        bpds.extend(bpd)
        logging.info(
          "CIFAR10(Test): ckpt: %d, batch: %d, mean bpd: %6f" % (ckpt, batch_id, np.mean(np.asarray(bpds))))
        batch_id = batch_id + 1

      with open("bpds_ResMLP_CIFAR10_test_fine_" + str(ckpt) + ".pkl", "wb") as g:
        pickle.dump(bpds, g)

      bpds = []
      batch_id = 0

      for x, y in enumerate(ood_loader[0]):
        batch = y[0]
        batch = transform(batch)
        feature = to_feature(batch)
        feature = scaler(feature).unsqueeze(dim=1).unsqueeze(dim=-1)
        bpd = likelihood_fn(score_model, feature)[0]
        bpd = bpd.detach().cpu().numpy().reshape(-1)
        bpds.extend(bpd)
        logging.info(
          "SVHN: ckpt: %d, batch: %d, mean bpd: %6f" % (ckpt, batch_id, np.mean(np.asarray(bpds))))
        batch_id = batch_id + 1

      with open("bpds_ResMLP_SVHN_fine_" + str(ckpt) + ".pkl", "wb") as g:
        pickle.dump(bpds, g)


    if config.eval.enable_sampling:
      ema.copy_to(score_model.parameters())
      score_model.eval()
      model.eval()
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
        if r==0:
          samples, n = sampling_fn(score_model)
          samples=samples.detach().cpu().numpy()
          samples_np = samples

        # Directory to save samples. Different for each host to avoid writing conflicts
        samples, n = sampling_fn(score_model)
        samples=samples.detach().cpu().numpy()
        samples_np = np.concatenate((samples_np, samples), axis=0)

      samples_tensor = torch.FloatTensor(samples_np)
      samples_tensor = samples_tensor.squeeze(dim=1).squeeze(dim=-1)
      torch.save(samples_tensor, 'samples_new_'+str(ckpt)+'.pt')
