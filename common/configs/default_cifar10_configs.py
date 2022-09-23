import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 1000002
  training.snapshot_freq = 100000
  training.log_freq = 250
  training.eval_freq = 500
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 100000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = False
  training.background = False
  training.feature_dim = 32
  training.hidden_dim = 128
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 10
  evaluate.end_ckpt = 10
  evaluate.batch_size = 128
  evaluate.enable_sampling = False
  evaluate.num_samples = 1000
  evaluate.enable_loss = False
  evaluate.enable_bpd = True
  evaluate.enable_KNN = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.OOD_dataset = 'SVHN'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.emb_dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.weight_decay_background = 0.0001
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config