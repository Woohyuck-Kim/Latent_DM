# Latent_DM

To train diffusion model on latent space, use the command below:
```train
python main_Latent.py --config=./configs/subvp/cifar10_ddpm_continuous.py -eval_folder=eval -mode=train --workdir=./
```

The current implementation uses pre-trained(CSI) resent18 encoder: to change it,
just modify model importing part and to_feature(batch, model) in run_lib_Latent.py(From line 83 to line 95). 

Other parameters like feature size(currently 32) and ResMLP hidden size(currently 128) can be modified from
/configs/default_cifar10_configs.py.
