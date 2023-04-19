# TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation
This software project accompanies the research paper [TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation](https://arxiv.org/abs/2303.04248)

# Citation:
```
@article{berthelot2023tract,
  title={TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation},
  author={Berthelot, David and Autef, Arnaud and Lin, Jierui and Yap, Dian Ang and Zhai, Shuangfei and Hu, Siyuan and Zheng, Daniel and Talbott, Walter and Gu, Eric},
  journal={arXiv preprint arXiv:2303.04248},
  year={2023}
}
```

# Setup

Git clone with `--recurse-submodules` to initialize EDM submodule properly.

Setup environment variables:

```bash
export ML_DATA=~/Data/DDPM-Images
export PYTHONPATH=$PYTHONPATH:.
```

Then run
```bash
sudo apt install python3.8-dev python3.8-venv python3-dev -y
```

Set up a virtualenv

```bash
python3.8 -m venv ~/tract_venv
source ~/tract_venv/bin/activate
```

or via `pyenv`

```bash
pyenv install 3.8.0
pyenv virtualenv 3.8.0 tract_venv
pyenv local tract_venv
```

then upgrade pip
```
pip install --upgrade pip
```

Install pip pkgs
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

# Setup data

Please follow [README](data/README.md) to setup datasets.

# Set up teacher models

Please follow [README](teacher/README.md) to setup teacher checkpoints.

# Set up EDM
For running with NVIDIA's Elucidated model (EDM), ensure the `edm/`
submodule has been initialized properly.

# Real activation stats for FID

Please follow the `Real activation statistics` section in [README](fid/README.md) in order to compute and save the real activation statistics to be used in FID evaluation.

# Training

The below commands will reproduce results from our paper when run on a cluster of 8 NVIDIA A100 or V100 GPUs.

Example: Run TC distillation on Cifar10 using distillation time schedule: 1024, 32, 1.

```bash
python tc_distill.py --dataset=cifar10 --time_schedule=1024,32,1 --fid_len=50000 --report_fid_len=8M --report_img_len=8M --train_len=96M
```

Example: Run TC distillation on Cifar10 using EDM teacher
```bash
python tc_distill_edm.py --dataset=cifar10 --time_schedule=40,1 --fid_len=50000 --report_fid_len=8M --report_img_len=8M --train_len=96M --batch=512
```

Getting help

```bash
python tc_distill.py --help
```

Tensorboard outputs are generated in a dir like `e/DATASET/MODEL/EXP_NAME/tb/`. For example, you can start tensorboard to view metrics like
```bash
tensorboard --logdir e/cifar10/EluDDIM05TCMultiStepx0\(EluUNet\)/aug_prob@0.0_batch@8_dropout@0.0_ema@0.9996_lr@0.001_lr_warmup@None_res@32_sema@0.1_time_schedule@40,1_timesteps@40/tb/ --bind_all
```

# Evaluation

Please follow [README](fid/README.md) to run FID evaluation.
