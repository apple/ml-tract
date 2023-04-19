# FID

## Real activation statistics

In order to compute FID, we first need to compute the activation statistics over all real images.

Run

```bash
python compute_fid_stats.py --dataset cifar10
python compute_fid_stats.py --dataset imagenet64
```

to compute and save the mean and std of real activation on CIFAR10 and 64x64 ImageNet.

## Compute FID with a trained model

```bash
python fid_model.py --dataset cifar10 --ckpt={the path to your model}
```

## Compute FID from generated samples

```bash
python fid_zip.py {the path to your zip file} --dataset cifar10
```