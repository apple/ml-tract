# Downloading datasets

## Cifar10
Download Cifar10 data through `torchvision`.
```bash
python -c "import torchvision; import os; torchvision.datasets.CIFAR10(os.getenv('ML_DATA'), train=True, download=True)"
```

## Class-conditional 64x64 ImageNet

 we use the official ILSVRC2012 dataset with manual center cropping and downsampling. To obtain this dataset, navigate to the [2012 challenge page](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and download the data in "Training images (Task 1 & 2)". This is a 138GB tar file containing 1000 sub-tar files, one per class.

## Preparation for Torch dataloaders
We need to prepare `$ML_DATA/imagenet/train/`. These instructions are adapted from a [pytorch example script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh)

```bash
# Create train directory; move .tar file; change directory
mkdir -p $ML_DATA/imagenet/train && mv ILSVRC2012_img_train.tar $ML_DATA/imagenet/train/ && cd $ML_DATA/imagenet/train
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
```

Our data downloading and pre-processing pipeline is largely adapted from https://github.com/openai/guided-diffusion. Thanks for open-sourcing!
