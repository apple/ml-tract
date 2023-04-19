#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import os
import pathlib
from flax import serialization
from tensorflow.compat.v2.io import gfile
from lib.zoo.unet import UNet
import numpy as np
import torch
import einops as ei
from absl import app, flags


def to_torch(x):
    return torch.nn.Parameter(torch.from_numpy(x.copy()))


def check_and_convert_gcs_filepath(filepath, raise_if_not_gcs=False):
    """Utility for loading model checkpoints from GCS."""
    local_filepath = filepath.split('/')[-1]
    if os.path.exists(local_filepath):
        print('loading from local copy of GCS file: ' + local_filepath)
    else:
        print('downloading file from GCS: ' + filepath)
        os.system('gsutil cp ' + filepath + ' ' + local_filepath)
    return local_filepath


def restore_from_path(ckpt_path, target):
    ckpt_path = check_and_convert_gcs_filepath(ckpt_path)
    with gfile.GFile(ckpt_path, 'rb') as fp:
        return serialization.from_bytes(target, fp.read())


def convert_conv(module_from, module_to):
    # PyTorch kernel has shape [outC, inC, kH, kW] and the Flax kernel has shape [kH, kW, inC, outC]
    module_to.weight = to_torch(module_from['kernel'].transpose(3, 2, 0, 1))
    module_to.bias = to_torch(module_from['bias'])


def convert_conv_after_qkv(module_from, module_to):
    module_to.weight = to_torch(ei.rearrange(module_from['kernel'], "nh h f -> f (nh h) 1 1"))
    module_to.bias = to_torch(module_from['bias'])


def convert_fc(module_from, module_to):
    # PyTorch kernel has shape [outC, inC] and the Flax kernel has shape [inC, outC]
    module_to.weight = to_torch(module_from['kernel'].transpose(1, 0))
    module_to.bias = to_torch(module_from['bias'])


def convert_group_norm(module_from, module_to):
    module_to.weight = to_torch(module_from['scale'])
    module_to.bias = to_torch(module_from['bias'])


def convert_qkv(module_from_q, module_from_k, module_from_v, module_to):
    weight = np.concatenate((module_from_q['kernel'], module_from_k['kernel'], module_from_v['kernel']), 2)
    module_to.weight = to_torch(ei.rearrange(weight, 'f nh h -> (nh h) f 1 1'))
    bias = np.concatenate((module_from_q['bias'], module_from_k['bias'], module_from_v['bias']), 1)
    module_to.bias = to_torch(ei.rearrange(bias, 'nh h -> (nh h)'))


def convert1x1conv(module_from, module_to):
    module_to.weight = to_torch(module_from['kernel'].transpose(1, 0)[:, :, None, None])
    module_to.bias = to_torch(module_from['bias'])


def convert_res_block(module_from, module_to):
    convert_group_norm(module_from['norm1'], module_to.norm1)
    convert_conv(module_from['conv1'], module_to.conv1)
    convert_fc(module_from['temb_proj'], module_to.time[1])
    convert_group_norm(module_from['norm2'], module_to.norm2)
    convert_conv(module_from['conv2'], module_to.conv2)
    if 'nin_shortcut' in module_from:
        convert1x1conv(module_from['nin_shortcut'], module_to.skip)


def convert_attention(module_from, module_to):
    convert_group_norm(module_from['norm'], module_to.norm)
    convert_qkv(module_from['q'], module_from['k'], module_from['v'], module_to.qkv)
    convert_conv_after_qkv(module_from['proj_out'], module_to.out)


def convert_down(module_from, module_to, n_down_blocks, n_res_blocks):
    convert_conv(module_from['conv_in'], module_to[0])
    module_to_idx = 1
    for i in range(n_down_blocks):
        for j in range(n_res_blocks):
            convert_res_block(module_from[f'down_{i}.block_{j}'], module_to[module_to_idx].resblocks)
            if f'down_{i}.attn_{j}' in module_from.keys():
                convert_attention(module_from[f'down_{i}.attn_{j}'], module_to[module_to_idx].attention)
            module_to_idx += 1
        # downsample layer is a res block
        if f'down_{i}.downsample' in module_from.keys():
            convert_res_block(module_from[f'down_{i}.downsample'], module_to[module_to_idx])
            module_to_idx += 1
    assert module_to_idx == len(module_to)


def convert_mid(module_from, module_to):
    convert_res_block(module_from['mid.block_1'], module_to[0].resblocks)
    convert_attention(module_from['mid.attn_1'], module_to[0].attention)
    convert_res_block(module_from['mid.block_2'], module_to[1].resblocks)


def convert_up(module_from, module_to, num_up_blocks, n_res_blocks):
    module_to_idx = 0
    for i in reversed(range(num_up_blocks)):
        for j in range(n_res_blocks + 1):
            convert_res_block(module_from[f'up_{i}.block_{j}'], module_to[module_to_idx].resblocks)
            if f'up_{i}.attn_{j}' in module_from.keys():
                convert_attention(module_from[f'up_{i}.attn_{j}'], module_to[module_to_idx].attention)
            module_to_idx += 1
        # upsample layer is a res block
        if f'up_{i}.upsample' in module_from.keys():
            convert_res_block(module_from[f'up_{i}.upsample'], module_to[module_to_idx])
            module_to_idx += 1
    assert module_to_idx == len(module_to)


def convert_out(module_from, module_to):
    convert_group_norm(module_from['norm_out'], module_to[0])
    convert_conv(module_from['conv_out'], module_to[2])


def convert_time(module_from, module_to):
    convert_fc(module_from['dense0'], module_to[0])
    convert_fc(module_from['dense1'], module_to[2])


def convert_class(module_from, module_to):
    convert_fc(module_from['class_emb'], module_to)


def convert(module_from, module_to, n_down_blocks, n_up_blocks, n_res_blocks, class_conditional=False):
    # downsample
    convert_down(module_from['ema_params'], module_to.down, n_down_blocks, n_res_blocks)
    # mid
    convert_mid(module_from['ema_params'], module_to.mid)
    # up
    convert_up(module_from['ema_params'], module_to.up, n_up_blocks, n_res_blocks)
    # out
    convert_out(module_from['ema_params'], module_to.out)
    # time
    convert_time(module_from['ema_params'], module_to.time)
    # class
    if class_conditional:
        convert_class(module_from['ema_params'], module_to.class_emb)


def cifar10(path: pathlib.Path):
    ckpt = restore_from_path('gs://gresearch/diffusion-distillation/cifar_original', None)
    net = UNet(in_channel=3,
               channel=256,
               emb_channel=1024,
               channel_multiplier=[1, 1, 1],
               n_res_blocks=3,
               attn_rezs=[8, 16],
               attn_heads=1,
               head_dim=None,
               use_affine_time=True,
               dropout=0.2,
               num_output=1,
               resample=True,
               num_classes=1)
    convert(ckpt, net, n_down_blocks=3, n_up_blocks=3, n_res_blocks=3)
    # save torch checkpoint
    torch.save(net.state_dict(), path / 'cifar_original.pt')
    return net


def imagenet64_conditional(path: pathlib.Path):
    ckpt = restore_from_path('gs://gresearch/diffusion-distillation/imagenet_original', None)
    net = UNet(in_channel=3,
               channel=192,
               emb_channel=768,
               channel_multiplier=[1, 2, 3, 4],
               n_res_blocks=3,
               init_rez=64,
               attn_rezs=[8, 16, 32],
               attn_heads=None,
               head_dim=64,
               use_affine_time=True,
               dropout=0.,
               num_output=2,  # predict signal and noise
               resample=True,
               num_classes=1000)
    convert(ckpt, net, n_down_blocks=4, n_up_blocks=4, n_res_blocks=3, class_conditional=True)
    # save torch checkpoint
    torch.save(net.state_dict(), path / 'imagenet_original.pt')
    return net


def main(_):
    path = pathlib.Path(flags.FLAGS.path)
    os.makedirs(path, exist_ok=True)
    imagenet64_conditional(path)
    cifar10(path)


if __name__ == '__main__':
    flags.DEFINE_string('path', './ckpts/', help='Path to save the checkpoints.')
    app.run(main)
