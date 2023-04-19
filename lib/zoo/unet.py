#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

# Mostly copied from https://github.com/rosinality/denoising-diffusion-pytorch
# modified to match https://github.com/google-research/google-research/blob/master/diffusion_distillation/diffusion_distillation/unet.py

import math
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

swish = F.silu


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    """Build sinusoidal embeddings (from Fairseq).
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      max_time: float: largest time input
      dtype: data type of the generated embeddings
    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps *= (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :].to(timesteps)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        scale=1,
        mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
            self, in_channel, out_channel, time_dim, resample, use_affine_time=False, dropout=0, group_norm=32
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        self.resample = resample
        time_out_dim = out_channel
        time_scale = 1

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10

        self.norm1 = nn.GroupNorm(group_norm, in_channel)
        self.activation1 = Swish()
        if self.resample:
            self.updown = {
                'up':  nn.Upsample(scale_factor=2, mode="nearest"),
                'down': nn.AvgPool2d(kernel_size=2, stride=2)
            }[self.resample]

        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        self.norm2 = nn.GroupNorm(group_norm, out_channel)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]
        out = self.norm1(input)
        out = self.activation1(out)

        if self.resample:
            out = self.updown(out)
            input = self.updown(input)

        out = self.conv1(out)

        if self.use_affine_time:
            gamma, beta = self.time(time).view(batch, -1, 1, 1).chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta
        else:
            out = out + self.time(time).view(batch, -1, 1, 1)
            out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, head_dim=None, group_norm=32):
        super().__init__()

        if head_dim is None:
            assert n_head is not None
            assert in_channel % n_head == 0
            self.n_head = n_head
            self.head_dim = in_channel // n_head
        else:
            assert n_head is None
            assert in_channel % head_dim == 0
            self.head_dim = head_dim
            self.n_head = in_channel // head_dim

        self.norm = nn.GroupNorm(group_norm, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, self.n_head, self.head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(self.head_dim)
        attn = attn.view(batch, self.n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, self.n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResBlockWithAttention(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            time_dim,
            dropout,
            resample,
            use_attention=False,
            attention_head: Optional[int] = 1,
            head_dim: Optional[int] = None,
            use_affine_time=False,
            group_norm=32,
    ):
        super().__init__()
        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, resample, use_affine_time, dropout, group_norm=group_norm
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head, head_dim=head_dim, group_norm=group_norm)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


class UNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            channel: int,
            emb_channel: int,
            channel_multiplier: List[int],
            n_res_blocks: int,
            attn_rezs: List[int],
            attn_heads: Optional[int],
            head_dim: Optional[int],
            use_affine_time: bool = False,
            dropout: float = 0,
            num_output: int = 1,
            resample: bool = False,
            init_rez: int = 32,
            logsnr_input_type: str = 'inv_cos',
            logsnr_scale_range: Tuple[float, float] = (-10., 10.),
            num_classes: int = 1
    ):
        super().__init__()

        self.resample = resample
        self.channel = channel
        self.logsnr_input_type = logsnr_input_type
        self.logsnr_scale_range = logsnr_scale_range
        self.num_classes = num_classes
        time_dim = emb_channel
        group_norm = 32

        n_block = len(channel_multiplier)

        if self.num_classes > 1:
            self.class_emb = nn.Linear(self.num_classes, time_dim)

        self.time = nn.Sequential(
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel, channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        cur_rez = init_rez
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        resample=None,
                        use_attention=cur_rez in attn_rezs,
                        attention_head=attn_heads,
                        head_dim=head_dim,
                        use_affine_time=use_affine_time,
                        group_norm=group_norm
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                if self.resample:
                    down_layers.append(ResBlock(
                        in_channel,
                        in_channel,
                        time_dim,
                        resample='down',
                        use_affine_time=use_affine_time,
                        dropout=dropout,
                        group_norm=group_norm
                    ))
                else:
                    down_layers.append(Downsample(in_channel))
                cur_rez = cur_rez // 2
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    resample=None,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    head_dim=head_dim,
                    use_affine_time=use_affine_time,
                    group_norm=group_norm
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    resample=None,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                    group_norm=group_norm
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        resample=None,
                        dropout=dropout,
                        use_attention=cur_rez in attn_rezs,
                        attention_head=attn_heads,
                        head_dim=head_dim,
                        use_affine_time=use_affine_time,
                        group_norm=group_norm
                    )
                )

                in_channel = channel_mult

            if i != 0:
                if self.resample:
                    up_layers.append(ResBlock(
                        in_channel,
                        in_channel,
                        time_dim,
                        resample='up',
                        use_affine_time=use_affine_time,
                        dropout=dropout,
                        group_norm=group_norm
                    ))
                else:
                    up_layers.append(Upsample(in_channel))
                cur_rez = cur_rez * 2

        self.up = nn.ModuleList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(group_norm, in_channel),
            Swish(),
            conv2d(in_channel, 3 * num_output, 3, padding=1, scale=1e-10),
        )

    def get_time_embed(self, logsnr):
        if self.logsnr_input_type == 'linear':
            logsnr_input = (logsnr - self.logsnr_scale_range[0]) / (self.logsnr_scale_range[1] - self.logsnr_scale_range[0])
        elif self.logsnr_input_type == 'sigmoid':
            logsnr_input = torch.sigmoid(logsnr)
        elif self.logsnr_input_type == 'inv_cos':
            logsnr_input = (torch.arctan(torch.exp(-0.5 * torch.clip(logsnr, -20., 20.))) / (0.5 * torch.pi))
        else:
            raise NotImplementedError(self.logsnr_input_type)
        time_emb = get_timestep_embedding(logsnr_input, embedding_dim=self.channel, max_time=1.)
        time_embed = self.time(time_emb)
        return time_embed

    def forward(self, input, logsnr, y=None):
        time_embed = self.get_time_embed(logsnr)

        # Class embedding
        assert self.num_classes >= 1
        if self.num_classes > 1:
            y_emb = nn.functional.one_hot(y, num_classes=self.num_classes).float()
            y_emb = self.class_emb(y_emb)
            time_embed += y_emb
        del y

        feats = []
        out = input
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)
            elif isinstance(layer, ResBlock):
                out = layer(out, time_embed)
            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)
            elif isinstance(layer, ResBlock):
                out = layer(out, time_embed)
            else:
                out = layer(out)

        out = self.out(out)

        return out
