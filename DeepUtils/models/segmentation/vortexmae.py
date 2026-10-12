"""VortexMAE with a VortexTransformer pathline compatibility adapter.

The paper model operates on Cartesian 3-D velocity volumes.  PyflowVis's
VortexTransformer dataset instead stores trajectories with shape ``[B,L,K,C]``
and an empty placeholder field.  This module keeps the paper's architectural
core -- 4x4x4 patch embedding, learned coordinate encoding, a four-stage 3-D
Swin encoder, masked reconstruction, and a lightweight U-Net decoder -- and
adds an explicit compatibility adapter for the trajectory data.

The adapter orders pathlines by their physical seed coordinates, places only
configured velocity features into a ``[B,C,L,H,W]`` pseudo-volume, and maps the
segmentation result back to the original pathline order.  This is necessarily
an approximation: pathlines are not a Cartesian Eulerian velocity grid.  For
paper-native dense tensors, use ``input_mode='volume'``.  The default forward
contract remains compatible with the existing trainer: input
``(placeholder_field, pathlines)`` and output probabilities ``[B,K]``.

Reference: Deng et al., "A masked autoencoder-based three-dimensional
foundation model for vortex identification", Physics of Fluids 37, 083624
(2025), DOI 10.1063/5.0281019.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS
from ...loss.build import LOSS


@LOSS.register_module()
class VortexMAELoss(nn.Module):
    """Paper Eq. (22) segmentation loss for probability predictions.

    The top-level :class:`VortexMAE` additionally contributes its cached MAE
    reconstruction term.  Registering the voxel/pathline term separately also
    keeps the encoder usable under the repository's optional ``BaseSeg``
    wrapper.
    """

    def __init__(self, alpha=1.0, beta=1.0, mae_weight=0.0, **kwargs):
        super().__init__()
        del mae_weight, kwargs
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, prediction, target):
        target = target.to(dtype=prediction.dtype)
        return self.alpha * F.binary_cross_entropy(
            prediction, target
        ) + self.beta * F.mse_loss(prediction, target)


def _triple(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"Expected three values, got {value}.")
        return tuple(int(v) for v in value)
    value = int(value)
    return (value, value, value)


def _window_partition(x, window_size):
    """Partition channel-last ``[B,D,H,W,C]`` data into 3-D windows."""
    batch_size, depth, height, width, channels = x.shape
    wd, wh, ww = window_size
    x = x.view(
        batch_size,
        depth // wd,
        wd,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        -1, wd * wh * ww, channels
    )


def _window_reverse(windows, window_size, batch_size, depth, height, width):
    """Reverse :func:`_window_partition`."""
    wd, wh, ww = window_size
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        depth // wd,
        height // wh,
        width // ww,
        wd,
        wh,
        ww,
        channels,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(
        batch_size, depth, height, width, channels
    )


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.activation(self.fc1(x)))
        return self.dropout2(self.fc2(x))


class WindowAttention3D(nn.Module):
    """3-D window attention with learned relative position bias."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.max_window_size = _triple(window_size)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}."
            )
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        wd, wh, ww = self.max_window_size
        bias_count = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(bias_count, self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(self.dim, self.dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def _relative_position_index(self, window_size, device):
        wd, wh, ww = window_size
        max_wd, max_wh, max_ww = self.max_window_size
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(wd, device=device),
                torch.arange(wh, device=device),
                torch.arange(ww, device=device),
                indexing="ij",
            )
        )
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += max_wd - 1
        relative[..., 1] += max_wh - 1
        relative[..., 2] += max_ww - 1
        relative[..., 0] *= (2 * max_wh - 1) * (2 * max_ww - 1)
        relative[..., 1] *= 2 * max_ww - 1
        return relative.sum(dim=-1).long()

    def forward(
        self,
        x,
        window_size,
        attention_mask=None,
        token_valid=None,
    ):
        batch_windows, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        relative_index = self._relative_position_index(
            window_size, device=x.device
        )
        relative_bias = self.relative_position_bias_table[
            relative_index.reshape(-1)
        ].reshape(token_count, token_count, self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_bias.unsqueeze(0).to(attention.dtype)

        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            attention = attention.view(
                batch_windows // window_count,
                window_count,
                self.num_heads,
                token_count,
                token_count,
            )
            attention = attention + attention_mask[None, :, None].to(
                attention.dtype
            )
            attention = attention.view(
                batch_windows, self.num_heads, token_count, token_count
            )

        if token_valid is not None:
            # Invalid/padded tokens must not contribute keys or values.  A
            # finite large-negative value also keeps all-invalid windows free
            # of softmax NaNs; their query outputs are zeroed below.
            attention = attention.masked_fill(
                ~token_valid[:, None, None, :], -10000.0
            )

        attention = self.attention_dropout(F.softmax(attention, dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(
            batch_windows, token_count, channels
        )
        x = self.projection_dropout(self.projection(x))
        if token_valid is not None:
            x = x * token_valid[..., None].to(x.dtype)
        return x


class SwinTransformerBlock3D(nn.Module):
    """A pre-norm W-MSA or SW-MSA block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shifted=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = _triple(window_size)
        self.shifted = bool(shifted)
        self.norm1 = nn.LayerNorm(self.dim)
        self.attention = WindowAttention3D(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(
            self.dim, int(self.dim * float(mlp_ratio)), dropout=dropout
        )

    @staticmethod
    def _axis_slices(window, shift):
        if shift == 0:
            return (slice(0, None),)
        return (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )

    def _attention_mask(self, shape, window_size, shift_size, device):
        depth, height, width = shape
        if not any(shift_size):
            return None
        mask = torch.zeros((1, depth, height, width, 1), device=device)
        counter = 0
        for ds in self._axis_slices(window_size[0], shift_size[0]):
            for hs in self._axis_slices(window_size[1], shift_size[1]):
                for ws in self._axis_slices(window_size[2], shift_size[2]):
                    mask[:, ds, hs, ws, :] = counter
                    counter += 1
        mask_windows = _window_partition(mask, window_size).squeeze(-1)
        attention_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        return attention_mask.masked_fill(
            attention_mask != 0, -100.0
        ).masked_fill(attention_mask == 0, 0.0)

    def forward(self, x, valid=None):
        batch_size, depth, height, width, channels = x.shape
        if valid is None:
            valid = torch.ones(
                batch_size,
                depth,
                height,
                width,
                dtype=torch.bool,
                device=x.device,
            )
        elif valid.shape != (batch_size, depth, height, width):
            raise ValueError(
                "Token-valid mask must match the Swin token grid, got "
                f"{tuple(valid.shape)} and {(batch_size, depth, height, width)}."
            )
        shortcut = x
        x = self.norm1(x)

        window_size = tuple(
            min(configured, actual)
            for configured, actual in zip(
                self.window_size, (depth, height, width)
            )
        )
        shift_size = tuple(
            window // 2 if self.shifted and actual > window else 0
            for window, actual in zip(window_size, (depth, height, width))
        )

        pad_depth = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_height = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_width = (window_size[2] - width % window_size[2]) % window_size[2]
        if pad_depth or pad_height or pad_width:
            x = F.pad(
                x,
                (0, 0, 0, pad_width, 0, pad_height, 0, pad_depth),
            )
            valid = F.pad(
                valid,
                (0, pad_width, 0, pad_height, 0, pad_depth),
                value=False,
            )
        padded_shape = x.shape[1:4]

        if any(shift_size):
            x = torch.roll(
                x, shifts=tuple(-v for v in shift_size), dims=(1, 2, 3)
            )
            valid = torch.roll(
                valid, shifts=tuple(-v for v in shift_size), dims=(1, 2, 3)
            )
        attention_mask = self._attention_mask(
            padded_shape, window_size, shift_size, x.device
        )
        windows = _window_partition(x, window_size)
        valid_windows = _window_partition(
            valid[..., None], window_size
        ).squeeze(-1)
        windows = self.attention(
            windows,
            window_size,
            attention_mask,
            token_valid=valid_windows,
        )
        x = _window_reverse(
            windows,
            window_size,
            batch_size,
            *padded_shape,
        )
        if any(shift_size):
            x = torch.roll(x, shifts=shift_size, dims=(1, 2, 3))
            valid = torch.roll(valid, shifts=shift_size, dims=(1, 2, 3))
        x = x[:, :depth, :height, :width, :].contiguous()
        valid = valid[:, :depth, :height, :width].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x * valid[..., None].to(x.dtype)


class PatchMerging3D(nn.Module):
    """Merge 2x2x2 neighbors and double the feature dimension."""

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(8 * self.dim)
        self.reduction = nn.Linear(8 * self.dim, 2 * self.dim, bias=False)

    def forward(self, x, valid=None):
        _, depth, height, width, _ = x.shape
        if valid is None:
            valid = torch.ones(
                x.shape[0],
                depth,
                height,
                width,
                dtype=torch.bool,
                device=x.device,
            )
        pad_depth = depth % 2
        pad_height = height % 2
        pad_width = width % 2
        if pad_depth or pad_height or pad_width:
            x = F.pad(
                x,
                (0, 0, 0, pad_width, 0, pad_height, 0, pad_depth),
            )
            valid = F.pad(
                valid,
                (0, pad_width, 0, pad_height, 0, pad_depth),
                value=False,
            )
        neighbors = [
            x[:, d::2, h::2, w::2, :]
            for d in range(2)
            for h in range(2)
            for w in range(2)
        ]
        x = torch.cat(neighbors, dim=-1)
        valid = torch.stack(
            [
                valid[:, d::2, h::2, w::2]
                for d in range(2)
                for h in range(2)
                for w in range(2)
            ],
            dim=-1,
        ).any(dim=-1)
        x = self.reduction(self.norm(x))
        return x * valid[..., None].to(x.dtype), valid


class SwinStage3D(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        dropout,
        attention_dropout,
        drop_path_rates,
        downsample,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shifted=(index % 2 == 1),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x, valid):
        for block in self.blocks:
            x = block(x, valid)
        skip = x
        skip_valid = valid
        if self.downsample is not None:
            x, valid = self.downsample(x, valid)
        return skip, x, skip_valid, valid


class CoordinatePositionEncoding3D(nn.Module):
    """Learned MLP encoding of normalized patch-center coordinates."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, shape, device, dtype):
        depth, height, width = shape
        coordinates = torch.stack(
            torch.meshgrid(
                (torch.arange(depth, device=device, dtype=dtype) + 0.5)
                / max(depth, 1),
                (torch.arange(height, device=device, dtype=dtype) + 0.5)
                / max(height, 1),
                (torch.arange(width, device=device, dtype=dtype) + 0.5)
                / max(width, 1),
                indexing="ij",
            ),
            dim=-1,
        )
        coordinates = coordinates.mul(2.0).sub(1.0)
        return self.mlp(coordinates)


class VortexMAEEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size=(4, 4, 4),
        embed_dim=66,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4),
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_size = _triple(patch_size)
        self.embed_dim = int(embed_dim)
        depths = tuple(int(value) for value in depths)
        num_heads = tuple(int(value) for value in num_heads)
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("VortexMAE requires four encoder stages.")

        self.patch_embedding = nn.Conv3d(
            int(in_channels),
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_encoding = CoordinatePositionEncoding3D(self.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        stage_dims = [self.embed_dim * (2**index) for index in range(4)]
        total_depth = sum(depths)
        drop_path_rates = torch.linspace(
            0.0, float(drop_path_rate), total_depth
        ).tolist()
        cursor = 0
        stages = []
        for index in range(4):
            stage_rates = drop_path_rates[cursor : cursor + depths[index]]
            cursor += depths[index]
            stages.append(
                SwinStage3D(
                    dim=stage_dims[index],
                    depth=depths[index],
                    num_heads=num_heads[index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rates=stage_rates,
                    downsample=index < 3,
                )
            )
        self.stages = nn.ModuleList(stages)
        self.output_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for dim in stage_dims]
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _make_mask(token_shape, ratio, device, random, eligible=None):
        batch_size, depth, height, width = token_shape
        token_count = depth * height * width
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        if eligible is None:
            eligible = torch.ones_like(mask)
        else:
            eligible = eligible.reshape(batch_size, token_count).bool()
        for batch_index in range(batch_size):
            indices = torch.nonzero(
                eligible[batch_index], as_tuple=False
            ).flatten()
            masked_count = int(round(indices.numel() * float(ratio)))
            if ratio > 0.0 and indices.numel() > 0:
                masked_count = max(1, masked_count)
            masked_count = min(masked_count, indices.numel())
            if masked_count == 0:
                continue
            if random:
                order = torch.randperm(indices.numel(), device=device)
                selected = indices[order[:masked_count]]
            else:
                positions = torch.linspace(
                    0,
                    indices.numel() - 1,
                    masked_count,
                    device=device,
                ).round().long()
                selected = indices[positions]
            mask[batch_index, selected] = True
        return mask.view(batch_size, depth, height, width)

    def forward(
        self,
        volume,
        mask_ratio=0.0,
        random_mask=True,
        valid_volume=None,
    ):
        x = self.patch_embedding(volume).permute(0, 2, 3, 4, 1)
        if valid_volume is None:
            patch_valid = torch.ones(
                x.shape[:4], dtype=torch.bool, device=x.device
            )
        else:
            patch_valid = F.max_pool3d(
                valid_volume.to(torch.float32),
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).squeeze(1).bool()
            if patch_valid.shape != x.shape[:4]:
                raise ValueError(
                    "Downsampled validity mask does not match patch tokens: "
                    f"{tuple(patch_valid.shape)} versus {tuple(x.shape[:4])}."
                )
        patch_mask = self._make_mask(
            x.shape[:4],
            mask_ratio,
            x.device,
            random=random_mask,
            eligible=patch_valid,
        )
        if patch_mask.any():
            x = torch.where(patch_mask[..., None], self.mask_token, x)
        x = x + self.position_encoding(x.shape[1:4], x.device, x.dtype)
        x = self.embedding_dropout(x) * patch_valid[..., None].to(x.dtype)

        features = []
        for stage, norm in zip(self.stages, self.output_norms):
            skip, x, skip_valid, patch_valid = stage(x, patch_valid)
            skip = norm(skip) * skip_valid[..., None].to(skip.dtype)
            features.append(skip.permute(0, 4, 1, 2, 3).contiguous())
        return features, patch_mask


class ResidualConvBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        return self.activation(self.conv2(x) + residual)


class DecoderFusion3D(nn.Module):
    def __init__(self, in_channels, out_channels, residual_blocks=0):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )
        self.refinement = nn.Sequential(
            *[
                ResidualConvBlock3D(out_channels)
                for _ in range(int(residual_blocks))
            ]
        )

    def forward(self, x):
        return self.refinement(self.fusion(x))


class VortexMAEDecoder3D(nn.Module):
    """Lightweight U-Net decoder used by both reconstruction and segmentation."""

    def __init__(self, embed_dim, patch_size, residual_blocks=0):
        super().__init__()
        patch_size = _triple(patch_size)
        c1, c2, c3, c4 = [int(embed_dim) * (2**index) for index in range(4)]
        self.bottleneck = DecoderFusion3D(c4, c4, residual_blocks)

        self.up4 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.fuse3 = DecoderFusion3D(2 * c3, c3, residual_blocks)
        self.up3 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.fuse2 = DecoderFusion3D(2 * c2, c2, residual_blocks)
        self.up2 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.fuse1 = DecoderFusion3D(2 * c1, c1, residual_blocks)

        # Fig. 2 restores H/4 directly to H at D1.
        self.up1 = nn.ConvTranspose3d(
            c1, c1, kernel_size=patch_size, stride=patch_size
        )
        self.full_resolution = DecoderFusion3D(c1, c1, residual_blocks)
        self.out_channels = c1

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[2:] != reference.shape[2:]:
            x = F.interpolate(
                x,
                size=reference.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return x

    def forward(self, features, output_shape):
        f1, f2, f3, f4 = features
        x = self.bottleneck(f4)
        x = self._resize_like(self.up4(x), f3)
        x = self.fuse3(torch.cat([x, f3], dim=1))
        x = self._resize_like(self.up3(x), f2)
        x = self.fuse2(torch.cat([x, f2], dim=1))
        x = self._resize_like(self.up2(x), f1)
        x = self.fuse1(torch.cat([x, f1], dim=1))
        x = self.up1(x)
        if x.shape[2:] != tuple(output_shape):
            x = F.interpolate(
                x,
                size=tuple(output_shape),
                mode="trilinear",
                align_corners=False,
            )
        return self.full_resolution(x)


@dataclass
class _PathlineLayout:
    line_to_cell: torch.Tensor
    valid_volume: torch.Tensor
    reconstruction_target: torch.Tensor
    original_line_count: int
    grid_shape: tuple[int, int]


class _VortexMAECore(nn.Module):
    VALID_STAGES = {"pretrain", "finetune", "joint"}

    def __init__(
        self,
        in_channels=7,
        input_mode="pathline",
        PathlineGroups=None,
        KpathlinePerGroup=4,
        pathline_grid_size=None,
        layout_mode="coordinate",
        padding_threshold=-999.0,
        normalize_pathlines=True,
        pathline_normalization="minmax",
        normalization_eps=1e-8,
        pathline_input_features=(5, 6),
        reconstruction_features=(5, 6),
        adapter_volume_size=(32, 32, 32),
        patch_size=(4, 4, 4),
        embed_dim=66,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(4, 4, 4),
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.1,
        decoder_residual_blocks=0,
        mask_ratio=0.25,
        training_stage="joint",
        reconstruction_activation="sigmoid",
        segmentation_classes=1,
        center_frame_only=False,
        center_frame_index=None,
        bce_weight=1.0,
        l2_weight=1.0,
        mae_aux_weight=0.1,
        mae_aux_in_eval=False,
        random_mask_in_eval=False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.in_channels = int(in_channels)
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in {"pathline", "volume"}:
            raise ValueError("input_mode must be 'pathline' or 'volume'.")
        self.pathline_groups = (
            None if PathlineGroups is None else int(PathlineGroups)
        )
        self.lines_per_group = int(KpathlinePerGroup)
        if pathline_grid_size is None:
            self.pathline_grid_size = None
        else:
            if not isinstance(pathline_grid_size, Sequence) or len(
                pathline_grid_size
            ) != 2:
                raise ValueError("pathline_grid_size must contain [height, width].")
            self.pathline_grid_size = tuple(
                int(value) for value in pathline_grid_size
            )
            if min(self.pathline_grid_size) < 1:
                raise ValueError("pathline_grid_size values must be positive.")
        self.layout_mode = str(layout_mode).lower()
        if self.layout_mode not in {"coordinate", "index"}:
            raise ValueError("layout_mode must be 'coordinate' or 'index'.")
        self.padding_threshold = float(padding_threshold)
        self.normalize_pathlines = bool(normalize_pathlines)
        self.pathline_normalization = str(pathline_normalization).lower()
        if self.pathline_normalization not in {"minmax", "zscore", "none"}:
            raise ValueError(
                "pathline_normalization must be 'minmax', 'zscore', or 'none'."
            )
        if not self.normalize_pathlines:
            self.pathline_normalization = "none"
        self.normalization_eps = float(normalization_eps)
        self.pathline_input_features = tuple(
            int(index) for index in pathline_input_features
        )
        self.reconstruction_features = tuple(
            int(index) for index in reconstruction_features
        )
        if self.input_mode == "pathline":
            if not self.pathline_input_features:
                raise ValueError("At least one pathline input feature is required.")
            if not self.reconstruction_features:
                raise ValueError("At least one reconstruction feature is required.")
            feature_indices = (
                self.pathline_input_features + self.reconstruction_features
            )
            if min(feature_indices) < 0 or max(feature_indices) >= self.in_channels:
                raise ValueError(
                    "A pathline input/reconstruction feature index is invalid."
                )
            volume_channels = len(self.pathline_input_features)
            reconstruction_channels = len(self.reconstruction_features)
            if adapter_volume_size is None:
                self.adapter_volume_size = None
            else:
                self.adapter_volume_size = _triple(adapter_volume_size)
                if min(self.adapter_volume_size) < 1:
                    raise ValueError("adapter_volume_size values must be positive.")
        else:
            volume_channels = self.in_channels
            reconstruction_channels = self.in_channels
            self.adapter_volume_size = None

        self.patch_size = _triple(patch_size)
        if self.adapter_volume_size is not None:
            minimum_hierarchy = tuple(8 * value for value in self.patch_size)
            if any(
                configured < minimum
                for configured, minimum in zip(
                    self.adapter_volume_size, minimum_hierarchy
                )
            ):
                raise ValueError(
                    "adapter_volume_size must be at least 8 * patch_size on "
                    "each axis so all four encoder scales remain distinct; "
                    f"got {self.adapter_volume_size}."
                )
        self.mask_ratio = float(mask_ratio)
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        self.training_stage = str(training_stage).lower()
        if self.training_stage not in self.VALID_STAGES:
            raise ValueError(
                f"training_stage must be one of {sorted(self.VALID_STAGES)}."
            )
        self.reconstruction_activation = str(reconstruction_activation).lower()
        if self.reconstruction_activation not in {"identity", "sigmoid"}:
            raise ValueError(
                "reconstruction_activation must be 'identity' or 'sigmoid'."
            )
        self.segmentation_classes = int(segmentation_classes)
        if self.segmentation_classes < 1:
            raise ValueError("segmentation_classes must be positive.")
        if self.input_mode == "pathline" and self.segmentation_classes != 1:
            raise ValueError(
                "Pathline mode requires segmentation_classes=1 to match "
                "VortexTransformer labels [B,K]."
            )
        self.center_frame_only = bool(center_frame_only)
        self.center_frame_index = center_frame_index
        self.bce_weight = float(bce_weight)
        self.l2_weight = float(l2_weight)
        self.mae_aux_weight = float(mae_aux_weight)
        self.mae_aux_in_eval = bool(mae_aux_in_eval)
        self.random_mask_in_eval = bool(random_mask_in_eval)

        self.swin = VortexMAEEncoder3D(
            in_channels=volume_channels,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = VortexMAEDecoder3D(
            embed_dim=embed_dim,
            patch_size=self.patch_size,
            residual_blocks=decoder_residual_blocks,
        )
        decoder_channels = self.decoder.out_channels
        self.reconstruction_head = nn.Conv3d(
            decoder_channels, reconstruction_channels, 3, padding=1
        )
        self.segmentation_head = nn.Conv3d(
            decoder_channels, self.segmentation_classes, 3, padding=1
        )
        self._cached_reconstruction_loss: Optional[torch.Tensor] = None
        self.apply(self._kaiming_initialization)

    @staticmethod
    def _kaiming_initialization(module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _grid_shape(line_count):
        height = max(1, int(math.floor(math.sqrt(line_count))))
        width = int(math.ceil(line_count / height))
        return height, width

    def _coordinate_layout(self, pathlines, valid, grid_shape):
        """Assign every line to one unique cell using seed-coordinate ranks."""
        batch_size, _, line_count, _ = pathlines.shape
        height, width = grid_shape
        first_valid = valid.to(torch.int64).argmax(dim=1)
        gather_index = first_valid[:, None, :, None].expand(-1, 1, -1, 2)
        seed_xy = pathlines[..., :2].gather(1, gather_index).squeeze(1)
        any_valid = valid.any(dim=1)
        seed_xy = torch.where(
            any_valid[..., None], seed_xy, torch.full_like(seed_xy, float("inf"))
        )

        line_to_cell = torch.empty(
            batch_size, line_count, dtype=torch.long, device=pathlines.device
        )
        for batch_index in range(batch_size):
            y_order = torch.argsort(seed_xy[batch_index, :, 1], stable=True)
            ordered_lines = []
            for row_start in range(0, line_count, width):
                row_lines = y_order[row_start : row_start + width]
                row_x = seed_xy[batch_index, row_lines, 0]
                ordered_lines.append(row_lines[torch.argsort(row_x, stable=True)])
            cell_to_line = torch.cat(ordered_lines)
            line_to_cell[batch_index, cell_to_line] = torch.arange(
                line_count, device=pathlines.device
            )
        return line_to_cell

    def _normalize_pathline_features(self, pathlines, valid):
        feature_valid = valid[..., None] & torch.isfinite(pathlines)
        safe = torch.where(feature_valid, pathlines, torch.zeros_like(pathlines))
        if self.pathline_normalization == "none":
            return safe
        weight = feature_valid.to(pathlines.dtype)
        if self.pathline_normalization == "zscore":
            denominator = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            mean = (safe * weight).sum(dim=(1, 2), keepdim=True) / denominator
            variance = (
                (safe - mean).square() * weight
            ).sum(dim=(1, 2), keepdim=True) / denominator
            normalized = (safe - mean) / torch.sqrt(
                variance + self.normalization_eps
            )
            return normalized * weight

        positive_inf = torch.full_like(pathlines, float("inf"))
        negative_inf = torch.full_like(pathlines, float("-inf"))
        minimum = torch.where(feature_valid, pathlines, positive_inf).amin(
            dim=(1, 2), keepdim=True
        )
        maximum = torch.where(feature_valid, pathlines, negative_inf).amax(
            dim=(1, 2), keepdim=True
        )
        has_value = feature_valid.any(dim=(1, 2), keepdim=True)
        minimum = torch.where(has_value, minimum, torch.zeros_like(minimum))
        maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
        normalized = (safe - minimum) / (
            maximum - minimum + self.normalization_eps
        )
        return normalized * weight

    def _pathlines_to_volume(self, pathlines):
        if pathlines.ndim != 4:
            raise ValueError(
                "VortexTransformer pathlines must have shape [B,L,K,C], "
                f"got {tuple(pathlines.shape)}."
            )
        batch_size, length, line_count, channels = pathlines.shape
        if length < 1 or line_count < 1:
            raise ValueError("Pathline length and count must both be positive.")
        if channels < self.in_channels:
            raise ValueError(
                f"Expected at least {self.in_channels} pathline features, "
                f"got {channels}."
            )
        pathlines = pathlines[..., : self.in_channels]
        valid = pathlines[..., 0] > self.padding_threshold
        normalized = self._normalize_pathline_features(pathlines, valid)

        if self.pathline_grid_size is None:
            grid_shape = self._grid_shape(line_count)
        else:
            grid_shape = tuple(int(value) for value in self.pathline_grid_size)
            if grid_shape[0] * grid_shape[1] < line_count:
                raise ValueError(
                    f"pathline_grid_size={grid_shape} has fewer than K={line_count} cells."
                )
        if self.layout_mode == "coordinate":
            line_to_cell = self._coordinate_layout(pathlines, valid, grid_shape)
        else:
            line_to_cell = torch.arange(
                line_count, device=pathlines.device
            ).unsqueeze(0).expand(batch_size, -1)

        cell_count = grid_shape[0] * grid_shape[1]
        input_source = normalized[..., self.pathline_input_features].permute(
            0, 3, 1, 2
        ).contiguous()
        target_source = normalized[..., self.reconstruction_features].permute(
            0, 3, 1, 2
        ).contiguous()
        input_channels = len(self.pathline_input_features)
        target_channels = len(self.reconstruction_features)
        volume_flat = normalized.new_zeros(
            batch_size, input_channels, length, cell_count
        )
        input_scatter_index = line_to_cell[:, None, None, :].expand(
            -1, input_channels, length, -1
        )
        volume_flat.scatter_(3, input_scatter_index, input_source)
        volume = volume_flat.view(
            batch_size, input_channels, length, *grid_shape
        )
        target_flat = normalized.new_zeros(
            batch_size, target_channels, length, cell_count
        )
        target_scatter_index = line_to_cell[:, None, None, :].expand(
            -1, target_channels, length, -1
        )
        target_flat.scatter_(3, target_scatter_index, target_source)
        reconstruction_target = target_flat.view(
            batch_size, target_channels, length, *grid_shape
        )

        valid_source = valid.unsqueeze(1)
        valid_flat = torch.zeros(
            batch_size,
            1,
            length,
            cell_count,
            dtype=torch.bool,
            device=pathlines.device,
        )
        valid_flat.scatter_(
            3,
            line_to_cell[:, None, None, :].expand(-1, 1, length, -1),
            valid_source,
        )
        valid_volume = valid_flat.view(batch_size, 1, length, *grid_shape)
        return volume, _PathlineLayout(
            line_to_cell=line_to_cell,
            valid_volume=valid_volume,
            reconstruction_target=reconstruction_target,
            original_line_count=line_count,
            grid_shape=grid_shape,
        )

    def _resize_pathline_inputs(self, volume, valid, reconstruction_target):
        """Upsample the compatibility grid enough to preserve four scales."""
        if self.adapter_volume_size is None:
            return volume, valid, reconstruction_target
        target_shape = tuple(
            max(actual, configured)
            for actual, configured in zip(
                volume.shape[2:], self.adapter_volume_size
            )
        )
        if target_shape == tuple(volume.shape[2:]):
            return volume, valid, reconstruction_target
        volume = F.interpolate(
            volume,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        reconstruction_target = F.interpolate(
            reconstruction_target,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        valid = F.interpolate(
            valid.to(volume.dtype), size=target_shape, mode="nearest"
        ).bool()
        valid_float = valid.to(volume.dtype)
        return (
            volume * valid_float,
            valid,
            reconstruction_target * valid_float,
        )

    def _volume_input(self, data):
        volume = data
        if isinstance(data, (tuple, list)):
            volume = data[0]
        if not torch.is_tensor(volume):
            raise TypeError("Volume input must be a torch.Tensor.")
        if volume.ndim == 4:
            volume = volume.unsqueeze(2)
        if volume.ndim != 5:
            raise ValueError(
                "Dense VortexMAE input must have shape [B,C,D,H,W] or "
                f"[B,D,H,W,C], got {tuple(volume.shape)}."
            )
        if volume.shape[1] == self.in_channels:
            pass
        elif volume.shape[-1] == self.in_channels:
            volume = volume.permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError(
                f"Could not locate {self.in_channels} input channels in "
                f"shape {tuple(volume.shape)}."
            )
        spatial_shape = volume.shape[2:]
        if any(size < patch for size, patch in zip(spatial_shape, self.patch_size)):
            raise ValueError(
                f"Dense input spatial shape {tuple(spatial_shape)} is smaller "
                f"than patch_size={self.patch_size}. The VortexTransformer "
                "placeholder field is not a valid dense flow volume."
            )
        valid = torch.ones(
            volume.shape[0],
            1,
            *spatial_shape,
            dtype=torch.bool,
            device=volume.device,
        )
        return volume, valid

    def _prepare_input(self, data):
        if self.input_mode == "pathline":
            if isinstance(data, (tuple, list)):
                if len(data) < 2:
                    raise ValueError(
                        "Pathline mode expects (placeholder_field, pathlines)."
                    )
                pathlines = data[1]
            else:
                pathlines = data
            volume, layout = self._pathlines_to_volume(pathlines)
            return volume, layout.valid_volume, layout.reconstruction_target, layout
        volume, valid = self._volume_input(data)
        return volume, valid, volume, None

    def _pad_volume(self, volume, valid, reconstruction_target):
        depth, height, width = volume.shape[2:]
        pd, ph, pw = self.patch_size
        pad_depth = (pd - depth % pd) % pd
        pad_height = (ph - height % ph) % ph
        pad_width = (pw - width % pw) % pw
        padding = (0, pad_width, 0, pad_height, 0, pad_depth)
        if any(padding):
            volume = F.pad(volume, padding)
            valid = F.pad(valid, padding, value=False)
            reconstruction_target = F.pad(reconstruction_target, padding)
        return volume, valid, reconstruction_target, (depth, height, width)

    def _decode(self, volume, valid, mask_ratio):
        random_mask = self.training or self.random_mask_in_eval
        features, patch_mask = self.swin(
            volume,
            mask_ratio=mask_ratio,
            random_mask=random_mask,
            valid_volume=valid,
        )
        decoded = self.decoder(features, volume.shape[2:])
        reconstruction = self.reconstruction_head(decoded)
        if self.reconstruction_activation == "sigmoid":
            reconstruction = torch.sigmoid(reconstruction)
        segmentation_logits = self.segmentation_head(decoded)
        return reconstruction, segmentation_logits, patch_mask

    def _voxel_mask(self, patch_mask, output_shape):
        mask = patch_mask[:, None].to(torch.float32)
        return F.interpolate(mask, size=output_shape, mode="nearest").bool()

    def reconstruction_loss(
        self,
        reconstruction,
        target,
        patch_mask,
        valid_volume,
    ):
        voxel_mask = self._voxel_mask(patch_mask, target.shape[2:])
        voxel_weight = voxel_mask & valid_volume
        denominator = voxel_weight.sum().clamp_min(1).to(target.dtype)
        channel_weight = voxel_weight.expand(
            -1, target.shape[1], -1, -1, -1
        ).to(target.dtype)
        squared_error = (reconstruction - target).square()
        # Eq. (16): sum the squared velocity-vector norm, then divide by
        # masked voxel count (not by voxel_count * channel_count).
        return (squared_error * channel_weight).sum() / denominator

    def _pathline_probabilities(self, logits, layout):
        valid = layout.valid_volume
        if logits.shape[1] != 1:
            raise ValueError(
                "Pathline segmentation currently requires segmentation_classes=1."
            )
        valid_float = valid.to(logits.dtype)
        pooled_logits = (logits * valid_float).sum(dim=2) / valid_float.sum(
            dim=2
        ).clamp_min(1.0)
        cell_logits = pooled_logits.flatten(2)
        gathered = torch.gather(
            cell_logits,
            2,
            layout.line_to_cell[:, None, :],
        ).squeeze(1)
        line_valid_cells = valid.any(dim=2).flatten(2)
        line_valid = torch.gather(
            line_valid_cells,
            2,
            layout.line_to_cell[:, None, :],
        ).squeeze(1)
        return torch.where(
            line_valid, torch.sigmoid(gathered), torch.zeros_like(gathered)
        )

    def _dense_probabilities(self, logits, original_shape):
        logits = logits[
            :,
            :,
            : original_shape[0],
            : original_shape[1],
            : original_shape[2],
        ]
        probabilities = torch.sigmoid(logits)
        if self.center_frame_only:
            depth = probabilities.shape[2]
            index = depth // 2 if self.center_frame_index is None else int(
                self.center_frame_index
            )
            if not 0 <= index < depth:
                raise ValueError(
                    f"center_frame_index={index} is invalid for depth={depth}."
                )
            probabilities = probabilities[:, :, index]
        if self.segmentation_classes == 1:
            probabilities = probabilities.squeeze(1)
        return probabilities

    def forward(self, data):
        volume, valid, reconstruction_target, layout = self._prepare_input(data)
        semantic_shape = tuple(volume.shape[2:])
        if layout is not None:
            volume, valid, reconstruction_target = self._resize_pathline_inputs(
                volume, valid, reconstruction_target
            )
        volume, valid, reconstruction_target, model_shape = self._pad_volume(
            volume, valid, reconstruction_target
        )
        needs_reconstruction = self.training_stage == "pretrain" or (
            self.training_stage == "joint"
            and (self.training or self.mae_aux_in_eval)
        )
        mask_ratio = self.mask_ratio if needs_reconstruction else 0.0
        reconstruction, logits, patch_mask = self._decode(
            volume, valid, mask_ratio
        )

        if needs_reconstruction:
            reconstruction_valid = valid
            self._cached_reconstruction_loss = self.reconstruction_loss(
                reconstruction,
                reconstruction_target,
                patch_mask,
                reconstruction_valid,
            )
        else:
            self._cached_reconstruction_loss = None

        if self.training_stage == "pretrain":
            reconstruction = reconstruction[
                :, :, : model_shape[0], : model_shape[1], : model_shape[2]
            ]
            if layout is not None and tuple(model_shape) != semantic_shape:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=semantic_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            return reconstruction
        if layout is not None:
            logits = logits[
                :, :, : model_shape[0], : model_shape[1], : model_shape[2]
            ]
            if tuple(model_shape) != semantic_shape:
                logits = F.interpolate(
                    logits,
                    size=semantic_shape,
                    mode="trilinear",
                    align_corners=False,
                )
            return self._pathline_probabilities(logits, layout)
        return self._dense_probabilities(logits, model_shape)

    def get_loss(self, prediction, target):
        if self.training_stage == "pretrain":
            if self._cached_reconstruction_loss is None:
                raise RuntimeError("forward() must be called before get_loss().")
            loss = self._cached_reconstruction_loss
            self._cached_reconstruction_loss = None
            return loss

        if prediction.shape != target.shape:
            raise ValueError(
                "VortexMAE segmentation prediction and target must match, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        target = target.to(dtype=prediction.dtype)
        segmentation_loss = self.bce_weight * F.binary_cross_entropy(
            prediction, target
        ) + self.l2_weight * F.mse_loss(prediction, target)
        if self.training_stage == "joint":
            if self._cached_reconstruction_loss is not None:
                segmentation_loss = (
                    segmentation_loss
                    + self.mae_aux_weight * self._cached_reconstruction_loss
                )
        self._cached_reconstruction_loss = None
        return segmentation_loss

    def forward_pretrain(self, data, mask_ratio=None):
        """Run explicit masked reconstruction without changing training_stage."""
        volume, valid, target, layout = self._prepare_input(data)
        semantic_shape = tuple(volume.shape[2:])
        if layout is not None:
            volume, valid, target = self._resize_pathline_inputs(
                volume, valid, target
            )
        volume, valid, target, model_shape = self._pad_volume(
            volume, valid, target
        )
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        reconstruction, _, patch_mask = self._decode(volume, valid, ratio)
        loss = self.reconstruction_loss(
            reconstruction, target, patch_mask, valid
        )
        reconstruction = reconstruction[
            :, :, : model_shape[0], : model_shape[1], : model_shape[2]
        ]
        if layout is not None and tuple(model_shape) != semantic_shape:
            reconstruction = F.interpolate(
                reconstruction,
                size=semantic_shape,
                mode="trilinear",
                align_corners=False,
            )
        return reconstruction, patch_mask, loss


@MODELS.register_module()
class VortexMAE(nn.Module):
    """Trainer-compatible VortexMAE wrapper.

    ``encoder_args`` is intentionally accepted at the top level.  The existing
    PyflowVis metadata loader writes VortexTransformer-specific values into
    ``model.encoder_args`` before model construction, while VortexMAE needs to
    own ``get_loss`` so the MAE reconstruction term can participate in the
    standard training loop.
    """

    def __init__(self, encoder_args=None, criterion_args=None, **kwargs):
        super().__init__()
        parameters = dict(encoder_args or {})
        parameters.pop("NAME", None)
        parameters.update(kwargs)
        criterion_args = dict(criterion_args or {})
        aliases = {
            "alpha": "bce_weight",
            "beta": "l2_weight",
            "mae_weight": "mae_aux_weight",
        }
        for source, destination in aliases.items():
            if source in criterion_args:
                parameters[destination] = criterion_args[source]
        for name in ("bce_weight", "l2_weight", "mae_aux_weight"):
            if name in criterion_args:
                parameters[name] = criterion_args[name]
        self.backbone = _VortexMAECore(**parameters)

    def forward(self, data):
        return self.backbone(data)

    def get_loss(self, prediction, target, inputs=None):
        del inputs
        return self.backbone.get_loss(prediction, target)

    def forward_pretrain(self, data, mask_ratio=None):
        return self.backbone.forward_pretrain(data, mask_ratio=mask_ratio)

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError("A checkpoint must be a path or a mapping.")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint 'state_dict' must be a mapping.")
        normalized = {}
        for key, value in state_dict.items():
            while key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("encoder."):
                key = key[len("encoder.") :]
            if not key.startswith("backbone."):
                key = "backbone." + key
            normalized[key] = value
        return normalized

    def load_pretrained_checkpoint(self, checkpoint, strict=False):
        """Load a pretraining/fine-tuning checkpoint.

        Shape-mismatched task heads are skipped so an encoder can be reused
        when reconstruction or segmentation channel counts change.
        """
        state_dict = self._checkpoint_state_dict(checkpoint)
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        if not any(key.startswith("backbone.swin.") for key in compatible):
            raise ValueError(
                "The checkpoint contains no compatible VortexMAE encoder "
                "weights; refusing a silent no-op warm start."
            )
        return self.load_state_dict(compatible, strict=strict)

    def load_v1_checkpoint(self, checkpoint):
        """Compatibility hook used by ``train.py --model_path``."""
        return self.load_pretrained_checkpoint(checkpoint, strict=False)


__all__ = ["VortexMAE"]
