"""Vortex-Seg-Net with dense-grid and pathline input adapters.

The paper network is a small sequential fully convolutional network: eight
3x3 (or 3x3x3) convolutions, no pooling, no skip connections, and channel
widths ``8, 16, 32, 64, 32, 16, 8, 1``.  The first six convolutions use SAME
padding and the final two use VALID padding, so a 15-wide input patch produces
an 11-wide output patch.

PyflowVis stores both dense VortexTransformer velocity grids and sparse
pathlines.  Dense 2-D/3-D tensors use the corresponding paper branch.  The
pathline adapter uses seed-coordinate ordering to build a regular trajectory
volume and maps predictions back to the original line order.  Its depth axis
is trajectory time/arc order rather than Cartesian ``z``; it is trainable with
the existing ``UnsteadyVastisPathlineSeg`` contract, but remains an
approximation of the paper's Cartesian velocity patches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS


def _as_tuple(values, expected_length, name):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    result = tuple(int(value) for value in values)
    if len(result) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {result}."
        )
    return result


class _ConvBNReLU(nn.Module):
    """One Vortex-Seg-Net convolutional layer."""

    def __init__(self, spatial_dims, in_channels, out_channels, padding, relu):
        super().__init__()
        convolution = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        batch_norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        self.convolution = convolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.batch_norm = batch_norm(out_channels)
        self.activation = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.activation(self.batch_norm(self.convolution(x)))


class _VortexSegNetCore(nn.Module):
    """The paper's eight-layer 2-D or 3-D FCN branch."""

    def __init__(self, spatial_dims, input_channels, feature_channels):
        super().__init__()
        self.spatial_dims = int(spatial_dims)
        if self.spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3.")
        feature_channels = _as_tuple(
            feature_channels, 8, "feature_channels"
        )
        if feature_channels[-1] != 1:
            raise ValueError("The final Vortex-Seg-Net channel must be 1.")

        layers = []
        current_channels = int(input_channels)
        for index, output_channels in enumerate(feature_channels):
            # Layers 1-6 are SAME; layers 7-8 are VALID.
            padding = 1 if index < 6 else 0
            # The paper says BN/ReLU follow every convolution, but a terminal
            # ReLU is inconsistent with BCE probabilities.  Keep BN on layer
            # eight and apply sigmoid outside the core; ReLU is used on all
            # seven hidden layers.
            layers.append(
                _ConvBNReLU(
                    self.spatial_dims,
                    current_channels,
                    output_channels,
                    padding=padding,
                    relu=index < 7,
                )
            )
            current_channels = output_channels
        self.layers = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        minimum_size = 5
        if any(size < minimum_size for size in x.shape[2:]):
            raise ValueError(
                "Vortex-Seg-Net needs every spatial axis to be at least 5 "
                "before its two VALID convolutions; got "
                f"{tuple(x.shape[2:])}."
            )
        return self.layers(x)


@dataclass
class _PathlineLayout:
    line_to_cell: torch.Tensor
    valid_volume: torch.Tensor
    grid_shape: tuple[int, int]


@MODELS.register_module()
class VortexSegNet(nn.Module):
    """Paper Vortex-Seg-Net with automatic PyflowVis input adaptation.

    Supported inputs:

    - dense 2-D velocity: ``[B,C,H,W]`` or ``[B,H,W,C]`` -> ``[B,H,W]``;
    - dense 3-D velocity: ``[B,C,D,H,W]`` or channel-last -> ``[B,D,H,W]``;
    - VortexTransformer pathlines: ``(dummy, [B,L,K,C])`` -> ``[B,K]``.

    ``preserve_input_shape=True`` adds the two-voxel halo required to apply
    the paper's two VALID convolutions to complete fields.  Set it to ``False``
    for the literal paper patch contract, e.g. 15^3 -> 11^3.
    """

    VALID_INPUT_MODES = {
        "auto",
        "pathline",
        "velocity",
        "velocity2d",
        "velocity3d",
        "dense2d",
        "dense3d",
    }

    def __init__(
        self,
        input_mode="auto",
        spatial_dims="auto",
        in_channels=7,
        dense_velocity_features=(0, 1),
        dense_3d_velocity_features=(0, 1, 2),
        pathline_velocity_features=(5, 6),
        feature_channels=(8, 16, 32, 64, 32, 16, 8, 1),
        preserve_input_shape=True,
        boundary_padding="replicate",
        normalize_input=True,
        normalization="minmax",
        normalization_eps=1e-8,
        pathline_grid_size=None,
        layout_mode="coordinate",
        padding_threshold=-999.0,
        PathlineGroups=None,
        KpathlinePerGroup=4,
        dense_channel_axis="auto",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_mode = str(input_mode).lower()
        if self.input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {sorted(self.VALID_INPUT_MODES)}."
            )
        if spatial_dims == "auto":
            self.spatial_dims = "auto"
        else:
            self.spatial_dims = int(spatial_dims)
            if self.spatial_dims not in {2, 3}:
                raise ValueError("spatial_dims must be 'auto', 2, or 3.")

        self.in_channels = int(in_channels)
        self.dense_velocity_features = tuple(
            int(index) for index in dense_velocity_features
        )
        self.dense_3d_velocity_features = tuple(
            int(index) for index in dense_3d_velocity_features
        )
        self.pathline_velocity_features = tuple(
            int(index) for index in pathline_velocity_features
        )
        if len(self.dense_velocity_features) != 2:
            raise ValueError(
                "dense_velocity_features must select exactly [u,v]."
            )
        if len(self.dense_3d_velocity_features) != 3:
            raise ValueError(
                "dense_3d_velocity_features must select exactly [u,v,w]."
            )
        if not self.pathline_velocity_features:
            raise ValueError("pathline_velocity_features cannot be empty.")
        if min(self.dense_velocity_features) < 0:
            raise ValueError("dense velocity feature indices must be nonnegative.")
        if min(self.dense_3d_velocity_features) < 0:
            raise ValueError(
                "dense 3-D velocity feature indices must be nonnegative."
            )
        if min(self.pathline_velocity_features) < 0:
            raise ValueError("pathline velocity feature indices must be nonnegative.")

        self.preserve_input_shape = bool(preserve_input_shape)
        self.boundary_padding = str(boundary_padding).lower()
        if self.boundary_padding not in {
            "constant",
            "replicate",
            "reflect",
            "circular",
        }:
            raise ValueError(
                "boundary_padding must be constant, replicate, reflect, or circular."
            )
        self.normalize_input = bool(normalize_input)
        self.normalization = str(normalization).lower()
        if self.normalization not in {"minmax", "zscore", "none"}:
            raise ValueError("normalization must be minmax, zscore, or none.")
        if not self.normalize_input:
            self.normalization = "none"
        self.normalization_eps = float(normalization_eps)
        if self.normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive.")

        if pathline_grid_size is None:
            self.pathline_grid_size = None
        else:
            self.pathline_grid_size = _as_tuple(
                pathline_grid_size, 2, "pathline_grid_size"
            )
            if min(self.pathline_grid_size) < 1:
                raise ValueError("pathline_grid_size values must be positive.")
        self.layout_mode = str(layout_mode).lower()
        if self.layout_mode not in {"coordinate", "index"}:
            raise ValueError("layout_mode must be coordinate or index.")
        self.padding_threshold = float(padding_threshold)
        self.pathline_groups = (
            None if PathlineGroups is None else int(PathlineGroups)
        )
        self.lines_per_group = int(KpathlinePerGroup)

        self.dense_channel_axis = str(dense_channel_axis).lower()
        if self.dense_channel_axis not in {"auto", "first", "last"}:
            raise ValueError("dense_channel_axis must be auto, first, or last.")

        build_2d = self.spatial_dims in {"auto", 2}
        build_3d = self.spatial_dims in {"auto", 3}
        self.core_2d = (
            _VortexSegNetCore(2, 2, feature_channels) if build_2d else None
        )
        self.core_3d = (
            _VortexSegNetCore(3, 3, feature_channels) if build_3d else None
        )

    @staticmethod
    def _grid_shape(line_count):
        height = max(1, int(math.floor(math.sqrt(line_count))))
        width = int(math.ceil(line_count / height))
        return height, width

    @staticmethod
    def _pad_channels(x, channel_count):
        if x.shape[1] > channel_count:
            return x[:, :channel_count]
        if x.shape[1] == channel_count:
            return x
        zeros = x.new_zeros(
            x.shape[0],
            channel_count - x.shape[1],
            *x.shape[2:],
        )
        return torch.cat([x, zeros], dim=1)

    def _normalize(self, x, valid=None):
        if valid is None:
            valid = torch.ones(
                x.shape[0],
                1,
                *x.shape[2:],
                dtype=torch.bool,
                device=x.device,
            )
        feature_valid = valid.expand_as(x) & torch.isfinite(x)
        safe = torch.where(feature_valid, x, torch.zeros_like(x))
        if self.normalization == "none":
            return safe

        reduce_dims = tuple(range(2, x.ndim))
        weight = feature_valid.to(x.dtype)
        if self.normalization == "zscore":
            denominator = weight.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
            mean = (safe * weight).sum(dim=reduce_dims, keepdim=True) / denominator
            variance = (
                (safe - mean).square() * weight
            ).sum(dim=reduce_dims, keepdim=True) / denominator
            normalized = (safe - mean) / torch.sqrt(
                variance + self.normalization_eps
            )
            return normalized * weight

        positive_inf = torch.full_like(x, float("inf"))
        negative_inf = torch.full_like(x, float("-inf"))
        minimum = torch.where(feature_valid, x, positive_inf).amin(
            dim=reduce_dims, keepdim=True
        )
        maximum = torch.where(feature_valid, x, negative_inf).amax(
            dim=reduce_dims, keepdim=True
        )
        has_value = feature_valid.any(dim=reduce_dims, keepdim=True)
        minimum = torch.where(has_value, minimum, torch.zeros_like(minimum))
        maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))
        normalized = (safe - minimum) / (
            maximum - minimum + self.normalization_eps
        )
        return normalized * weight

    def _with_halo(self, x):
        padding = (2, 2) * (x.ndim - 2)
        if self.boundary_padding == "constant":
            return F.pad(x, padding, mode="constant", value=0.0)
        if self.boundary_padding == "reflect" and any(
            size <= 2 for size in x.shape[2:]
        ):
            raise ValueError(
                "reflect boundary padding requires every spatial axis > 2."
            )
        return F.pad(x, padding, mode=self.boundary_padding)

    def _channel_first(self, x, required_channels):
        if self.dense_channel_axis == "first":
            if x.shape[1] < required_channels:
                raise ValueError(
                    f"Channel-first input has only {x.shape[1]} channels."
                )
            return x
        if self.dense_channel_axis == "last":
            if x.shape[-1] < required_channels:
                raise ValueError(
                    f"Channel-last input has only {x.shape[-1]} channels."
                )
            order = (0, x.ndim - 1, *range(1, x.ndim - 1))
            return x.permute(order).contiguous()

        first_plausible = required_channels <= x.shape[1] <= 8
        last_plausible = required_channels <= x.shape[-1] <= 8
        if first_plausible and not last_plausible:
            return x
        if last_plausible and not first_plausible:
            order = (0, x.ndim - 1, *range(1, x.ndim - 1))
            return x.permute(order).contiguous()
        if first_plausible:
            # PyflowVis datasets emit channel-first tensors.
            return x
        raise ValueError(
            "Could not identify the velocity channel axis in shape "
            f"{tuple(x.shape)}; set dense_channel_axis explicitly."
        )

    def _prepare_dense(self, data, spatial_dims):
        if isinstance(data, (tuple, list)):
            if not data:
                raise ValueError("Dense input tuple/list is empty.")
            data = data[0]
        if not torch.is_tensor(data):
            raise TypeError("Dense Vortex-Seg-Net input must be a tensor.")
        expected_ndim = spatial_dims + 2
        if data.ndim != expected_ndim:
            raise ValueError(
                f"Dense {spatial_dims}-D input must have {expected_ndim} "
                f"dimensions, got {tuple(data.shape)}."
            )
        feature_indices = (
            self.dense_velocity_features
            if spatial_dims == 2
            else self.dense_3d_velocity_features
        )
        required_channels = max(feature_indices) + 1
        x = self._channel_first(data, required_channels)
        if max(feature_indices) >= x.shape[1]:
            raise ValueError(
                f"Dense {spatial_dims}-D velocity features={feature_indices} are "
                f"invalid for {x.shape[1]} channels."
            )
        x = x[:, feature_indices]
        target_channels = 2 if spatial_dims == 2 else 3
        x = self._pad_channels(x, target_channels)
        return self._normalize(x)

    def _coordinate_layout(self, pathlines, valid, grid_shape):
        batch_size, _, line_count, _ = pathlines.shape
        width = grid_shape[1]
        first_valid = valid.to(torch.int64).argmax(dim=1)
        gather_index = first_valid[:, None, :, None].expand(-1, 1, -1, 2)
        seed_xy = pathlines[..., :2].gather(1, gather_index).squeeze(1)
        seed_xy = torch.where(
            valid.any(dim=1)[..., None],
            seed_xy,
            torch.full_like(seed_xy, float("inf")),
        )

        line_to_cell = torch.empty(
            batch_size, line_count, dtype=torch.long, device=pathlines.device
        )
        for batch_index in range(batch_size):
            y_order = torch.argsort(seed_xy[batch_index, :, 1], stable=True)
            ordered_rows = []
            for row_start in range(0, line_count, width):
                row_lines = y_order[row_start : row_start + width]
                row_x = seed_xy[batch_index, row_lines, 0]
                ordered_rows.append(
                    row_lines[torch.argsort(row_x, stable=True)]
                )
            cell_to_line = torch.cat(ordered_rows)
            line_to_cell[batch_index, cell_to_line] = torch.arange(
                line_count, device=pathlines.device
            )
        return line_to_cell

    def _prepare_pathlines(self, data):
        if isinstance(data, (tuple, list)):
            if len(data) < 2:
                raise ValueError(
                    "Pathline mode expects (placeholder_field, pathlines)."
                )
            pathlines = data[1]
        else:
            pathlines = data
        if not torch.is_tensor(pathlines) or pathlines.ndim != 4:
            raise ValueError(
                "Pathlines must be a tensor [B,L,K,C], got "
                f"{type(pathlines).__name__} "
                f"{getattr(pathlines, 'shape', None)}."
            )
        batch_size, length, line_count, feature_count = pathlines.shape
        required_features = max(self.pathline_velocity_features) + 1
        if feature_count < required_features:
            raise ValueError(
                f"Expected at least {required_features} pathline features, "
                f"got {feature_count}."
            )
        if length < 1 or line_count < 1:
            raise ValueError("Pathline length and count must be positive.")

        valid = pathlines[..., 0] > self.padding_threshold
        velocity = pathlines[..., self.pathline_velocity_features].permute(
            0, 3, 1, 2
        ).contiguous()
        velocity = self._normalize(velocity, valid[:, None])

        grid_shape = (
            self._grid_shape(line_count)
            if self.pathline_grid_size is None
            else self.pathline_grid_size
        )
        if grid_shape[0] * grid_shape[1] < line_count:
            raise ValueError(
                f"pathline_grid_size={grid_shape} has fewer than K={line_count} cells."
            )
        if self.layout_mode == "coordinate":
            line_to_cell = self._coordinate_layout(
                pathlines, valid, grid_shape
            )
        else:
            line_to_cell = torch.arange(
                line_count, device=pathlines.device
            ).unsqueeze(0).expand(batch_size, -1)

        cell_count = grid_shape[0] * grid_shape[1]
        input_channels = velocity.shape[1]
        volume_flat = velocity.new_zeros(
            batch_size, input_channels, length, cell_count
        )
        scatter_index = line_to_cell[:, None, None, :].expand(
            -1, input_channels, length, -1
        )
        volume_flat.scatter_(3, scatter_index, velocity)
        volume = volume_flat.view(
            batch_size, input_channels, length, *grid_shape
        )

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
            valid[:, None],
        )
        valid_volume = valid_flat.view(
            batch_size, 1, length, *grid_shape
        )
        volume = self._pad_channels(volume, 3)
        return volume, _PathlineLayout(
            line_to_cell=line_to_cell,
            valid_volume=valid_volume,
            grid_shape=grid_shape,
        )

    def _forward_2d(self, data):
        if self.core_2d is None:
            raise ValueError("The 2-D Vortex-Seg-Net branch is disabled.")
        x = self._prepare_dense(data, spatial_dims=2)
        if self.preserve_input_shape:
            x = self._with_halo(x)
        logits = self.core_2d(x).squeeze(1)
        return torch.sigmoid(logits)

    def _forward_3d(self, data):
        if self.core_3d is None:
            raise ValueError("The 3-D Vortex-Seg-Net branch is disabled.")
        x = self._prepare_dense(data, spatial_dims=3)
        if self.preserve_input_shape:
            x = self._with_halo(x)
        logits = self.core_3d(x).squeeze(1)
        return torch.sigmoid(logits)

    def _forward_pathlines(self, data):
        if self.core_3d is None:
            raise ValueError("Pathline mode requires the 3-D branch.")
        volume, layout = self._prepare_pathlines(data)
        # Pathline labels cover every seed, so always add the two-cell halo and
        # retain the semantic LxHxW shape even in literal paper-patch mode.
        logits = self.core_3d(self._with_halo(volume))
        valid = layout.valid_volume.to(logits.dtype)
        pooled_logits = (logits * valid).sum(dim=2) / valid.sum(
            dim=2
        ).clamp_min(1.0)
        cell_logits = pooled_logits.flatten(2)
        gathered = torch.gather(
            cell_logits, 2, layout.line_to_cell[:, None, :]
        ).squeeze(1)
        cell_valid = layout.valid_volume.any(dim=2).flatten(2)
        line_valid = torch.gather(
            cell_valid, 2, layout.line_to_cell[:, None, :]
        ).squeeze(1)
        probabilities = torch.sigmoid(gathered)
        return torch.where(
            line_valid, probabilities, torch.zeros_like(probabilities)
        )

    def forward(self, data):
        mode = self.input_mode
        if mode == "auto":
            if isinstance(data, (tuple, list)):
                mode = "pathline"
            elif torch.is_tensor(data) and data.ndim == 4:
                mode = "velocity2d"
            elif torch.is_tensor(data) and data.ndim == 5:
                mode = "velocity3d"
            else:
                raise ValueError(
                    "Could not infer Vortex-Seg-Net input mode from "
                    f"{type(data).__name__}."
                )
        if mode == "pathline":
            return self._forward_pathlines(data)
        if mode in {"velocity2d", "dense2d"}:
            return self._forward_2d(data)
        if mode in {"velocity3d", "dense3d"}:
            return self._forward_3d(data)
        if mode == "velocity":
            tensor = data[0] if isinstance(data, (tuple, list)) else data
            if not torch.is_tensor(tensor):
                raise TypeError("velocity mode expects a tensor.")
            if tensor.ndim == 4:
                return self._forward_2d(data)
            if tensor.ndim == 5:
                return self._forward_3d(data)
            raise ValueError(
                "velocity mode expects a 4-D or 5-D batched tensor."
            )
        raise RuntimeError(f"Unhandled input_mode={mode!r}.")


__all__ = ["VortexSegNet"]
