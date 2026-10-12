"""Physics-aware attention for pathline vortex segmentation.

V2 deliberately lives in a separate module so the established V0 and V1
implementations remain untouched.  It reuses V1's trajectory descriptors and
adds them to the *edges* of the KNN attention graph, rather than only to the
node embedding.  A second, lightweight physics-aware attention pools each
pathline through time.

Both additions are residual adapters with zero-initialized gates.  Loading a
V1 checkpoint into V2 therefore reproduces V1 at initialization and lets
fine-tuning learn only improvements that the validation loss supports.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS
from .pathline_transformer import (
    KNNPathlineTransformerLayer,
    PathlineTransformerV1,
)


class PhysicsAwareKNNPathlineTransformerLayer(
        KNNPathlineTransformerLayer):
    """V1-compatible KNN attention with a head-wise physical edge bias.

    The original V1 attention logits are retained verbatim.  For edge ``i,j``
    and head ``h`` we add

        tanh(g_h) * ( <Wq_h p_i, Wk_h p_j> / sqrt(r)
                      + MLP_h(R_ij) )

    where ``p`` is the normalized V1 physics state and ``R_ij`` contains
    differences/products of vortex-relevant invariants plus graph-structure
    indicators.  The gate ``g`` is initialized to zero by default, making the
    layer exactly V1-compatible when warm-started.
    """

    # V1 physics descriptors most directly related to material rotation and
    # coherent vortical motion.  The full state is still used by the learned
    # query/key projections.
    DEFAULT_CORE_FEATURE_INDICES = (3, 12, 13, 14, 16, 19, 20)

    def __init__(
            self,
            dim,
            physics_dim,
            k=16,
            num_heads=8,
            physics_rank=4,
            relation_hidden_dim=32,
            dropout=0.0,
            physics_bias_init=0.0,
            physics_bias_clip=8.0,
            core_feature_indices=None):
        super().__init__(dim=dim, k=k)
        self.physics_dim = int(physics_dim)
        self.num_heads = int(num_heads)
        self.physics_rank = int(physics_rank)
        self.physics_bias_clip = float(physics_bias_clip)
        if self.num_heads < 1:
            raise ValueError("physics attention requires num_heads >= 1.")
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dmodel={self.dim} must be divisible by "
                f"physics_attention_heads={self.num_heads}.")
        if self.physics_rank < 1:
            raise ValueError("physics_attention_rank must be positive.")

        indices = (
            self.DEFAULT_CORE_FEATURE_INDICES
            if core_feature_indices is None
            else tuple(int(index) for index in core_feature_indices)
        )
        if not indices:
            raise ValueError(
                "At least one core physics feature is required.")
        if min(indices) < 0 or max(indices) >= self.physics_dim:
            raise ValueError(
                f"Physics feature indices {indices} are invalid for "
                f"physics_dim={self.physics_dim}.")
        self.core_feature_indices = indices

        self.physics_norm = nn.LayerNorm(self.physics_dim)
        projection_dim = self.num_heads * self.physics_rank
        self.physics_query = nn.Linear(
            self.physics_dim, projection_dim, bias=False)
        self.physics_key = nn.Linear(
            self.physics_dim, projection_dim, bias=False)

        # signed delta, absolute delta and pairwise product for every selected
        # invariant, followed by seven geometry/graph-structure scalars.
        relation_dim = 3 * len(indices) + 7
        hidden_dim = max(8, int(relation_hidden_dim))
        self.physics_relation_norm = nn.LayerNorm(relation_dim)
        self.physics_relation_mlp = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_heads),
        )
        self.physics_logit_scale = nn.Parameter(torch.full(
            (self.num_heads,), float(physics_bias_init)))

    @staticmethod
    def _neighbor_ids(node_ids, knn_idx):
        """Gather a one-dimensional node-id tensor with batched KNN indices."""
        return node_ids[knn_idx]

    def _physics_bias(
            self, physics, position, knn_idx, time_ids, line_ids,
            group_ids):
        batch_size, num_nodes, _ = physics.shape
        neighbor_count = knn_idx.shape[-1]

        # Prevent an invalid/padded physical value from contaminating all edge
        # logits.  Finite valid V1 descriptors pass through unchanged.
        physics = torch.nan_to_num(
            physics, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                min=-1e4, max=1e4)
        normalized = self.physics_norm(physics)

        query = self.physics_query(normalized).reshape(
            batch_size, num_nodes, self.num_heads, self.physics_rank)
        key = self.physics_key(normalized).reshape(
            batch_size, num_nodes,
            self.num_heads * self.physics_rank)
        neighbor_key = self.get_graph_feature(
            key, knn_idx).reshape(
                batch_size, num_nodes, neighbor_count,
                self.num_heads, self.physics_rank)
        projected_similarity = (
            query.unsqueeze(2) * neighbor_key
        ).sum(dim=-1) / math.sqrt(float(self.physics_rank))

        core = normalized[..., self.core_feature_indices]
        neighbor_core = self.get_graph_feature(core, knn_idx)
        center_core = core.unsqueeze(2)
        core_delta = neighbor_core - center_core

        neighbor_position = self.get_graph_feature(position, knn_idx)
        relative_position = neighbor_position - position.unsqueeze(2)
        spatial_distance = torch.linalg.vector_norm(
            relative_position[..., 0:2], dim=-1)
        physical_delta_t = relative_position[..., 2]

        neighbor_time_ids = self._neighbor_ids(time_ids, knn_idx)
        center_time_ids = time_ids.view(1, num_nodes, 1)
        discrete_delta_t = (
            neighbor_time_ids - center_time_ids).to(physics.dtype)
        time_span = (
            time_ids.max() - time_ids.min()).clamp_min(1).to(physics.dtype)
        normalized_delta_t = discrete_delta_t / time_span

        neighbor_line_ids = self._neighbor_ids(line_ids, knn_idx)
        neighbor_group_ids = self._neighbor_ids(group_ids, knn_idx)
        center_line_ids = line_ids.view(1, num_nodes, 1)
        center_group_ids = group_ids.view(1, num_nodes, 1)
        same_line = neighbor_line_ids.eq(center_line_ids)
        same_group = neighbor_group_ids.eq(center_group_ids)
        same_time = neighbor_time_ids.eq(center_time_ids)

        structure = torch.stack([
            torch.log1p(spatial_distance),
            torch.asinh(physical_delta_t),
            normalized_delta_t,
            normalized_delta_t.abs(),
            same_line.to(physics.dtype),
            same_group.to(physics.dtype),
            same_time.to(physics.dtype),
        ], dim=-1)
        relation = torch.cat([
            core_delta,
            core_delta.abs(),
            center_core * neighbor_core,
            structure,
        ], dim=-1)
        relation_bias = self.physics_relation_mlp(
            self.physics_relation_norm(relation))

        gate = torch.tanh(self.physics_logit_scale).view(
            1, 1, 1, self.num_heads)
        head_bias = gate * (projected_similarity + relation_bias)
        head_bias = head_bias.clamp(
            min=-self.physics_bias_clip,
            max=self.physics_bias_clip,
        )
        return head_bias

    def forward(
            self, x, pos, knn_idx, physics, time_ids, line_ids,
            group_ids, valid=None):
        knn_feat = self.get_graph_feature(x, knn_idx)
        knn_pos = self.get_graph_feature(pos, knn_idx)

        pos_enc = self.pos_mlp(knn_pos - pos.unsqueeze(2))
        query = self.linear_q(x).unsqueeze(2)
        key = self.linear_k(knn_feat)
        value = self.linear_v(knn_feat) + pos_enc

        energy = query - key + pos_enc
        attention_logits = self.attn_mlp(energy)
        physics_bias = self._physics_bias(
            physics=physics,
            position=pos,
            knn_idx=knn_idx,
            time_ids=time_ids,
            line_ids=line_ids,
            group_ids=group_ids,
        )
        attention_shape = attention_logits.shape
        attention_logits = (
            attention_logits.reshape(
                *attention_shape[:-1],
                self.num_heads,
                self.dim // self.num_heads,
            )
            + physics_bias.unsqueeze(-1)
        ).reshape(attention_shape)

        neighbor_valid = None
        if valid is not None:
            neighbor_valid = self.get_graph_feature(
                valid.to(x.dtype).unsqueeze(-1), knn_idx).bool()
            attention_logits = attention_logits.masked_fill(
                ~neighbor_valid, -1e4)
        attention = F.softmax(attention_logits, dim=-2)
        if neighbor_valid is not None:
            attention = attention * neighbor_valid.to(attention.dtype)
            attention = attention / attention.sum(
                dim=-2, keepdim=True).clamp_min(1e-8)

        output = torch.sum(attention * value, dim=2)
        output = F.relu(self.linear_out(output))
        if valid is not None:
            output = output * valid.to(output.dtype).unsqueeze(-1)
        return output


class PhysicsAwareTemporalPool(nn.Module):
    """Head-wise temporal attention used as a gated residual over V1 pooling."""

    def __init__(
            self, dim, physics_dim, num_heads=8, hidden_dim=64,
            dropout=0.0, gate_init=0.0):
        super().__init__()
        self.dim = int(dim)
        self.physics_dim = int(physics_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1 or self.dim % self.num_heads != 0:
            raise ValueError(
                f"dmodel={self.dim} must be divisible by "
                f"temporal_attention_heads={self.num_heads}.")
        self.head_dim = self.dim // self.num_heads

        self.physics_feature_norm = nn.LayerNorm(self.dim)
        self.physics_state_norm = nn.LayerNorm(self.physics_dim)
        hidden_dim = max(8, int(hidden_dim))
        self.physics_score_mlp = nn.Sequential(
            nn.Linear(self.dim + self.physics_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_heads),
        )
        self.physics_pool_projection = nn.Linear(self.dim, self.dim)
        self.physics_pool_scale = nn.Parameter(
            torch.tensor(float(gate_init)))

    def forward(self, features, physics, valid, baseline):
        # features: [B,L,K,C], physics: [B,L,K,P], valid: [B,L,K]
        score_input = torch.cat([
            self.physics_feature_norm(features),
            self.physics_state_norm(torch.nan_to_num(
                physics, nan=0.0, posinf=1e4, neginf=-1e4).clamp(
                    min=-1e4, max=1e4)),
        ], dim=-1)
        scores = self.physics_score_mlp(score_input)
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1e4)
        weights = F.softmax(scores, dim=1)
        weights = weights * valid.to(weights.dtype).unsqueeze(-1)
        weights = weights / weights.sum(
            dim=1, keepdim=True).clamp_min(1e-8)

        batch_size, length, line_count, _ = features.shape
        head_values = features.reshape(
            batch_size, length, line_count,
            self.num_heads, self.head_dim)
        attended = (
            weights.unsqueeze(-1) * head_values
        ).sum(dim=1).reshape(batch_size, line_count, self.dim)
        residual = self.physics_pool_projection(attended)
        return baseline + torch.tanh(self.physics_pool_scale) * residual


@MODELS.register_module()
class PathlineTransformerV2(PathlineTransformerV1):
    """PathlineTransformerV1 with explicit physics-aware attention.

    V1's feature construction, sampling, graph topology, masking, propagation
    and output contract are kept intact.  V2 changes only how neighbors and
    temporal samples are weighted.
    """

    def __init__(
            self, in_channels, PathlineGroups=None, KpathlinePerGroup=4,
            num_classes=1, num_encoder_layers=3, dmodel=144, dropout=0.15,
            k=16, temporal_sampling_ratio=0.5, group_sampling_ratio=0.625,
            full_inference=False, padding_threshold=-999.0,
            gradient_regularization=1e-5,
            physics_attention_heads=8, physics_attention_rank=4,
            physics_relation_hidden_dim=32, physics_bias_init=0.0,
            physics_bias_clip=8.0, physics_core_feature_indices=None,
            temporal_attention_heads=8, temporal_attention_hidden_dim=64,
            temporal_gate_init=0.0, **kwargs):
        super().__init__(
            in_channels=in_channels,
            PathlineGroups=PathlineGroups,
            KpathlinePerGroup=KpathlinePerGroup,
            num_classes=num_classes,
            num_encoder_layers=num_encoder_layers,
            dmodel=dmodel,
            dropout=dropout,
            k=k,
            temporal_sampling_ratio=temporal_sampling_ratio,
            group_sampling_ratio=group_sampling_ratio,
            full_inference=full_inference,
            padding_threshold=padding_threshold,
            gradient_regularization=gradient_regularization,
            **kwargs,
        )
        physics_dim = len(self.physics_feature_names)

        # Preserve every base-layer parameter name so a V1 state dict is a
        # direct strict=False warm start.  Only the physics_* keys are new.
        upgraded_layers = nn.ModuleList()
        for base_layer in self.transformer_layers:
            upgraded = PhysicsAwareKNNPathlineTransformerLayer(
                dim=self.dim,
                physics_dim=physics_dim,
                k=self.knn_k,
                num_heads=physics_attention_heads,
                physics_rank=physics_attention_rank,
                relation_hidden_dim=physics_relation_hidden_dim,
                dropout=dropout,
                physics_bias_init=physics_bias_init,
                physics_bias_clip=physics_bias_clip,
                core_feature_indices=physics_core_feature_indices,
            )
            upgraded.load_state_dict(base_layer.state_dict(), strict=False)
            upgraded_layers.append(upgraded)
        self.transformer_layers = upgraded_layers

        self.physics_temporal_pool = PhysicsAwareTemporalPool(
            dim=self.dim,
            physics_dim=physics_dim,
            num_heads=temporal_attention_heads,
            hidden_dim=temporal_attention_hidden_dim,
            dropout=dropout,
            gate_init=temporal_gate_init,
        )

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if isinstance(checkpoint, (str, bytes, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "A V1 checkpoint must be a path or a mapping.")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Checkpoint 'state_dict' must be a mapping.")

        normalized = {}
        for key, value in state_dict.items():
            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("encoder."):
                key = key[len("encoder."):]
            normalized[key] = value
        return normalized

    def load_v1_checkpoint(self, checkpoint):
        """Warm-start from V1 and reject silent non-V1 incompatibilities."""
        state_dict = self._checkpoint_state_dict(checkpoint)
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_non_physics = [
            key for key in incompatible.missing_keys
            if "physics_" not in key
        ]
        if unexpected or missing_non_physics:
            details = []
            if unexpected:
                details.append(
                    f"unexpected keys: {unexpected[:12]}")
            if missing_non_physics:
                details.append(
                    f"missing V1 keys: {missing_non_physics[:12]}")
            raise RuntimeError(
                "Checkpoint is not compatible with "
                f"PathlineTransformerV1 ({'; '.join(details)}).")
        return incompatible

    @staticmethod
    def _flat_node_ids(
            time_indices, line_indices, pathline_per_group):
        length = time_indices.numel()
        line_count = line_indices.numel()
        time_ids = time_indices[:, None].expand(
            length, line_count).reshape(-1)
        line_ids = line_indices[None, :].expand(
            length, line_count).reshape(-1)
        group_ids = torch.div(
            line_ids, int(pathline_per_group),
            rounding_mode="floor",
        )
        return time_ids, line_ids, group_ids

    def forward(self, data):
        _, pathline_src = data
        physics_features, full_valid = self._compute_physics_features(
            pathline_src)
        time_indices, line_indices, line_mask = self._sample_indices(
            pathline_src)
        sampled_pathline = pathline_src.index_select(
            1, time_indices).index_select(2, line_indices)
        sampled_physics = physics_features.index_select(
            1, time_indices).index_select(2, line_indices)
        valid = full_valid.index_select(
            1, time_indices).index_select(2, line_indices)

        safe_pathline = torch.where(
            valid.unsqueeze(-1),
            sampled_pathline,
            torch.zeros_like(sampled_pathline),
        )
        batch_size, length, line_count, channels = safe_pathline.shape
        points = safe_pathline.reshape(
            batch_size, length * line_count, channels)
        physics = sampled_physics.reshape(
            batch_size, length * line_count,
            len(self.physics_feature_names))
        flat_valid = valid.reshape(batch_size, length * line_count)
        position = points[..., 0:3]
        raw_feature = points[..., 3:7]

        feature = torch.cat([
            self.position_embedding(position),
            self.raw_feature_embedding(raw_feature),
            self.physics_embedding(physics),
        ], dim=-1)
        feature = (
            feature * flat_valid.to(feature.dtype).unsqueeze(-1))
        knn_idx = self._masked_knn(position, flat_valid)
        time_ids, line_ids, group_ids = self._flat_node_ids(
            time_indices=time_indices,
            line_indices=line_indices,
            pathline_per_group=self.pathline_per_group,
        )
        for layer in self.transformer_layers:
            feature = feature + self.residual_dropout(
                layer(
                    feature,
                    position,
                    knn_idx,
                    physics=physics,
                    time_ids=time_ids,
                    line_ids=line_ids,
                    group_ids=group_ids,
                    valid=flat_valid,
                )
            )
        feature = self.norm(feature)
        feature = (
            feature * flat_valid.to(feature.dtype).unsqueeze(-1))
        feature = feature.reshape(
            batch_size, length, line_count, self.dim)

        v1_pool = self._masked_temporal_pool(feature, valid)
        sampled_line_feature = self.physics_temporal_pool(
            features=feature,
            physics=sampled_physics,
            valid=valid,
            baseline=v1_pool,
        )
        full_features = self._propagate_features(
            pathline_src[:, 0, :, 0:3],
            sampled_pathline[:, 0, :, 0:3],
            sampled_line_feature,
            line_mask,
        )
        return self.output(self.fc(full_features)).squeeze(-1)
