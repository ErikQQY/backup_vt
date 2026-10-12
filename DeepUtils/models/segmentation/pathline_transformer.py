import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from torch import nn
from .samplingLayers import *
# # from ..layers import KANLinear

class KNNPathlineTransformerLayer(nn.Module):
    def __init__(self, dim,k=16):
        super().__init__()
        self.k = k
        self.dim = dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
    
        self.linear_out =nn.Linear(dim, dim)

    def forward(self, x, pos, knn_idx, valid=None):
        # x: (B, N, C), pos: (B, N, 3)
        B, N, C = x.shape
        # Get the features and positions of k-neighbors
        knn_feat = self.get_graph_feature(x, knn_idx)  # (B, N, k, C)
        knn_pos = self.get_graph_feature(pos,knn_idx)  # (B, N, k, 3)
     
        pos_enc = self.pos_mlp(knn_pos - pos.unsqueeze(2))  # (B, N, k, C)
        # pos_enc = self.pos_mlp( pos.unsqueeze(2).repeat(1,1,self.k,1))  # (B, N, k, C)

        q = self.linear_q(x).unsqueeze(2)  # (B, N, 1, C)
        k = self.linear_k(knn_feat)  # (B, N, k, C)
        v = self.linear_v(knn_feat)  # (B, N, k, C)

        energy = q - k + pos_enc  # (B, N, k, C)
        v=v+pos_enc
        attn = self.attn_mlp(energy)
        if valid is not None:
            neighbor_valid = self.get_graph_feature(
                valid.to(x.dtype).unsqueeze(-1), knn_idx).bool()
            attn = attn.masked_fill(~neighbor_valid, -1e4)
        attn = F.softmax(attn, dim=-2)  # (B, N, k, C)
        if valid is not None:
            attn = attn * neighbor_valid.to(attn.dtype)
            attn = attn / attn.sum(dim=-2, keepdim=True).clamp_min(1e-8)

        out = torch.sum(attn * v, dim=2)  # (B, N, C)
        out = F.relu(self.linear_out(out))
        if valid is not None:
            out = out * valid.to(out.dtype).unsqueeze(-1)
        return out

    def get_graph_feature(self, x, idx):
        batch_size, num_points, num_dims = x.size()
        k = idx.size(2)
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        x = x.view(batch_size, num_points, k, num_dims)
        return x
    
# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=100, beta=1000):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        # xyz=xyz.permute(0,2,1)
        B,  N,_= xyz.shape    
        device = xyz.device
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().to(device)  
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        # position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        position_embed = position_embed.reshape(B,N,self.out_dim)
        return position_embed
    



        
 



        


@MODELS.register_module()
class PathlineTransformerV0(nn.Module):
    def __init__(self, in_channels, PathlineGroups,KpathlinePerGroup, num_classes=1, num_encoder_layers=3,dmodel=252,dropout=0.1,k=16,**kwargs):
        super().__init__()
        self.input_dim = in_channels
        self.dim = dmodel
        self.knn_k=k #knn neighbor size
        self.pathlinePerGroup=KpathlinePerGroup
        
        self.keep_Groups = int(0.5* PathlineGroups)
        if  self.keep_Groups % 2 != 0:
            self.keep_Groups= (self.keep_Groups //2) * 2         
        # self.keep_Groups = PathlineGroups    
        # self.raw_pos_embedding = PosE_Initial(3, dmodel//2)#dmodel must be mutiple of inchannels.
        self.raw_pos_embedding = nn.Linear(3, dmodel//2)#dmodel must be mutiple of inchannels.
        self.feature_embedding = nn.Linear(in_channels-3, dmodel//2)
        
        self.transformer_layers = nn.ModuleList([
            KNNPathlineTransformerLayer(dmodel) for _ in range(num_encoder_layers)
        ])
        self.feature_propagation = nn.Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)
        self.fc = nn.Linear(dmodel,  num_classes)
        
        self.output=nn.Sigmoid()
        # self.vector_field_feature_exct=ReferenceFrameCNN(2,32,32,5, self.dim ,dropout=dropout)

    #code of jit script
    # def forward(self, data):
    #     pathline_src=data
    #     linesPerGroup=4
    #     B,L_Full_length,K_totalLInes,_=pathline_src.shape
    #     # tmp_sampled_pathline,temporal_indices=PathlineTemporalSamplingLayer(pathline_src,random=False)
    #     temporal_indices = torch.arange(0, L_Full_length, step=2)[:8]
    #     temporal_indices[0]=0
    #     temporal_indices[-1]=L_Full_length-1
    #     temporal_sampled_pathline = pathline_src[:,temporal_indices,:,:]
        
    #     # sampled_pathline,pathline_mask=PathlineSpatialSamplingLayer(tmp_sampled_pathline,self.keep_Groups,self.pathlinePerGroup,random=False)
    #     group_indices = torch.arange(0, self.inputTotalGroups, step=max(1, self.inputTotalGroups // self.keep_Groups))[:self.keep_Groups]
    #     pathline_mask = torch.zeros(K_totalLInes, dtype=torch.bool)
    #     for idx in group_indices:
    #         start = idx * linesPerGroup
    #         end = start + linesPerGroup
    #         pathline_mask[start:end] = True
    #     sampled_pathline = temporal_sampled_pathline[:, :, pathline_mask, :]

        
    #     _, L, sampleK, C =sampled_pathline.shape
    #     points=sampled_pathline.reshape(B,L*sampleK,C)   
    #     # points: (B, N, 3+C)
    #     pos = points[:, :, :3]
    #     # Find k nearest neighbors
    #     inner = -2 * torch.matmul(pos, pos.transpose(2, 1))
    #     xx = torch.sum(pos**2, dim=2, keepdim=True)
    #     pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #     knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1)[1]  # (B, N, k)
        
    #     pos_emb = self.raw_pos_embedding(pos)
    #     feature= self.feature_embedding(points[:, :, 3:])
    #     x=torch.concat((pos_emb,feature),dim=-1)
        
    #     for layer in self.transformer_layers:
    #         x = x + layer(x, pos,knn_idx)
    #     x = self.norm(x)
    #     x=x.reshape(B,L,sampleK,self.dim)
    #     #x shape [B,K,Dimodel]
    #     # global  pool
    #     x = x.mean(dim=1)+x.max(dim=1)[0]
        
    #     # return  self.output(self.fc(x))
    #     full_features = self.propagate_features(pathline_src[:,0,:,:], sampled_pathline[:,0,:,:], x,pathline_mask)
     
    #     # Apply final layers to get output for all pathlines
    #     full_output = self.output(self.fc(full_features)).squeeze(-1)
    #     return full_output    
    
    def forward(self, data):
        _,pathline_src=data
        
        tmp_sampled_pathline,temporal_indices=PathlineTemporalSamplingLayer(pathline_src,random=True)
        sampled_pathline,pathline_mask=PathlineSpatialSamplingLayer(tmp_sampled_pathline,self.keep_Groups,self.pathlinePerGroup,random=True)
        B, L, sampleK, C =sampled_pathline.shape
        points=sampled_pathline.reshape(B,L*sampleK,C)   
        # points: (B, N, 3+C)
        pos = points[:, :, :3]
        # Find k nearest neighbors
        inner = -2 * torch.matmul(pos, pos.transpose(2, 1))
        xx = torch.sum(pos**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1)[1]  # (B, N, k)
        
        pos_emb = self.raw_pos_embedding(pos)
        feature= self.feature_embedding(points[:, :, 3:])
        x=torch.concat((pos_emb,feature),dim=-1)
        
        for layer in self.transformer_layers:
            x = x + layer(x, pos,knn_idx)
        x = self.norm(x)
        x=x.reshape(B,L,sampleK,self.dim)
        #x shape [B,K,Dimodel]
        # global  pool
        x = x.mean(dim=1)+x.max(dim=1)[0]
        
        # return  self.output(self.fc(x))
        full_features = self.propagate_features(pathline_src[:,0,:,:], sampled_pathline[:,0,:,:], x,pathline_mask)
     
        # Apply final layers to get output for all pathlines
        full_output = self.output(self.fc(full_features)).squeeze(-1)
        return full_output
    
    
    
    
    def propagate_features(self, full_pathline, sampled_pathline, sampled_features,mask):
        B, K, C = full_pathline.shape
        _, sampled_K, _ = sampled_pathline.shape
        
        sampled_start_pos = sampled_pathline.reshape(B, sampled_K, C)[:, :, :3]
        full_start_pos = full_pathline.reshape(B, K, C)[:, :, :3]

        # Calculate pairwise distances
        inner = -2 * torch.matmul(full_start_pos, sampled_start_pos.transpose(2, 1))
        xx = torch.sum(full_start_pos**2, dim=2, keepdim=True)
        yy = torch.sum(sampled_start_pos**2, dim=2, keepdim=True).transpose(2, 1)
        pairwise_distance = xx + inner + yy

        # Find k nearest neighbors
        _, knn_idx = pairwise_distance.topk(k=min(self.knn_k,sampled_K), dim=-1, largest=False)

         # Gather features of k-nearest neighbors
        knn_features = sampled_features.view(B, -1, self.dim).unsqueeze(1).expand(-1, K, -1, -1)
        knn_features = knn_features.gather(2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.dim))
        
        # Calculate weights based on distance
        weights = F.softmax(-pairwise_distance.gather(2, knn_idx), dim=2)

        # Weighted sum of neighbor features
        propagated_features = torch.sum(weights.unsqueeze(-1) * knn_features, dim=2)
        # propagated_features=propagated_features.reshape(B,K,self.dim)
        
        # Combine sampled and propagated features
        full_features = torch.zeros(B, K, self.dim, device=sampled_features.device)
        full_features[:,mask, :] = sampled_features.view(B, -1, self.dim)
        full_features[:, ~mask, :] = self.feature_propagation(propagated_features[:, ~mask, :])

        # shape of full_features=(B,  K, dmodel)
        return full_features

@MODELS.register_module()
class PathlineTransformerV1(nn.Module):
    """V0-compatible transformer augmented with compact physics descriptors.

    Unlike the strict and hybrid SE(2) models, V1 keeps the successful raw
    seven-channel information path and cross-time KNN graph. Physics features
    are computed on the complete trajectory before temporal sampling, embedded
    by a small MLP, and concatenated with raw position/feature embeddings.
    """

    physics_feature_names = (
        "log_ivd",
        "log_distance",
        "log_speed",
        "log_lavd",
        "ivd_rate",
        "log_acceleration",
        "speed_rate",
        "log_arc_length",
        "log_tortuosity",
        "cross_radius",
        "log_relative_speed",
        "radial_velocity",
        "tangential_velocity",
        "local_angular_velocity",
        "angular_velocity_deviation",
        "velocity_divergence",
        "vorticity",
        "log_strain_magnitude",
        "velocity_gradient_determinant",
        "log_swirl_strength",
        "q_criterion",
    )

    def __init__(
            self, in_channels, PathlineGroups=None, KpathlinePerGroup=4,
            num_classes=1, num_encoder_layers=3, dmodel=144, dropout=0.15,
            k=16, temporal_sampling_ratio=0.5, group_sampling_ratio=0.625,
            full_inference=False, padding_threshold=-999.0,
            gradient_regularization=1e-5, **kwargs):
        super().__init__()
        if int(in_channels) < 7:
            raise ValueError(
                "PathlineTransformerV1 expects "
                "[x,y,t,IVD,distance,u,v].")
        if int(KpathlinePerGroup) != 4:
            raise ValueError(
                "PathlineTransformerV1 physics descriptors require the "
                "generator's four-arm cross groups.")
        if int(dmodel) < 12:
            raise ValueError("PathlineTransformerV1 requires dmodel >= 12.")

        self.input_dim = int(in_channels)
        self.dim = int(dmodel)
        self.knn_k = int(k)
        self.pathline_per_group = int(KpathlinePerGroup)
        self.input_groups = (
            None if PathlineGroups is None else int(PathlineGroups))
        self.temporal_sampling_ratio = float(temporal_sampling_ratio)
        self.group_sampling_ratio = float(group_sampling_ratio)
        self.full_inference = bool(full_inference)
        self.padding_threshold = float(padding_threshold)
        self.gradient_regularization = float(gradient_regularization)

        position_dim = self.dim // 3
        raw_feature_dim = self.dim // 3
        physics_dim = self.dim - position_dim - raw_feature_dim
        self.position_embedding = nn.Linear(3, position_dim)
        # Preserve the exact raw V0 information path: IVD, distance, u and v.
        self.raw_feature_embedding = nn.Linear(4, raw_feature_dim)
        physics_feature_dim = len(self.physics_feature_names)
        self.physics_embedding = nn.Sequential(
            nn.LayerNorm(physics_feature_dim),
            nn.Linear(physics_feature_dim, physics_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(physics_dim, physics_dim),
        )

        self.transformer_layers = nn.ModuleList([
            KNNPathlineTransformerLayer(self.dim, k=self.knn_k)
            for _ in range(int(num_encoder_layers))
        ])
        self.residual_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.dim)
        self.feature_propagation = nn.Linear(self.dim, self.dim)
        self.fc = nn.Linear(self.dim, int(num_classes))
        self.output = nn.Sigmoid()

    @staticmethod
    def _deterministic_indices(total, count, device):
        if count >= total:
            return torch.arange(total, device=device)
        return torch.linspace(
            0, total - 1, count, device=device).round().long().unique()

    def _sample_indices(self, pathlines):
        _, length, line_count, _ = pathlines.shape
        if line_count % self.pathline_per_group != 0:
            raise ValueError(
                f"Input has {line_count} lines; complete four-arm groups "
                "are required.")
        group_count = line_count // self.pathline_per_group
        device = pathlines.device

        if not self.training and self.full_inference:
            time_indices = torch.arange(length, device=device)
            group_indices = torch.arange(group_count, device=device)
        else:
            kept_times = min(
                length,
                max(2 if length > 1 else 1,
                    int(round(length * self.temporal_sampling_ratio))),
            )
            kept_groups = min(
                group_count,
                max(1, int(round(
                    group_count * self.group_sampling_ratio))),
            )
            if self.training:
                time_indices = torch.sort(
                    torch.randperm(length, device=device)[:kept_times])[0]
                time_indices[0] = 0
                time_indices[-1] = length - 1
                time_indices = torch.unique(time_indices, sorted=True)
                group_indices = torch.sort(
                    torch.randperm(group_count, device=device)[:kept_groups])[0]
            else:
                time_indices = self._deterministic_indices(
                    length, kept_times, device)
                group_indices = self._deterministic_indices(
                    group_count, kept_groups, device)

        arm_indices = torch.arange(
            self.pathline_per_group, device=device)
        line_indices = (
            group_indices[:, None] * self.pathline_per_group
            + arm_indices[None, :]
        ).reshape(-1)
        line_mask = torch.zeros(
            line_count, dtype=torch.bool, device=device)
        line_mask[line_indices] = True
        return time_indices, line_indices, line_mask

    @staticmethod
    def _masked_group_mean(values, valid):
        weight = valid.to(values.dtype).unsqueeze(-1)
        return (
            (values * weight).sum(dim=3)
            / weight.sum(dim=3).clamp_min(1.0)
        )

    @staticmethod
    def _prepend_zero(values):
        return torch.cat([torch.zeros_like(values[:, :1]), values], dim=1)

    def _compute_physics_features(self, pathlines):
        """Return [B,L,K,P] descriptors from full, un-sampled trajectories."""
        if pathlines.ndim != 4 or pathlines.shape[-1] < 7:
            raise ValueError(
                "Physics features expect [B,L,K,C] with C >= 7.")
        batch_size, length, line_count, channels = pathlines.shape
        if line_count % self.pathline_per_group != 0:
            raise ValueError(
                f"Pathline count {line_count} is not divisible by the "
                f"{self.pathline_per_group}-arm group size.")
        group_count = line_count // self.pathline_per_group
        valid = pathlines[..., 0] > self.padding_threshold
        safe = torch.where(
            valid.unsqueeze(-1), pathlines, torch.zeros_like(pathlines))

        position = safe[..., 0:2]
        time = safe[..., 2]
        ivd = safe[..., 3].clamp_min(0.0)
        distance = safe[..., 4].clamp_min(0.0)
        velocity = safe[..., 5:7]
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        valid_f = valid.to(pathlines.dtype)

        if length > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            pair_valid_f = pair_valid.to(pathlines.dtype)
            delta_t = (
                time[:, 1:] - time[:, :-1]).abs().clamp_min(1e-6)

            ivd_rate_step = (
                (ivd[:, 1:] - ivd[:, :-1])
                / delta_t
                * pair_valid_f
            )
            velocity_delta = velocity[:, 1:] - velocity[:, :-1]
            acceleration_step = (
                velocity_delta
                / delta_t.unsqueeze(-1)
                * pair_valid_f.unsqueeze(-1)
            )
            speed_rate_step = (
                (speed[:, 1:] - speed[:, :-1])
                / delta_t
                * pair_valid_f
            )
            path_increment = (
                torch.linalg.vector_norm(
                    position[:, 1:] - position[:, :-1], dim=-1)
                * pair_valid_f
            )
            lavd_increment = (
                0.5 * (ivd[:, 1:] + ivd[:, :-1])
                * delta_t
                * pair_valid_f
            )
            ivd_rate = self._prepend_zero(ivd_rate_step)
            acceleration = self._prepend_zero(acceleration_step)
            speed_rate = self._prepend_zero(speed_rate_step)
            arc_length = self._prepend_zero(
                torch.cumsum(path_increment, dim=1))
            lavd = self._prepend_zero(
                torch.cumsum(lavd_increment, dim=1))
        else:
            ivd_rate = torch.zeros_like(ivd)
            acceleration = torch.zeros_like(velocity)
            speed_rate = torch.zeros_like(speed)
            arc_length = torch.zeros_like(distance)
            lavd = torch.zeros_like(ivd)

        acceleration_magnitude = torch.linalg.vector_norm(
            acceleration, dim=-1)
        tortuosity = arc_length / distance.clamp_min(1e-4)

        grouped_position = position.reshape(
            batch_size, length, group_count,
            self.pathline_per_group, 2)
        grouped_velocity = velocity.reshape(
            batch_size, length, group_count,
            self.pathline_per_group, 2)
        grouped_valid = valid.reshape(
            batch_size, length, group_count,
            self.pathline_per_group)
        grouped_valid_f = grouped_valid.to(pathlines.dtype)
        center = self._masked_group_mean(
            grouped_position, grouped_valid)
        center_velocity = self._masked_group_mean(
            grouped_velocity, grouped_valid)
        relative_position = (
            grouped_position - center.unsqueeze(3)
        ) * grouped_valid_f.unsqueeze(-1)
        relative_velocity = (
            grouped_velocity - center_velocity.unsqueeze(3)
        ) * grouped_valid_f.unsqueeze(-1)

        radius = torch.linalg.vector_norm(relative_position, dim=-1)
        relative_speed = torch.linalg.vector_norm(
            relative_velocity, dim=-1)
        safe_radius = radius.clamp_min(1e-6)
        radial_velocity = (
            relative_position * relative_velocity
        ).sum(dim=-1) / safe_radius
        tangential_velocity = (
            relative_position[..., 0] * relative_velocity[..., 1]
            - relative_position[..., 1] * relative_velocity[..., 0]
        ) / safe_radius

        rotated_position = torch.stack([
            -relative_position[..., 1],
            relative_position[..., 0],
        ], dim=-1)
        rotation_denominator = (
            relative_position.square().sum(dim=-1)
            * grouped_valid_f
        ).sum(dim=3)
        local_omega = (
            (
                (rotated_position * relative_velocity).sum(dim=-1)
                * grouped_valid_f
            ).sum(dim=3)
            / rotation_denominator.clamp_min(1e-8)
        )
        omega_weight = rotation_denominator
        frame_omega = (
            (local_omega * omega_weight).sum(dim=2)
            / omega_weight.sum(dim=2).clamp_min(1e-8)
        )
        omega_deviation = local_omega - frame_omega.unsqueeze(2)

        # Least-squares local velocity gradient q ~= A r for each cross.
        covariance_rr = torch.einsum(
            "blgai,blgaj->blgij",
            relative_position,
            relative_position,
        )
        covariance_vr = torch.einsum(
            "blgai,blgaj->blgij",
            relative_velocity,
            relative_position,
        )
        identity = torch.eye(
            2, device=pathlines.device, dtype=pathlines.dtype
        ).view(1, 1, 1, 2, 2)
        regularized_rr = (
            covariance_rr + self.gradient_regularization * identity)
        velocity_gradient = torch.linalg.solve(
            regularized_rr,
            covariance_vr.transpose(-1, -2),
        ).transpose(-1, -2)

        du_dx = velocity_gradient[..., 0, 0]
        du_dy = velocity_gradient[..., 0, 1]
        dv_dx = velocity_gradient[..., 1, 0]
        dv_dy = velocity_gradient[..., 1, 1]
        divergence = du_dx + dv_dy
        vorticity = dv_dx - du_dy
        strain_normal = du_dx - dv_dy
        strain_shear = du_dy + dv_dx
        strain_magnitude = torch.sqrt(
            strain_normal.square()
            + strain_shear.square()
            + 1e-12
        )
        gradient_determinant = du_dx * dv_dy - du_dy * dv_dx
        gradient_discriminant = (
            divergence.square() - 4.0 * gradient_determinant)
        swirl_strength = 0.5 * torch.sqrt(
            (-gradient_discriminant).clamp_min(0.0))
        symmetric_norm = (
            du_dx.square()
            + dv_dy.square()
            + 0.5 * strain_shear.square()
        )
        antisymmetric_norm = 0.5 * vorticity.square()
        q_criterion = 0.5 * (
            antisymmetric_norm - symmetric_norm)

        def expand_group(feature):
            return feature.unsqueeze(3).expand(
                -1, -1, -1, self.pathline_per_group).reshape(
                    batch_size, length, line_count)

        physics_features = torch.stack([
            torch.log1p(ivd),
            torch.log1p(distance),
            torch.log1p(speed),
            torch.log1p(lavd.clamp_min(0.0)),
            torch.asinh(ivd_rate),
            torch.log1p(acceleration_magnitude),
            torch.asinh(speed_rate),
            torch.log1p(arc_length),
            torch.log1p(tortuosity.clamp_min(0.0)),
            radius.reshape(batch_size, length, line_count),
            torch.log1p(
                relative_speed.reshape(
                    batch_size, length, line_count)),
            torch.asinh(
                radial_velocity.reshape(
                    batch_size, length, line_count)),
            torch.asinh(
                tangential_velocity.reshape(
                    batch_size, length, line_count)),
            torch.asinh(expand_group(local_omega)),
            torch.asinh(expand_group(omega_deviation)),
            torch.asinh(expand_group(divergence)),
            torch.asinh(expand_group(vorticity)),
            torch.log1p(expand_group(strain_magnitude)),
            torch.asinh(expand_group(gradient_determinant)),
            torch.log1p(expand_group(swirl_strength)),
            torch.asinh(expand_group(q_criterion)),
        ], dim=-1)
        physics_features = (
            physics_features * valid_f.unsqueeze(-1))
        return physics_features, valid

    def encode_physics_features(self, pathlines):
        """Public hook for feature-distribution checks and ablations."""
        physics, valid = self._compute_physics_features(pathlines)
        return physics, valid, self.physics_feature_names

    def _masked_knn(self, position, valid):
        pairwise_distance = torch.cdist(position, position, p=2)
        pairwise_distance = pairwise_distance.masked_fill(
            ~valid.unsqueeze(1),
            torch.finfo(pairwise_distance.dtype).max,
        )
        neighbor_count = min(self.knn_k, position.shape[1])
        return pairwise_distance.topk(
            k=neighbor_count, dim=-1, largest=False)[1]

    @staticmethod
    def _masked_temporal_pool(features, valid):
        weight = valid.to(features.dtype).unsqueeze(-1)
        mean = (
            (features * weight).sum(dim=1)
            / weight.sum(dim=1).clamp_min(1.0)
        )
        features_for_max = features.masked_fill(
            ~valid.unsqueeze(-1), torch.finfo(features.dtype).min)
        maximum = features_for_max.max(dim=1)[0]
        has_valid = valid.any(dim=1)
        maximum = torch.where(
            has_valid.unsqueeze(-1), maximum, torch.zeros_like(maximum))
        return mean + maximum

    def _propagate_features(
            self, full_start_position, sampled_start_position,
            sampled_features, line_mask):
        if bool(line_mask.all()):
            return sampled_features
        full_xy = full_start_position[..., 0:2]
        sampled_xy = sampled_start_position[..., 0:2]
        pairwise_distance = torch.cdist(
            full_xy, sampled_xy, p=2)
        neighbor_count = min(self.knn_k, sampled_xy.shape[1])
        distance, neighbor_idx = pairwise_distance.topk(
            k=neighbor_count, dim=-1, largest=False)
        expanded = sampled_features.unsqueeze(1).expand(
            -1, full_xy.shape[1], -1, -1)
        neighbors = expanded.gather(
            2,
            neighbor_idx.unsqueeze(-1).expand(
                -1, -1, -1, self.dim),
        )
        weights = F.softmax(-distance, dim=2)
        propagated = torch.sum(
            weights.unsqueeze(-1) * neighbors, dim=2)

        full_features = torch.zeros(
            full_xy.shape[0], full_xy.shape[1], self.dim,
            device=sampled_features.device,
            dtype=sampled_features.dtype,
        )
        full_features[:, line_mask] = sampled_features
        full_features[:, ~line_mask] = self.feature_propagation(
            propagated[:, ~line_mask])
        return full_features

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
        for layer in self.transformer_layers:
            feature = feature + self.residual_dropout(
                layer(
                    feature, position, knn_idx,
                    valid=flat_valid,
                )
            )
        feature = self.norm(feature)
        feature = (
            feature * flat_valid.to(feature.dtype).unsqueeze(-1))
        feature = feature.reshape(
            batch_size, length, line_count, self.dim)
        sampled_line_feature = self._masked_temporal_pool(
            feature, valid)

        full_features = self._propagate_features(
            pathline_src[:, 0, :, 0:3],
            sampled_pathline[:, 0, :, 0:3],
            sampled_line_feature,
            line_mask,
        )
        return self.output(self.fc(full_features)).squeeze(-1)


class SerializedPathlineAttention(nn.Module):
    """Patch attention over a serialized, packed pathline point cloud.

    ``order`` and ``inverse`` keep the point features in their original packed
    layout between layers.  Only the temporary QKV tensor is reordered, which
    makes switching the space-filling curve in consecutive layers cheap.
    """

    def __init__(
            self, dim, num_heads=6, patch_size=64, qkv_bias=True,
            attn_drop=0.0, proj_drop=0.0, attention_backend="sdpa"):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        if self.num_heads < 1 or self.dim % self.num_heads != 0:
            raise ValueError(
                f"dmodel={self.dim} must be divisible by "
                f"num_heads={self.num_heads}.")
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if attention_backend not in {"sdpa", "math"}:
            raise ValueError(
                "attention_backend must be either 'sdpa' or 'math'.")

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = float(attn_drop)
        self.attention_backend = attention_backend
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=bool(qkv_bias))
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def _attention(self, query, key, value):
        dropout = self.attn_drop if self.training else 0.0
        if (
                self.attention_backend == "sdpa"
                and hasattr(F, "scaled_dot_product_attention")):
            # PyTorch dispatches this call to Flash/Memory-Efficient Attention
            # when the active device and dtype support it.
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout)

        # Portable/debug fallback.  The default configuration uses SDPA and
        # therefore does not materialize this attention matrix when a fused
        # backend is available.
        score = torch.matmul(
            query.float(), key.float().transpose(-2, -1)) * self.scale
        weight = F.softmax(score, dim=-1).to(query.dtype)
        weight = F.dropout(weight, p=dropout, training=self.training)
        return torch.matmul(weight, value)

    def forward(self, x, order, inverse, pad, unpad, patch_size):
        channels = x.shape[-1]
        # pad maps padded serialized positions to real serialized positions.
        serialized_index = order.index_select(0, pad)
        qkv = self.qkv(x).index_select(0, serialized_index)
        qkv = qkv.reshape(
            -1, patch_size, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = self._attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(-1, channels)

        # Drop borrowed padding tokens and restore the canonical packed order.
        attended = attended.index_select(0, unpad)
        attended = attended.index_select(0, inverse)
        return self.proj_drop(self.proj(attended))


class SerializedPathlineTransformerBlock(nn.Module):
    """PTv3-style pre-norm attention block without external dependencies."""

    def __init__(
            self, dim, num_heads=6, patch_size=64, mlp_ratio=2.0,
            qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
            attention_backend="sdpa"):
        super().__init__()
        hidden_dim = max(int(dim), int(round(float(dim) * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(int(dim))
        self.attn = SerializedPathlineAttention(
            dim=dim,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attention_backend=attention_backend,
        )
        self.norm2 = nn.LayerNorm(int(dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(proj_drop)),
            nn.Linear(hidden_dim, int(dim)),
            nn.Dropout(float(proj_drop)),
        )

    def forward(self, x, order, inverse, pad, unpad, patch_size):
        x = x + self.attn(
            self.norm1(x), order, inverse, pad, unpad, patch_size)
        x = x + self.mlp(self.norm2(x))
        return x


@MODELS.register_module()
class PathlineTransformerV3(PathlineTransformerV1):
    """PathlineTransformerV1 with PointTransformerV3-style attention.

    V1's input embeddings, 21 physics descriptors, sampling, masked temporal
    pooling, feature propagation and output contract are retained.  Its
    quadratic all-pairs distance calculation and KNN attention graph are
    replaced by space-filling-curve serialization and fixed-size patch
    attention.  Valid tokens are packed before attention, so padded pathline
    samples consume neither attention memory nor compute.

    This is a pathline-specific, pure-PyTorch adaptation of PTv3.  It uses
    ``torch.nn.functional.scaled_dot_product_attention`` instead of requiring
    Pointcept's spconv, torch-scatter and flash-attn dependencies.
    """

    supported_serialization_orders = {
        "z", "z-trans", "hilbert", "hilbert-trans",
    }

    def __init__(
            self, in_channels, PathlineGroups=None, KpathlinePerGroup=4,
            num_classes=1, num_encoder_layers=3, dmodel=144, dropout=0.15,
            k=16, temporal_sampling_ratio=0.5, group_sampling_ratio=0.625,
            full_inference=False, padding_threshold=-999.0,
            gradient_regularization=1e-5,
            serialization_orders=(
                "z", "z-trans", "hilbert", "hilbert-trans"),
            serialization_grid_size=0.01, serialization_depth=16,
            patch_size=64, num_heads=6, mlp_ratio=2.0, qkv_bias=True,
            attn_drop=0.0, proj_drop=0.0, shuffle_orders=True,
            attention_backend="sdpa", validate_serialization=False,
            **kwargs):
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

        if isinstance(serialization_orders, str):
            serialization_orders = (serialization_orders,)
        self.serialization_orders = tuple(serialization_orders)
        if not self.serialization_orders:
            raise ValueError("serialization_orders cannot be empty.")
        unknown_orders = (
            set(self.serialization_orders)
            - self.supported_serialization_orders
        )
        if unknown_orders:
            raise ValueError(
                f"Unsupported serialization order(s): "
                f"{sorted(unknown_orders)}.")

        self.serialization_depth = int(serialization_depth)
        if not 1 <= self.serialization_depth <= 16:
            raise ValueError(
                "serialization_depth must be in [1, 16], matching PTv3's "
                "48-bit spatial-code limit.")
        grid_size = torch.as_tensor(
            serialization_grid_size, dtype=torch.float32).flatten()
        if grid_size.numel() == 1:
            grid_size = grid_size.repeat(3)
        if grid_size.numel() != 3:
            raise ValueError(
                "serialization_grid_size must be a scalar or [x, y, t].")
        if not bool(torch.isfinite(grid_size).all()) or bool(
                (grid_size <= 0).any()):
            raise ValueError(
                "serialization_grid_size values must be finite and positive.")
        # Persist quantization scale in checkpoints because changing it changes
        # token order without changing any parameter shape.
        self.register_buffer("serialization_grid_size", grid_size)

        self.patch_size = int(patch_size)
        self.num_heads = int(num_heads)
        self.shuffle_orders = bool(shuffle_orders)
        self.attention_backend = str(attention_backend)
        self.validate_serialization = bool(validate_serialization)
        if int(num_encoder_layers) < 1:
            raise ValueError("num_encoder_layers must be positive.")
        self.active_serialization_order_count = min(
            len(self.serialization_orders), int(num_encoder_layers))

        # Replace the KNN modules constructed by V1.  ``k`` remains in V3 only
        # for V1's sampled-to-full pathline feature propagation.
        self.transformer_layers = nn.ModuleList([
            SerializedPathlineTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                patch_size=self.patch_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                attention_backend=self.attention_backend,
            )
            for _ in range(int(num_encoder_layers))
        ])

    @staticmethod
    def _morton_encode(grid_coord, depth):
        """Encode [..., 3] integer coordinates with a 3-D Morton code."""
        x_coord = grid_coord[..., 0].long()
        y_coord = grid_coord[..., 1].long()
        z_coord = grid_coord[..., 2].long()
        code = torch.zeros_like(x_coord)
        for bit in range(int(depth)):
            code = (
                code
                | (((x_coord >> bit) & 1) << (3 * bit + 2))
                | (((y_coord >> bit) & 1) << (3 * bit + 1))
                | (((z_coord >> bit) & 1) << (3 * bit))
            )
        return code

    @classmethod
    def _hilbert_encode(cls, grid_coord, depth):
        """Encode [..., 3] coordinates using Skilling's Hilbert transform."""
        # Convert axes to the transposed Hilbert representation.  This is a
        # tensorized form of the algorithm used by PTv3's Hilbert serializer.
        axes = grid_coord.long().clone()
        most_significant_bit = 1 << (int(depth) - 1)
        bit = most_significant_bit
        while bit > 1:
            lower_mask = bit - 1
            for dimension in range(3):
                axis_zero = axes[..., 0]
                axis_current = axes[..., dimension]
                exchange = (axis_zero ^ axis_current) & lower_mask
                active = (axis_current & bit) != 0
                new_zero = torch.where(
                    active, axis_zero ^ lower_mask, axis_zero ^ exchange)
                new_current = torch.where(
                    active, axis_current, axis_current ^ exchange)
                axes[..., 0] = new_zero
                if dimension != 0:
                    axes[..., dimension] = new_current
            bit >>= 1

        for dimension in range(1, 3):
            axes[..., dimension] = (
                axes[..., dimension] ^ axes[..., dimension - 1])
        correction = torch.zeros_like(axes[..., 0])
        bit = most_significant_bit
        while bit > 1:
            correction = correction ^ torch.where(
                (axes[..., 2] & bit) != 0,
                torch.full_like(correction, bit - 1),
                torch.zeros_like(correction),
            )
            bit >>= 1
        axes = axes ^ correction.unsqueeze(-1)
        return cls._morton_encode(axes, depth)

    def _encode_serialization_pattern(self, grid_coord, pattern):
        if pattern.endswith("-trans"):
            grid_coord = grid_coord[..., [1, 0, 2]]
            pattern = pattern[:-len("-trans")]
        if pattern == "z":
            return self._morton_encode(
                grid_coord, self.serialization_depth)
        if pattern == "hilbert":
            return self._hilbert_encode(
                grid_coord, self.serialization_depth)
        raise RuntimeError(f"Unexpected serialization pattern: {pattern}.")

    @torch.no_grad()
    def _serialize_positions(self, position, valid):
        """Return packing and bidirectional serialization index mappings."""
        if not torch.jit.is_tracing():
            if position.ndim != 3 or position.shape[-1] != 3:
                raise ValueError("position must have shape [B, N, 3].")
            if valid.shape != position.shape[:2]:
                raise ValueError("valid must have shape [B, N].")

        if self.validate_serialization and not torch.jit.is_tracing():
            finite = torch.isfinite(position).all(dim=-1)
            if bool((valid & ~finite).any()):
                raise ValueError(
                    "Valid pathline positions must contain only finite values.")
        position = torch.nan_to_num(
            position.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)

        masked_position = position.masked_fill(
            ~valid.unsqueeze(-1), float("inf"))
        origin = masked_position.amin(dim=1, keepdim=True)
        has_valid = valid.any(dim=1).view(-1, 1, 1)
        origin = torch.where(has_valid, origin, torch.zeros_like(origin))
        grid_size = self.serialization_grid_size.to(position.device)
        grid_coord = torch.floor(
            (position - origin) / grid_size.view(1, 1, 3)).long()

        max_grid_coord = (1 << self.serialization_depth) - 1
        if self.validate_serialization and not torch.jit.is_tracing():
            valid_grid_coord = torch.where(
                valid.unsqueeze(-1), grid_coord, torch.zeros_like(grid_coord))
            if bool((valid_grid_coord > max_grid_coord).any()):
                observed = int(valid_grid_coord.max().item())
                raise ValueError(
                    f"serialization grid coordinate {observed} exceeds the "
                    f"depth-{self.serialization_depth} limit "
                    f"({max_grid_coord}); increase serialization_grid_size.")
        grid_coord = grid_coord.clamp(min=0, max=max_grid_coord)

        batch_size, node_count = valid.shape
        packed_index = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        batch_id = torch.arange(
            batch_size, device=position.device, dtype=torch.long
        ).view(-1, 1).expand(-1, node_count)
        packed_batch = batch_id.reshape(-1).index_select(0, packed_index)
        packed_grid = grid_coord.reshape(-1, 3).index_select(
            0, packed_index)

        max_batch_size = 1 << (63 - 3 * self.serialization_depth)
        if (
                not torch.jit.is_tracing()
                and batch_size > max_batch_size):
            raise ValueError(
                "Batch index and spatial serialization code exceed int64.")
        batch_prefix = packed_batch << (3 * self.serialization_depth)

        patterns = self.serialization_orders
        if self.training and self.shuffle_orders and len(patterns) > 1:
            permutation = torch.randperm(len(patterns)).tolist()
            patterns = tuple(patterns[index] for index in permutation)
        # Do not encode orders that no attention block can consume.  With the
        # default three V1-depth blocks, training samples three of the four
        # PTv3 patterns while deterministic evaluation uses the first three.
        patterns = patterns[:self.active_serialization_order_count]

        orders = []
        inverses = []
        packed_rank = torch.arange(
            packed_index.numel(), device=position.device, dtype=torch.long)
        for pattern in patterns:
            spatial_code = self._encode_serialization_pattern(
                packed_grid, pattern)
            order = torch.argsort(
                batch_prefix | spatial_code, stable=True)
            inverse = torch.empty_like(order)
            inverse.scatter_(0, order, packed_rank)
            orders.append(order)
            inverses.append(inverse)

        counts = valid.sum(dim=1, dtype=torch.long)
        return (
            packed_index,
            counts,
            torch.stack(orders, dim=0),
            torch.stack(inverses, dim=0),
        )

    @staticmethod
    @torch.no_grad()
    def _serialized_padding(counts, patch_size_max):
        """Build PTv3-style borrowed-token padding for non-empty batches."""
        device = counts.device
        patch_size = int(patch_size_max)
        batch_ids = torch.arange(
            counts.numel(), device=device, dtype=torch.long)

        padded_counts = torch.div(
            counts + patch_size - 1,
            patch_size,
            rounding_mode="floor",
        ) * patch_size
        padded_counts = torch.where(
            counts > 0, padded_counts, torch.zeros_like(padded_counts))
        padded_offsets = torch.cumsum(padded_counts, dim=0)
        serialized_offsets = torch.cumsum(counts, dim=0)
        padded_starts = torch.cat([
            counts.new_zeros(1), padded_offsets[:-1],
        ])
        serialized_starts = torch.cat([
            counts.new_zeros(1), serialized_offsets[:-1],
        ])

        padded_batch = torch.repeat_interleave(batch_ids, padded_counts)
        padded_rank = torch.arange(
            padded_counts.sum(), device=device, dtype=torch.long)
        local_padded_rank = padded_rank - padded_starts.index_select(
            0, padded_batch)
        padded_batch_count = counts.index_select(0, padded_batch)
        extra_rank = local_padded_rank - padded_batch_count

        # Match PTv3 by borrowing missing tail tokens from the preceding part
        # of the same serialized sample.  When an entire sample is shorter
        # than one patch, cycle its own tokens instead of crossing batches.
        borrowed_large = (
            padded_batch_count - patch_size + extra_rank)
        borrowed_small = torch.remainder(
            extra_rank, padded_batch_count.clamp_min(1))
        borrowed_rank = torch.where(
            padded_batch_count >= patch_size,
            borrowed_large,
            borrowed_small,
        )
        source_rank = torch.where(
            local_padded_rank < padded_batch_count,
            local_padded_rank,
            borrowed_rank,
        )
        pad = serialized_starts.index_select(
            0, padded_batch) + source_rank

        serialized_batch = torch.repeat_interleave(batch_ids, counts)
        serialized_rank = torch.arange(
            counts.sum(), device=device, dtype=torch.long)
        local_serialized_rank = (
            serialized_rank
            - serialized_starts.index_select(0, serialized_batch)
        )
        unpad = padded_starts.index_select(
            0, serialized_batch) + local_serialized_rank
        return patch_size, pad, unpad

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

        packed_index, counts, orders, inverses = self._serialize_positions(
            position, flat_valid)
        patch_size, pad, unpad = self._serialized_padding(
            counts, self.patch_size)
        packed_feature = feature.reshape(
            batch_size * length * line_count, self.dim
        ).index_select(0, packed_index)
        for layer_index, layer in enumerate(self.transformer_layers):
            order_index = (
                layer_index % self.active_serialization_order_count)
            packed_feature = layer(
                packed_feature,
                orders[order_index],
                inverses[order_index],
                pad,
                unpad,
                patch_size,
            )

        dense_feature = feature.new_zeros(
            batch_size * length * line_count, self.dim)
        dense_feature.index_copy_(0, packed_index, packed_feature)
        feature = dense_feature.reshape(
            batch_size, length * line_count, self.dim)
        feature = self.norm(feature)
        feature = (
            feature * flat_valid.to(feature.dtype).unsqueeze(-1))
        feature = feature.reshape(
            batch_size, length, line_count, self.dim)
        sampled_line_feature = self._masked_temporal_pool(feature, valid)

        full_features = self._propagate_features(
            pathline_src[:, 0, :, 0:3],
            sampled_pathline[:, 0, :, 0:3],
            sampled_line_feature,
            line_mask,
        )
        return self.output(self.fc(full_features)).squeeze(-1)
