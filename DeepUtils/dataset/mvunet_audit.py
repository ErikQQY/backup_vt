"""Summarize raw train/validation data without changing either split."""
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re

import numpy as np


def _observer_group_size(root):
    manifest_path = root / 'split_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        value = manifest.get('observers_per_field')
        if value is not None and int(value) > 0:
            return int(value), str(manifest_path)
    npz_path = root / 'dataset_manifest.npz'
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as manifest:
            if 'observers_per_field' in manifest:
                value = int(manifest['observers_per_field'].item())
                if value > 0:
                    return value, str(npz_path)
    return None, None


def _summarize(dataset, sample_count, group_size, seed):
    if len(dataset) == 0:
        raise ValueError('Cannot audit an empty MVUNet dataset.')
    patterns = Counter()
    positive_pixels = empty_masks = full_masks = pixels = 0
    fractions = []
    for label in dataset.label:
        mask = label.detach().cpu().numpy()
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError('MVUNet masks must contain only finite binary 0/1 values.')
        count = int(mask.sum())
        positive_pixels += count
        pixels += mask.size
        empty_masks += count == 0
        full_masks += count == mask.size
        fractions.append(count / mask.size)
        patterns[hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()] += 1

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), min(sample_count, len(dataset)), replace=False)
    minimum, maximum = np.full(4, np.inf), np.full(4, -np.inf)
    total, total_squared, channel_pixels = np.zeros(4), np.zeros(4), 0
    for index in indices:
        field = dataset.data[int(index)].detach().cpu().numpy()
        if field.ndim != 3 or field.shape[0] != 4 or not np.isfinite(field).all():
            raise ValueError(f'MVUNet sample {index} must have finite [4,Y,X] channels.')
        flattened = field.reshape(4, -1).astype(np.float64)
        minimum = np.minimum(minimum, flattened.min(axis=1))
        maximum = np.maximum(maximum, flattened.max(axis=1))
        total += flattened.sum(axis=1)
        total_squared += (flattened ** 2).sum(axis=1)
        channel_pixels += flattened.shape[1]
    mean = total / channel_pixels
    std = np.sqrt(np.maximum(total_squared / channel_pixels - mean ** 2, 0))
    sample_types, groups = Counter(), set()
    paths = getattr(dataset, 'sample_paths', [])
    parsed = 0
    for path in paths:
        match = re.fullmatch(r'sample_(\d+)(.*)', Path(path).stem)
        if match:
            parsed += 1
            sample_types[match.group(2) or '(untyped)'] += 1
            if group_size:
                groups.add(int(match.group(1)) // group_size)
    groups_known = group_size is not None and parsed == len(dataset)
    return {
        'samples': len(dataset),
        'pixels': pixels,
        'foreground_fraction': positive_pixels / pixels,
        'empty_masks': int(empty_masks), 'full_masks': int(full_masks),
        'unique_mask_patterns': len(patterns),
        'largest_mask_pattern_count': max(patterns.values()),
        'mask_fraction_quantiles': np.quantile(fractions, [0, .25, .5, .75, 1]).tolist(),
        'observer_groups': len(groups) if groups_known else None,
        'sample_type_counts': dict(sample_types),
        'channel_statistics_samples': len(indices),
        'raw_channel_min': minimum.tolist(), 'raw_channel_max': maximum.tolist(),
        'raw_channel_mean': mean.tolist(), 'raw_channel_std': std.tolist(),
    }, groups if groups_known else None


def audit_mvunet_splits(train_dataset, val_dataset, *, sample_count=512, seed=4259):
    """Counts masks over all samples; channel statistics use a fixed random subset.

    A repeated mask is not proof of duplicate vector fields. Group IDs use the
    explicit SplitMVUNet contract (global sample id // observers_per_field),
    and are left unknown when the group size or IDs cannot be established.
    """
    if sample_count < 1:
        raise ValueError('dataset_audit_samples must be positive.')
    root = Path(train_dataset.directory_path)
    group_size, source = _observer_group_size(root)
    report = {'observers_per_field': group_size, 'group_metadata_source': source,
              'channels': ['u', 'v', 'curl', 'IVD'], 'splits': {}, 'warnings': []}
    groups = {}
    for name, dataset in [('train', train_dataset), ('validation', val_dataset)]:
        if dataset is None:
            continue
        summary, groups[name] = _summarize(dataset, sample_count, group_size, seed)
        report['splits'][name] = summary
        logging.info(
            'MVUNet data %s: samples=%d, observer_groups=%s, mask_patterns=%d, foreground=%.6f.',
            name, summary['samples'], summary['observer_groups'],
            summary['unique_mask_patterns'], summary['foreground_fraction'])
        logging.info('MVUNet raw %s channel mean=%s, std=%s.', name,
                     summary['raw_channel_mean'], summary['raw_channel_std'])
    if groups.get('train') is not None and groups.get('validation') is not None:
        overlap = groups['train'] & groups['validation']
        report['overlapping_observer_groups'] = len(overlap)
        if overlap:
            report['warnings'].append(
                f'{len(overlap)} observer groups appear in both train and validation; check split_manifest.json.')
    else:
        report['overlapping_observer_groups'] = None
        report['warnings'].append('Observer-group separation could not be verified from metadata and filenames.')
    for warning in report['warnings']:
        logging.warning('MVUNet data audit: %s', warning)
    return report
