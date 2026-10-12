"""Joint grid symmetries for raw [u, v, curl, IVD] fields and [Y,X] masks."""
import torch


def augment_mvunet_pair(field, target, *, k=None, reflect=None, allow_quarter_turns=True):
    """Apply an exact grid rotation/reflection before scalar normalization.

    Rows increase with physical y in these binary files. Therefore rot90(k=1)
    maps physical vectors (u,v) to (v,-u), not (-v,u). Reflections reverse curl
    (a pseudoscalar); IVD and segmentation are scalars. No interpolation or
    in-place modification of the preloaded tensors is used.
    """
    if field.ndim != 3 or field.shape[0] != 4 or target.shape != field.shape[1:]:
        raise ValueError('MVUNet augmentation requires [4,Y,X] inputs and [Y,X] targets.')
    if k is None:
        quarter_turns = allow_quarter_turns and field.shape[-2] == field.shape[-1]
        k = int(torch.randint(4 if quarter_turns else 2, ()).item())
        if not quarter_turns:
            k *= 2
    if k not in (0, 1, 2, 3):
        raise ValueError('Rotation k must be 0, 1, 2 or 3.')
    if reflect is None:
        reflect = bool(torch.randint(2, ()).item())
    result, mask = field.clone(), target.clone()
    if reflect:
        result = result.flip(-1)
        mask = mask.flip(-1)
        result[0].neg_()
        result[2].neg_()
    result = torch.rot90(result, k, (-2, -1))
    mask = torch.rot90(mask, k, (-2, -1))
    velocity = result[:2].clone()
    if k == 1:
        result[0], result[1] = velocity[1], -velocity[0]
    elif k == 2:
        result[:2] = -velocity
    elif k == 3:
        result[0], result[1] = -velocity[1], velocity[0]
    return result.contiguous(), mask.contiguous()
