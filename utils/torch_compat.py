"""PyTorch version/device compatibility helpers."""
import inspect

import torch


def build_adam_optimizer(parameters, lr, weight_decay, **kwargs):
    """
    Adam with foreach/fused disabled when supported — avoids occasional
    Windows + CUDA failures in the multi-tensor Adam kernels
    (e.g. torch._foreach_mul_ + cudaErrorUnknown).
    """
    extra = dict(kwargs)
    try:
        sig = inspect.signature(torch.optim.Adam)
        if 'foreach' in sig.parameters and 'foreach' not in extra:
            extra['foreach'] = False
        if 'fused' in sig.parameters and 'fused' not in extra:
            extra['fused'] = False
    except (TypeError, ValueError):
        pass  # older PyTorch, fall back to defaults
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **extra)
