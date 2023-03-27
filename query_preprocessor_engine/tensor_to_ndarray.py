import numpy as np
import torch


def cast_tensor_to_ndarray(x: torch.Tensor, use_gpu: bool = False) -> np.ndarray:
    """
    Casts a PyTorch tensor to a numpy ndarray.

    Args:
    - x (torch.Tensor): The input tensor to cast.
    - use_gpu (bool): A flag indicating whether to use GPU or not.

    Returns:
    - A numpy ndarray with the same shape and data as the input tensor.
    """
    if use_gpu:
        x = x.detach().cpu()
    return x.detach().numpy()
