### utils.py
# Utility functions.
###
import math

import torch
import numpy as np
from typing import Tuple
import torch.nn as nn

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


# class GradientReverseLayer(torch.autograd.Function):
#     def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
#         self.iter_num = iter_num
#         self.alpha = alpha
#         self.low_value = low_value
#         self.high_value = high_value
#         self.max_iter = max_iter

#     @staticmethod
#     def forward(ctx, input):
#         ctx.iter_num += 1
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(self, grad_output):
#         self.coeff = calc_coeff(self.iter_num, self.high_value, self.low_value, self.alpha, self.max_iter)
#         return -self.coeff * grad_output

class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 1000
    @staticmethod
    def forward(ctx, input):
        GradientReverseLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 1
        low = 0.0
        high = 0.1
        iter_num, max_iter = GradientReverseLayer.iter_num, GradientReverseLayer.max_iter 
        coeff = calc_coeff(iter_num, high, low, alpha, max_iter)
        return -coeff * gradOutput


def cnn_compute_positions_and_flatten(
    features,
) -> Tuple:
    """Flatten CNN features to remove spatial dims and return them with correspoding positions."""
    spatial_dims = features.shape[2:]
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # reorder into format (batch_size, flattened_spatial_dims, feature_dim).
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return flattened, positions


def transformer_compute_positions(
    features,
) :
    """Compute positions for Transformer features."""
    n_tokens = features.shape[1]
    image_size = math.sqrt(n_tokens)
    image_size_int = int(image_size)
    assert (
        image_size_int == image_size
    ), "Position computation for Transformers requires square image"

    spatial_dims = (image_size_int, image_size_int)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    return positions


class SoftPositionEmbed(torch.nn.Module):
    """Embeding of positions using convex combination of learnable tensors.

    This assumes that the input positions are between 0 and 1.
    """

    def __init__(
        self, n_spatial_dims: int, feature_dim: int, cnn_channel_order=False, savi_style=False
    ):
        """__init__.

        Args:
            n_spatial_dims (int): Number of spatial dimensions.
            feature_dim (int): Dimensionality of the input features.
            cnn_channel_order (bool): Assume features are in CNN channel order (i.e. C x H x W).
            savi_style (bool): Use savi style positional encoding, where positions are normalized
                between -1 and 1 and a single dense layer is used for embedding.
        """
        super().__init__()
        self.savi_style = savi_style
        n_features = n_spatial_dims if savi_style else 2 * n_spatial_dims
        self.dense = nn.Linear(in_features=n_features, out_features=feature_dim)
        self.cnn_channel_order = cnn_channel_order

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        if self.savi_style:
            # Rescale positional encoding to -1 to 1
            positions = (positions - 0.5) * 2
        else:
            positions = torch.cat([positions, 1 - positions], axis=-1)
        emb_proj = self.dense(positions)
        if self.cnn_channel_order:
            emb_proj = emb_proj.permute(*range(inputs.ndim - 3), -1, -3, -2)
        return inputs + emb_proj
