# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MAEPretrainHead(BaseModule):
    """Head for MAE Pre-training.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 img_width=2650, 
                 img_height=1024,
                 patch_size: int = 16,
                 patch_width: int = 25,
                 patch_height: int = 16) -> None:
                 
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.img_height = img_height
        self.img_width  = img_width
        self.loss_module = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, 3, H, W)`.

        Returns:
            torch.Tensor: Patchified images. The shape is
            :math:`(B, L, \text{patch_height} \times \text{patch_width} \times 3)`.
        """
        ph = self.patch_height
        pw = self.patch_width
        assert imgs.shape[2] % ph == 0 and imgs.shape[3] % pw == 0

        h = imgs.shape[2] // ph
        w = imgs.shape[3] // pw
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_height} \times \text{patch_width} \times 3)`.

        Returns:
            torch.Tensor: The shape is :math:`(B, 3, H, W)`.
        """
        ph = self.patch_height
        pw = self.patch_width
        h = self.img_height // ph
        w = self.img_width // pw
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * ph, w * pw))
        return imgs
    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.

        Args:
            target (torch.Tensor): Image with the shape of B x 3 x H x W

        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target
    
    def get_norm_pix_mean_var(self, target):
        target = self.patchify(target)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        return mean, var

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        target = self.construct_target(target)

        loss = self.loss_module(pred, target, mask)

        return loss
