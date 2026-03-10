from mmseg.models.builder import BACKBONES
from torch import nn
from .. import builder
from mmseg.utils import split_images
from torch import Tensor
import torch


@BACKBONES.register_module()
class DualEncoder(nn.Module):
    def __init__(self, clip_encoder, img_encoder, pretrained=None):
        super().__init__()
        self.clip_encoder = builder.build_backbone(clip_encoder)
        if pretrained is not None:
            img_encoder['pretrained'] = pretrained
        self.img_encoder = builder.build_backbone(img_encoder)

    def forward(self, img, mask, img_metas):
        img_feat = self.img_encoder(img)
        clip_feat = self.clip_encoder(img=img, mask=mask, img_metas=img_metas)
        out = img_feat + clip_feat
        return out

    def init_weights(self, pretrained=None):
        self.clip_encoder.init_weights()
        self.img_encoder.init_weights(pretrained)



def merge_batches(x1: Tensor, x2: Tensor):
    """ merge two batches each contains B images into a 2*B batch of images
    in order to adapt to MMSegmentation """

    assert x1.ndim == 4 and x2.ndim == 4, f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)