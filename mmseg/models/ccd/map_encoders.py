'''
Map encoders.
'''
import torch
from torch import nn
from mmcv.cnn import ConvModule
from ...ops import resize
from torch.nn import functional as F
import timm
from .. import builder

from mmcv.utils import Registry

MAP_ENCODERS = Registry('map_encoder')


@MAP_ENCODERS.register_module()
class BasicMapEncoder(nn.Module):
    '''
    Three-layer CNN for encoding map info.
    '''
    def __init__(
        self, 
        n_semantic_classes, 
        ignore_index, 
        out_channels, 
        scale, 
        num_scales, 
        norm_cfg
    ):
        super(BasicMapEncoder, self).__init__()
        self.n_semantic_classes = n_semantic_classes
        self.ignore_index = ignore_index
        self.num_scales = num_scales
        self.scale = [scale for _ in range(self.num_scales)]
        self.out_channels = [out_channels for _ in range(self.num_scales)]

        self.layer1 = nn.Conv2d(
            in_channels=n_semantic_classes + 1,  # one extra for ignore index
            out_channels=self.out_channels[0],
            kernel_size=1,
        )
        self.layer2 = ConvModule(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg
        )
        self.layer3 = ConvModule(
            in_channels=self.out_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg
        )

    def init_weights(self):
        pass

    def forward(self, gt_semantic_seg_pre, **kwargs):
        if gt_semantic_seg_pre.ndim == 4:
            gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)
        B, H, W = gt_semantic_seg_pre.shape
        one_hot_channels = self.n_semantic_classes + 1
        # last index for ignore
        if self.ignore_index > self.n_semantic_classes:
            _gt_semantic_seg_pre = torch.clone(gt_semantic_seg_pre)
            _gt_semantic_seg_pre[gt_semantic_seg_pre == self.ignore_index] = self.n_semantic_classes
        else:
            _gt_semantic_seg_pre = gt_semantic_seg_pre
        with torch.no_grad():
            one_hot = nn.functional.one_hot(
                _gt_semantic_seg_pre.long(), num_classes=one_hot_channels)
            one_hot = one_hot.permute(0, 3, 1, 2).reshape(
                B, one_hot_channels, H, W).float()
            one_hot = one_hot.contiguous()
            one_hot = resize(one_hot, scale_factor=self.scale[0],
                             mode='bilinear', align_corners=False)

        x = self.layer1(one_hot)
        x = self.layer2(x)
        x = self.layer3(x)
        return [x for _ in range(self.num_scales)]


@MAP_ENCODERS.register_module()
class ConcatClipMapEncoder(BasicMapEncoder):
    '''
    (n_binary_map_feat|prompt_feat) -> mixer -> map_feat
    '''
    def __init__(self, **kwargs):
        prompt_embed = kwargs.pop('prompt_embed', None)
        out_channels = kwargs['out_channels']
        norm_cfg = kwargs['norm_cfg']
        act_cfg = kwargs.pop('act_cfg', dict(type='ReLU'))
        super().__init__(**kwargs)
        in_channels = prompt_embed + out_channels
        self.mixer = ConcatMapMixer(in_channels=in_channels, out_channels=out_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, gt_semantic_seg_pre, mask):
        map_embed = super().forward(gt_semantic_seg_pre=gt_semantic_seg_pre)
        map_embed = map_embed[0]
        map_shape, mask_shape = map_embed.shape[-2:], mask.shape[-2:]
        if map_shape != mask_shape:
            mask = resize(mask, size=map_shape, mode='bilinear', align_corners=False)
        x = self.mixer(m1=map_embed, m2=mask)
        return [x for _ in range(self.num_scales)]


class ConcatMapMixer(nn.Module):
    '''
    Concat(m1, m2) -> mixer ->out
    '''
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg=None):
        super().__init__()

        self.layer1 = nn.Conv2d(
            in_channels=in_channels,  # one extra for ignore index
            out_channels=out_channels,
            kernel_size=1,
        )
        self.layer2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.layer3 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            dilation=2,
            padding=4,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def init_weights(self):
        pass

    def forward(self, m1, m2):
        m = torch.cat([m1, m2], dim=1)
        x1 = self.layer1(m)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        o = x3 + x1
        return o

