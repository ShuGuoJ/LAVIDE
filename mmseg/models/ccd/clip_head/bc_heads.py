'''
Binary change subheads for Cross-modal CD.
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.utils import build_from_cfg
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from ...losses import accuracy
import copy

from mmseg.models.cd.fhd import split_batches
from mmseg.models.decode_heads import SegformerHead
from ..bc_heads import BaseHeadBC, ConcatModule, KConcatModule, ContrastiveModule, DWKConcatModule
from ..map_encoders import MAP_ENCODERS
from ...builder import HEADS, build_head
from ....ops import resize
from ..cross_modal.bc_heads import CrossModalMapFormerHead
from ..bc_heads import AlignedContrastiveModule, ScaleContrastiveModule, DistilledModule, MulDistilledModule
from mmseg.models.utils.custom_fun import get_boundary


def get_output(x):
    if len(x) == 4:
        return x
    elif len(x) == 6:
        img_feat, clip_img_feat, mask = x[:-2], x[-2], x[-1]
        return img_feat, clip_img_feat, mask
    elif len(x) == 7:
        img_feat, clip_img_feat, mask, prompt = x[:-3], x[-3], x[-2], x[-1]
        return img_feat, clip_img_feat, mask, prompt
    else:
        raise NotImplementedError


@HEADS.register_module()
class CrossModalLavideHead(CrossModalMapFormerHead):
    '''
    (clip image feat -> distillate -> mit image feat) -> mit_image feat + (clip text feat + map feat) -> n_experts ->
    binary classification -> change detection
    '''
    def __init__(self, route_head, **kwargs):
        clip_proj_dim = kwargs.pop('clip_proj_dim', 512)
        if kwargs.get('in_channels_img', None) is not None:
            kwargs['in_channels_img'] = None
        fusion_channels = kwargs.pop('fusion_channels', 512)
        super().__init__(**kwargs)
        num_inputs = len(self.in_channels)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=fusion_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.temporal_fusion_modules = nn.ModuleList(
            [KConcatModule(
                in_channels=self.in_channels[s] + self.map_encoder.out_channels[s],
                out_channels=self.channels,
                k=self.k + (1 if self.extra_branch else 0),
                norm_cfg=self.norm_cfg
            ) for s in range(num_inputs)]
        )

        self.distilled_module = DistilledModule(
            in_channels_img=self.in_channels[-1],
            proj_channels=clip_proj_dim,
            align_corners=self.align_corners
        )

        attention_weights = []
        for s in range(len(self.in_channels)):
            attn_cfg = {'in_channels': self.in_channels[s] + self.map_encoder.out_channels[s],
                        'out_channels': self.k * self.channels, 'kernel_size': 1, 'norm_cfg': self.norm_cfg
                        }
            attn_cfg.update(route_head)
            attention_weights.append(build_head(attn_cfg))
        self.attention_weights = nn.ModuleList(attention_weights)

    def forward_train(
        self,
        inputs,
        img_metas,
        train_cfg,
        gt_semantic_seg,
        gt_semantic_seg_pre,
        gt_semantic_seg_post=None
    ):
        inputs, mask = inputs[:-1], inputs[-1]
        inputs, clip_img_feat = inputs[:-1], inputs[-1]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre=gt_semantic_seg_pre, mask=mask)
        f2_list = []
        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1_ = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)
            else:
                m1_ = m1

            h = module(features=[f2, m1_])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s]([f2, m1_]) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0],
                self.k,
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)
            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)
            f2_list.append(f2)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        bc_logit = self.cls_seg(out)
        losses = self.losses(seg_logit=bc_logit, seg_label=gt_semantic_seg)

        f2_merged = self.contrastive_img_forward(f2_list)
        contrastive_losses = self.contrastive_module(
            bc=gt_semantic_seg,
            g1=map_features[0],
            f2=f2_merged,
            f1=None
        )
        losses.update(contrastive_losses)

        distilled_losses = self.distilled_module(f2_list[-1], clip_img_feat)
        losses.update(distilled_losses)
        return losses

    def forward(self, inputs, gt_semantic_seg_pre):
        inputs, mask = inputs[:-1], inputs[-1]
        inputs, clip_img_feat = inputs[:-1], inputs[-1]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        map_features = self.map_encoder(gt_semantic_seg_pre=gt_semantic_seg_pre, mask=mask)
        bitemporal_features = []
        for s, module in enumerate(self.temporal_fusion_modules):
            f2 = x[s]
            m1 = map_features[s]
            if m1.shape[2:] != f2.shape[2:]:
                m1 = resize(m1, size=f2.shape[2:], mode='bilinear', align_corners=self.align_corners)

            h = module(features=[f2, m1])

            if self.extra_branch:
                f_extra = h[:,-self.channels:]
                h = h[:,:-self.channels]

            h_k = h.reshape(
                h.shape[0],
                self.k,
                self.channels,
                h.shape[2],
                h.shape[3]
            ) # (B,K,C,H,W)
            attn_weights = self.attention_weights[s]([f2, m1]) # (B,KC, H, W)
            attn_weights = attn_weights.reshape(
                h_k.shape[0],
                self.k,
                h_k.shape[2],
                h_k.shape[3],
                h_k.shape[4]).softmax(dim=1) # (B,K,C,H,W)
            f = (h_k * attn_weights).sum(dim=1)  # (B,C,H,W)
            if self.extra_branch:
                f = f + f_extra
            f = resize(input=f, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            bitemporal_features.append(f)

        out = self.fusion_conv(torch.cat(bitemporal_features, dim=1))
        out = self.cls_seg(out)

        return out

