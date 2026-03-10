'''
Subheads for binary change detection at instance level.
'''
import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmcv.cnn import ConvModule
import copy

from mmseg.models.cd.fhd import split_batches
from mmseg.models.decode_heads import SegformerHead
from .map_encoders import MAP_ENCODERS
from ..builder import HEADS
from ..decode_heads.decode_head import BaseDecodeHead
from ..cd.fhd import MLP, FHD_Module
from ...ops import resize
from ...utils import cos_similarity
# for inst loss
from scipy.ndimage import label as sci_label
# from .clip_head.bc_heads import CrossModalMapFormerHeadPlusPlusClipDistAttn
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

@HEADS.register_module()
class InstHeadBC(nn.Module):
    '''
    Base class for binary change subheads.
    '''
    def __init__(self, channels, n_semantic_classes=1, ignore_index=255, loss_decode=dict(type='CrossEntropyLoss', ignore_index=255)):
        super().__init__()
        self.cls_seg = nn.Linear(channels, 3)
        self.n_semantic_classes = n_semantic_classes
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index

    def forward_train(
        self,
        inputs,
        gt_semantic_seg,
        gt_semantic_seg_pre=None
    ):
        # inputs: [B, C, H, W], gt_semantic_seg: [B, H, W], gt_semantic_seg_pre: [B, H, W]
        if isinstance(inputs, list):
            feat = inputs[0]
        else:
            feat = inputs
        B, _, H, W = feat.shape

        if gt_semantic_seg.ndim == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)
        if gt_semantic_seg.shape[2:] != (H, W):
            gt_semantic_seg = gt_semantic_seg.float()
            gt_semantic_seg = nn.functional.interpolate(
                gt_semantic_seg, size=(H, W), mode='nearest')
            gt_semantic_seg = gt_semantic_seg.long()
        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        if gt_semantic_seg_pre.ndim == 3:
            gt_semantic_seg_pre = gt_semantic_seg_pre.unsqueeze(1)
        if gt_semantic_seg_pre.shape[2:] != (H, W):
            gt_semantic_seg_pre = gt_semantic_seg_pre.float()
            gt_semantic_seg_pre = nn.functional.interpolate(
                gt_semantic_seg_pre, size=(H, W), mode='nearest')
            gt_semantic_seg_pre = gt_semantic_seg_pre.long()
        gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)

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
                B, one_hot_channels, H, W)
            one_hot = one_hot.contiguous()
            # 将one_hot沿着batch的维度拆分成B个C*H*W的tensor并用list存储
            one_hot_list = torch.split(one_hot, 1, dim=0)
            one_hot_list_np = [x.cpu().numpy()[0] for x in one_hot_list]
            inst_list_np = [sci_label(x)[0] for x in one_hot_list_np]
            inst_list = [torch.from_numpy(x).to(one_hot.device) for x in inst_list_np]
            inst_map = torch.stack(inst_list, dim=0).sum(dim=1)
            max_inst_index = inst_map.max()
            print(max_inst_index)
            inst_map = inst_map - 1
            inst_one_hot_map = nn.functional.one_hot(
                inst_map.long(), num_classes=max_inst_index)
            inst_one_hot_map = inst_one_hot_map.permute(0, 3, 1, 2).reshape(
                B, max_inst_index, H * W)
            inst_one_hot_map = inst_one_hot_map.contiguous().to(feat.dtype)  # [B, N, H*W]
            _gt_semantic_seg = gt_semantic_seg.reshape(B, H * W, 1)
            inst_cd_sum = torch.bmm(inst_one_hot_map, (_gt_semantic_seg == 1).to(feat.dtype))
            inst_cd_ratio = inst_cd_sum / (inst_one_hot_map.sum(dim=-1, keepdim=True) + 1e-8)
            inst_cd_gt = torch.ones((B, max_inst_index, 1), dtype=gt_semantic_seg.dtype, device=gt_semantic_seg.device)
            inst_cd_gt[inst_cd_ratio > 0.98] = 2
            inst_cd_gt[inst_cd_ratio < 0.02] = 0
            invalid_mask = inst_one_hot_map.sum(dim=-1) == 0
            ignore_sum = torch.bmm(inst_one_hot_map, (_gt_semantic_seg == self.ignore_index).to(feat.dtype))
            ignore_ratio = ignore_sum / (inst_one_hot_map.sum(dim=-1, keepdim=True) + 1e-8)
            ignore_mask = ignore_ratio > 0.98
            inst_cd_gt[invalid_mask] = self.ignore_index
            inst_cd_gt[ignore_mask] = self.ignore_index

        feat = feat.reshape(B, -1, H * W)
        inst_feat = torch.bmm(inst_one_hot_map, feat.permute(0, 2, 1))  # [B, N, C]
        inst_feat = inst_feat / (inst_one_hot_map.sum(dim=-1, keepdim=True) + 1e-8)
        inst_bc_logit = self.cls_seg(inst_feat)
        inst_bc_logit = inst_bc_logit.permute(0, 2, 1).contiguous()
        inst_cd_gt = inst_cd_gt[..., 0]
        losses = self.losses(seg_logit=inst_bc_logit, seg_label=inst_cd_gt)
        losses = {f'inst_{k}': v for k, v in losses.items()}
        # return losses
        return losses

    def forward_test(
        self,
        inputs,
        gt_semantic_seg_pre
    ):
        # inputs: [B, C, H, W], gt_semantic_seg: [B, H, W], gt_semantic_seg_pre: [B, H, W]
        if isinstance(inputs, list):
            feat = inputs[0]
        else:
            feat = inputs
        B, _, H, W = feat.shape

        if gt_semantic_seg_pre.ndim == 3:
            gt_semantic_seg_pre = gt_semantic_seg_pre.unsqueeze(1)
        if gt_semantic_seg_pre.shape[2:] != (H, W):
            gt_semantic_seg_pre = gt_semantic_seg_pre.float()
            gt_semantic_seg_pre = nn.functional.interpolate(
                gt_semantic_seg_pre, size=(H, W), mode='nearest')
            gt_semantic_seg_pre = gt_semantic_seg_pre.long()
        gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)

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
                B, one_hot_channels, H, W)
            one_hot = one_hot.contiguous()
            # 将one_hot沿着batch的维度拆分成B个C*H*W的tensor并用list存储
            one_hot_list = torch.split(one_hot, 1, dim=0)
            one_hot_list_np = [x.cpu().numpy()[0] for x in one_hot_list]
            inst_list_np = [sci_label(x)[0] for x in one_hot_list_np]
            inst_list = [torch.from_numpy(x).to(one_hot.device) for x in inst_list_np]
            inst_map = torch.stack(inst_list, dim=0).sum(dim=1)
            max_inst_index = inst_map.max()
            inst_map = inst_map - 1
            inst_one_hot_map = nn.functional.one_hot(
                inst_map.long(), num_classes=max_inst_index)
            inst_one_hot_map = inst_one_hot_map.permute(0, 3, 1, 2).reshape(
                B, max_inst_index, H * W)
            inst_one_hot_map = inst_one_hot_map.contiguous().to(feat.dtype)  # [B, N, H*W]

        feat = feat.reshape(B, -1, H * W)
        inst_feat = torch.bmm(inst_one_hot_map, feat.permute(0, 2, 1))  # [B, N, C]
        inst_feat = inst_feat / (inst_one_hot_map.sum(dim=-1, keepdim=True) + 1e-8)
        inst_bc_logit = self.cls_seg(inst_feat) # [B, N, 3]
        inst_bc_map = torch.bmm(inst_one_hot_map.permute(0, 2, 1), inst_bc_logit)
        inst_bc_map = inst_bc_map.permute(0, 2, 1).contiguous().reshape(B, 3, H, W)
        return inst_bc_map


    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss