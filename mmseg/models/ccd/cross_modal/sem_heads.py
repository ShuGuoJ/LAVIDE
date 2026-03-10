'''
Semantic segmentation subheads for Cross-modal CD.
'''
import torch
from ...builder import HEADS
from ..sem_heads import DummySemHead
from ...decode_heads import SegformerHead
from copy import deepcopy
from ...losses import accuracy
from mmseg.ops import resize
from mmcv.runner import auto_fp16, force_fp32
from ...builder import HEADS, build_head

@HEADS.register_module()
class CrossModalSegformerSemHead(SegformerHead):
    '''
    Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        return super(CrossModalSegformerSemHead, self).forward_train(
            inputs=inputs,
            img_metas=img_metas,
            train_cfg=train_cfg,
            gt_semantic_seg=gt_semantic_seg_post
        )

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHead, self).forward_test(
            inputs=inputs, 
            img_metas=img_metas, 
            test_cfg=test_cfg)


@HEADS.register_module()
class CrossModalSegformerSemHeadSem(SegformerHead):
    '''
    Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        losses = super(CrossModalSegformerSemHeadSem, self).forward_train(
            inputs=inputs,
            img_metas=img_metas,
            train_cfg=train_cfg,
            gt_semantic_seg=gt_semantic_seg_post
        )
        sems = inputs[4:7]
        for i, sem in enumerate(sems):
            sem_less = sem[:, :-1]
            ith_losses = self.losses_(seg_logit=sem_less, seg_label=gt_semantic_seg_post)
            ith_losses = {f'{i}th_{k}': v for k, v in ith_losses.items()}
            losses.update(ith_losses)

        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses_(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        multi_losses = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        if isinstance(multi_losses, dict):
            loss.update(multi_losses)
        else:
            loss['loss_seg'] = multi_losses
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHeadSem, self).forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg)


@HEADS.register_module()
class CrossModalSegformerSemHeadSemV2(SegformerHead):
    '''
    Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        losses = super(CrossModalSegformerSemHeadSemV2, self).forward_train(
            inputs=inputs,
            img_metas=img_metas,
            train_cfg=train_cfg,
            gt_semantic_seg=gt_semantic_seg_post
        )
        sems = inputs[4:7]
        sem_gt = gt_semantic_seg_post.clone().detach()
        sem_gt[sem_gt == self.ignore_index] = self.num_classes
        for i, sem in enumerate(sems):
            # sem_gt = gt_semantic_seg_pre.clone()
            ith_losses = self.losses_(seg_logit=sem, seg_label=sem_gt)
            ith_losses = {f'{i}th_{k}': v for k, v in ith_losses.items()}
            losses.update(ith_losses)

        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses_(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        multi_losses = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight)
        if isinstance(multi_losses, dict):
            loss.update(multi_losses)
        else:
            loss['loss_seg'] = multi_losses
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHeadSemV2, self).forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg)

    # def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
    #     sems = inputs[4:7]
    #     return sems[-2][:, :-1]


@HEADS.register_module()
class CrossModalSegformerSemHeadWeight(CrossModalSegformerSemHead):
    '''
    weighted Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        # seg losses on unchanged regions
        unchanged_gt_semantic_seg_post = gt_semantic_seg_post.clone()
        unchanged_gt_semantic_seg_post[gt_semantic_seg==1] = self.ignore_index
        unchanged_seg_losses = self.losses(seg_logits, unchanged_gt_semantic_seg_post)
        # print(unchanged_seg_losses)
        # seg losses on changed regions
        changed_gt_semantic_seg_post = gt_semantic_seg_post.clone()
        changed_gt_semantic_seg_post[gt_semantic_seg==0] = self.ignore_index
        changed_gt_semantic_seg_post[gt_semantic_seg==self.ignore_index] = self.ignore_index
        # gt_semantic_seg_post_ = torch.ones_like(gt_semantic_seg_post, device=gt_semantic_seg_post.device, dtype=gt_semantic_seg_post.dtype)
        # gt_semantic_seg_post_ = gt_semantic_seg_post_ * self.ignore_index
        changed_seg_losses = self.losses(seg_logits, changed_gt_semantic_seg_post)
        # print(changed_seg_losses)
        losses = dict()
        # print(f'unchanged: {torch.sum(gt_semantic_seg==0) + 1e-8}')
        # print(f'changed: {torch.sum(gt_semantic_seg==1) + 1e-8}')
        losses['loss_seg'] = unchanged_seg_losses['loss_seg'] / (torch.sum(unchanged_gt_semantic_seg_post!=self.ignore_index) + 1e-8) + changed_seg_losses['loss_seg'] / (torch.sum(changed_gt_semantic_seg_post!=self.ignore_index) + 1e-8)

        seg_logits_ = resize(
            input=seg_logits,
            size=gt_semantic_seg_post.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = gt_semantic_seg_post.squeeze(1)
        losses['acc_seg'] = accuracy(seg_logits_, seg_label)
        # print(losses)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHead, self).forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg)


@HEADS.register_module()
class CrossModalSegformerSemHeadWeightV2(CrossModalSegformerSemHead):
    '''
    weighted Segformer's semantic segmentation head.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        # seg losses on unchanged regions
        gt_semantic_seg_post_ = gt_semantic_seg_post.clone()
        gt_semantic_seg_post_[gt_semantic_seg==1] = self.ignore_index
        unchanged_seg_losses = self.losses(seg_logits, gt_semantic_seg_post_)
        # print(unchanged_seg_losses)
        # seg losses on changed regions
        gt_semantic_seg_post_ = gt_semantic_seg_post.clone()
        gt_semantic_seg_post_[gt_semantic_seg==0] = self.ignore_index
        # gt_semantic_seg_post_ = torch.ones_like(gt_semantic_seg_post, device=gt_semantic_seg_post.device, dtype=gt_semantic_seg_post.dtype)
        # gt_semantic_seg_post_ = gt_semantic_seg_post_ * self.ignore_index
        changed_seg_losses = self.losses(seg_logits, gt_semantic_seg_post_)
        # print(changed_seg_losses)
        losses = dict()
        # print(f'unchanged: {torch.sum(gt_semantic_seg==0) + 1e-8}')
        # print(f'changed: {torch.sum(gt_semantic_seg==1) + 1e-8}')
        losses['loss_seg'] = unchanged_seg_losses['loss_seg'] + changed_seg_losses['loss_seg']

        seg_logits_ = resize(
            input=seg_logits,
            size=gt_semantic_seg_post.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = gt_semantic_seg_post.squeeze(1)
        losses['acc_seg'] = accuracy(seg_logits_, seg_label)
        # print(losses)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        return super(CrossModalSegformerSemHead, self).forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg)


@HEADS.register_module()
class CrossModalDummySemHead(DummySemHead):
    '''
    Placeholder for semantic segmentation head if one is only interested in BCD (not SCD).
    '''
    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        B, _, H, W = inputs[0].shape
        return torch.rand(B, self.num_classes, H, W, dtype=inputs[0].dtype, device=inputs[0].device)


@HEADS.register_module()
class WeaklySegformerSemHead(CrossModalSegformerSemHead):
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        mask = gt_semantic_seg == 1
        gt_semantic_seg_pre_ = deepcopy(gt_semantic_seg_pre)
        gt_semantic_seg_pre_[mask] = self.ignore_index
        return super(CrossModalSegformerSemHead, self).forward_train(
            inputs=inputs,
            img_metas=img_metas,
            train_cfg=train_cfg,
            gt_semantic_seg=gt_semantic_seg_pre_
        )


@HEADS.register_module()
class WeaklyCrossModalSegformerSemHeadV2(CrossModalSegformerSemHead):
    '''
    using binary change label to supervised semantic probability.
    '''
    def __init__(self, weakly_sem_head, interpolate_mode='bilinear', **kwargs):
        super().__init__(**kwargs)
        num_inputs = len(self.in_channels)
        init_weakly_sem_head = {
            'in_channels': self.channels * num_inputs,
            'channels': self.channels,
            'norm_cfg': self.norm_cfg
        }
        init_weakly_sem_head.update(weakly_sem_head)
        self.weakly_sem_head = build_head(init_weakly_sem_head)


    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        weakly_losses = self.weakly_sem_head.forward_train(seg_logits,
                                                           gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                           gt_semantic_seg=gt_semantic_seg)

        return weakly_losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
         seg_logits = super(CrossModalSegformerSemHead, self).forward_test(
            inputs=inputs,
            img_metas=img_metas,
            test_cfg=test_cfg)

         seg_logits = self.weakly_sem_head(seg_logits)
         return seg_logits


@HEADS.register_module()
class WeaklyPlusSLCrossModalSegformerSemHeadV2(WeaklyCrossModalSegformerSemHeadV2):
    '''
    using hybrid semantic and binary change labels to supervised semantic probability.
    '''

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        gt_semantic_seg_pre_first, gt_semantic_seg_pre_last = torch.chunk(gt_semantic_seg_pre, 2)
        gt_semantic_seg_post_first, gt_semantic_seg_post_last = torch.chunk(gt_semantic_seg_post, 2)
        gt_semantic_seg_first, gt_semantic_seg_last = torch.chunk(gt_semantic_seg, 2)
        seg_logits_post_first, seg_logits_post_last = torch.chunk(seg_logits, 2)

        weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post_first,
                                                           gt_semantic_seg_pre=gt_semantic_seg_pre_first,
                                                           gt_semantic_seg=gt_semantic_seg_first)

        # the last half used for full supervision
        fully_losses = self.losses(seg_logit=seg_logits_post_last, seg_label=gt_semantic_seg_post_last)

        losses = fully_losses
        losses.update(weakly_losses)

        return losses


@HEADS.register_module()
class WeaklyPlusSLCrossModalSegformerSemHeadV2Weight(WeaklyCrossModalSegformerSemHeadV2):
    '''
    using hybrid semantic and binary change labels to supervised semantic probability.
    '''

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        gt_semantic_seg_pre_first, gt_semantic_seg_pre_last = torch.chunk(gt_semantic_seg_pre, 2)
        gt_semantic_seg_post_first, gt_semantic_seg_post_last = torch.chunk(gt_semantic_seg_post, 2)
        gt_semantic_seg_first, gt_semantic_seg_last = torch.chunk(gt_semantic_seg, 2)
        seg_logits_post_first, seg_logits_post_last = torch.chunk(seg_logits, 2)

        weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post_first,
                                                           gt_semantic_seg_pre=gt_semantic_seg_pre_first,
                                                           gt_semantic_seg=gt_semantic_seg_first)

        # seg losses on unchanged regions
        unchanged_gt_semantic_seg_post_last = gt_semantic_seg_post_last.clone()
        unchanged_gt_semantic_seg_post_last[gt_semantic_seg_last == 1] = self.ignore_index
        unchanged_seg_losses_last = self.losses(seg_logits_post_last, unchanged_gt_semantic_seg_post_last)
        # print(unchanged_seg_losses)
        # seg losses on changed regions
        changed_gt_semantic_seg_post_last = gt_semantic_seg_post_last.clone()
        changed_gt_semantic_seg_post_last[gt_semantic_seg_last == 0] = self.ignore_index
        changed_gt_semantic_seg_post_last[gt_semantic_seg_last == self.ignore_index] = self.ignore_index
        # gt_semantic_seg_post_ = torch.ones_like(gt_semantic_seg_post, device=gt_semantic_seg_post.device, dtype=gt_semantic_seg_post.dtype)
        # gt_semantic_seg_post_ = gt_semantic_seg_post_ * self.ignore_index
        changed_seg_losses_last = self.losses(seg_logits_post_last, changed_gt_semantic_seg_post_last)
        # print(changed_seg_losses)
        losses = dict()
        # print(f'unchanged: {torch.sum(gt_semantic_seg==0) + 1e-8}')
        # print(f'changed: {torch.sum(gt_semantic_seg==1) + 1e-8}')
        losses['loss_seg'] = unchanged_seg_losses_last['loss_seg'] / (
                    torch.sum(unchanged_gt_semantic_seg_post_last != self.ignore_index) + 1e-8) + changed_seg_losses_last[
                                 'loss_seg'] / (torch.sum(changed_gt_semantic_seg_post_last != self.ignore_index) + 1e-8)

        seg_logits_post_last_ = resize(
            input=seg_logits_post_last,
            size=gt_semantic_seg_post_last.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = gt_semantic_seg_post_last.squeeze(1)
        losses['acc_seg'] = accuracy(seg_logits_post_last_, seg_label)

        losses.update(weakly_losses)

        return losses