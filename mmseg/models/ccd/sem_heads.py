'''
Subheads for semantic segmentation.
'''
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.ops import resize
from mmcv.runner import force_fp32
from ..losses import accuracy
from ..cd.fhd import split_batches
from ..decode_heads import SegformerHead, SETRUPHead, UPerHead, FCNHead, LRHead
# from ..decode_heads import SETRUPHead
# from ..decode_heads import UPerHead
from ..builder import HEADS, build_head
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class SegformerSemHead(SegformerHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class SegformerSemHeadOnlyPost(SegformerSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class SegformerSemHeadOnlyPostSem(SegformerSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        # t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        # t1_2, t2_2 = split_batches(s2)
        # t1_3, t2_3 = split_batches(s3)
        # t1_4, t2_4 = split_batches(s4)
        t2_1, t2_2, t2_3, t2_4 = s1, s2, s3, s4
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        # t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        # t1_2, t2_2 = split_batches(s2)
        # t1_3, t2_3 = split_batches(s3)
        # t1_4, t2_4 = split_batches(s4)
        t2_1, t2_2, t2_3, t2_4 = s1, s2, s3, s4
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class SegformerSemHeadOnlyPostSemWeight(SegformerSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        # t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        # t1_2, t2_2 = split_batches(s2)
        # t1_3, t2_3 = split_batches(s3)
        # t1_4, t2_4 = split_batches(s4)
        t2_1, t2_2, t2_3, t2_4 = s1, s2, s3, s4
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        seg_logits = seg_logits_post

        # seg losses on unchanged regions
        unchanged_gt_semantic_seg_post = gt_semantic_seg_post.clone()
        unchanged_gt_semantic_seg_post[gt_semantic_seg == 1] = self.ignore_index
        unchanged_seg_losses = self.losses(seg_logits, unchanged_gt_semantic_seg_post)
        # print(unchanged_seg_losses)
        # seg losses on changed regions
        changed_gt_semantic_seg_post = gt_semantic_seg_post.clone()
        changed_gt_semantic_seg_post[gt_semantic_seg == 0] = self.ignore_index
        changed_gt_semantic_seg_post[gt_semantic_seg == self.ignore_index] = self.ignore_index
        # gt_semantic_seg_post_ = torch.ones_like(gt_semantic_seg_post, device=gt_semantic_seg_post.device, dtype=gt_semantic_seg_post.dtype)
        # gt_semantic_seg_post_ = gt_semantic_seg_post_ * self.ignore_index
        changed_seg_losses = self.losses(seg_logits, changed_gt_semantic_seg_post)
        # print(changed_seg_losses)
        losses = dict()
        # print(f'unchanged: {torch.sum(gt_semantic_seg==0) + 1e-8}')
        # print(f'changed: {torch.sum(gt_semantic_seg==1) + 1e-8}')
        losses['loss_seg'] = unchanged_seg_losses['loss_seg'] / (
                    torch.sum(unchanged_gt_semantic_seg_post != self.ignore_index) + 1e-8) + changed_seg_losses[
                                 'loss_seg'] / (torch.sum(changed_gt_semantic_seg_post != self.ignore_index) + 1e-8)

        seg_logits_ = resize(
            input=seg_logits,
            size=gt_semantic_seg_post.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = gt_semantic_seg_post.squeeze(1)
        losses['acc_seg'] = accuracy(seg_logits_, seg_label)

        # losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)
        #
        # losses = dict(
        #     loss_seg = losses_post['loss_seg'],
        #     acc_seg = losses_post['acc_seg']
        # )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        # t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        # t1_2, t2_2 = split_batches(s2)
        # t1_3, t2_3 = split_batches(s3)
        # t1_4, t2_4 = split_batches(s4)
        t2_1, t2_2, t2_3, t2_4 = s1, s2, s3, s4
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class SegformerSemHeadCross(SegformerSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_pre)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class DummySemHead(nn.Module):
    '''
    Placeholder for semantic segmentation head if one is only interested in BCD (not SCD).
    '''
    def __init__(self, num_classes, align_corners, **kwargs):
        super(DummySemHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners

    def init_weights(self, *args, **kwargs):
        pass
    
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        losses = dict(
            loss_seg = torch.tensor(0., dtype=inputs[0].dtype, device=inputs[0].device),
            acc_seg = torch.tensor(0., dtype=inputs[0].dtype, device=inputs[0].device)
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        Bx2, _, H, W = inputs[0].shape
        return torch.rand(Bx2 // 2, self.num_classes, H, W, dtype=inputs[0].dtype, device=inputs[0].device)


@HEADS.register_module()
class SimpleSemHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        pass

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        seg_logits_pre, seg_logits_post = split_batches(inputs)
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg=0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg=0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        seg_logits_pre, seg_logits_post = split_batches(inputs)

        return seg_logits_post


@HEADS.register_module()
class SimpleSemHeadOnlyPost(SimpleSemHead):
    '''
    Only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        seg_logits_pre, seg_logits_post = split_batches(inputs)
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg= losses_post['loss_seg'],
            acc_seg= losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class SimpleSemHeadV2(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        pass

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
            losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

            losses = dict(
                loss_seg=0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
                acc_seg=0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
            )
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

            losses = dict(
                loss_seg=losses_post['loss_seg'],
                acc_seg=losses_post['acc_seg']
            )
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        # seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError

        return seg_logits_post


@HEADS.register_module()
class SimpleSemHeadV2D(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        pass

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 4:
            seg_logits_pre, seg_logits_post = inputs[2], inputs[3]
            losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

            losses = dict(
                loss_seg=0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
                acc_seg=0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
            )
        elif len(inputs) == 3:
            seg_logits_post = inputs[2]

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

            losses = dict(
                loss_seg=losses_post['loss_seg'],
                acc_seg=losses_post['acc_seg']
            )
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        # seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
        if len(inputs) == 4:
            seg_logits_pre, seg_logits_post = inputs[2], inputs[3]
        elif len(inputs) == 3:
            seg_logits_post = inputs[2]
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError

        return seg_logits_post


@HEADS.register_module()
class SimpleSemHeadV2Cross(SimpleSemHeadV2):
    '''
    Supervising semantic segmentation of images with maps
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
            losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_pre)

            losses = dict(
                loss_seg=0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
                acc_seg=0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
            )
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]

            losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_pre)

            losses = dict(
                loss_seg=losses_post['loss_seg'],
                acc_seg=losses_post['acc_seg']
            )
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        return losses

    def forward_test(self, **kwargs):
        seg_logits_post = super().forward_test(**kwargs)
        seg_logits_post = seg_logits_post[:, :-1, ...]
        return seg_logits_post


@HEADS.register_module()
class SimpleSemHeadV2WS(BaseDecodeHead):
    '''
    using binary labels to supervise semantic segmentation
    '''
    def __init__(self, weakly_sem_head, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        num_inputs = len(self.in_channels)
        init_weakly_sem_head = {
            'in_channels': self.channels * num_inputs,
            'channels': self.channels,
            'norm_cfg': self.norm_cfg
        }
        init_weakly_sem_head.update(weakly_sem_head)
        self.weakly_sem_head = build_head(init_weakly_sem_head)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        pass

    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]

            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=gt_semantic_seg)

            return weakly_losses
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]

            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=gt_semantic_seg)
            return weakly_losses
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        # return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        # seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        seg_logits_post = self.weakly_sem_head(seg_logits_post)

        return seg_logits_post

    def forward_dummy(self, img):
        """Used for computing network flops. See
        `tools/analysis/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.weakly_sem_head(x)
        return outs


@HEADS.register_module()
class SimpleSemHeadV2WSPlusSL(SimpleSemHeadV2WS):
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        gt_semantic_seg_pre_first, gt_semantic_seg_pre_last = torch.chunk(gt_semantic_seg_pre, 2)
        gt_semantic_seg_post_first, gt_semantic_seg_post_last = torch.chunk(gt_semantic_seg_post, 2)
        gt_semantic_seg_first, gt_semantic_seg_last = torch.chunk(gt_semantic_seg, 2)
        if len(inputs) == 3:
            seg_logits_pre, seg_logits_post = inputs[1], inputs[2]

            seg_logits_post_first, seg_logits_post_last = torch.chunk(seg_logits_post, 2)

            # the first half used for weakly supervision
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post_first,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre_first,
                                                               gt_semantic_seg=gt_semantic_seg_first)

            # the last half used for full supervision
            fully_losses = self.losses(seg_logit=seg_logits_post_last, seg_label=gt_semantic_seg_post_last)

            losses = fully_losses
            losses.update(weakly_losses)

            return losses
        elif len(inputs) == 2:
            seg_logits_post = inputs[1]

            seg_logits_post_first, seg_logits_post_last = torch.chunk(seg_logits_post, 2)

            # the first half used for weakly supervision
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post_first,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre_first,
                                                               gt_semantic_seg=gt_semantic_seg_first)

            # the last half used for full supervision
            fully_losses = self.losses(seg_logit=seg_logits_post_last, seg_label=gt_semantic_seg_post_last)

            losses = fully_losses
            losses.update(weakly_losses)

            return losses
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        # return losses


@HEADS.register_module()
class SimpleSemHeadV2US(SimpleSemHeadV2WS):
    '''
    using binary labels to supervise semantic segmentation
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 3:
            bc_logit, seg_logits_pre, seg_logits_post = inputs
            pesudo_bc = bc_logit.max(1, keepdim=True)[1]
            pesudo_bc = pesudo_bc.detach()
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=pesudo_bc)

            return weakly_losses
        elif len(inputs) == 2:
            bc_logit, seg_logits_post = inputs
            pesudo_bc = bc_logit.max(1, keepdim=True)[1]
            pesudo_bc = pesudo_bc.detach()
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=pesudo_bc)
            return weakly_losses
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError
        # return losses


@HEADS.register_module()
class SimpleSemHeadV2USPlusMapPrior(SimpleSemHeadV2WS):
    '''
    using binary labels and gt_semantic_seg_pre to supervise semantic segmentation
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post,
                      gt_semantic_seg=None):
        # assert len(inputs) == 3, f'Expect 3 inputs, got {len(inputs)}'
        if len(inputs) == 3:
            bc_logit, seg_logits_pre, seg_logits_post = inputs
            pesudo_bc = bc_logit.max(1, keepdim=True)[1]
            pesudo_bc = pesudo_bc.detach()
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=pesudo_bc)

            losses = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_pre)
            losses.update(weakly_losses)

            return losses
        elif len(inputs) == 2:
            bc_logit, seg_logits_post = inputs
            pesudo_bc = bc_logit.max(1, keepdim=True)[1]
            pesudo_bc = pesudo_bc.detach()
            weakly_losses = self.weakly_sem_head.forward_train(seg_logits_post,
                                                               gt_semantic_seg_pre=gt_semantic_seg_pre,
                                                               gt_semantic_seg=pesudo_bc)
            losses = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_pre)
            losses.update(weakly_losses)

            return losses
        else:
            print(f'Expect more than 2 inputs, got {len(inputs)}')
            raise NotImplementedError


@HEADS.register_module()
class SimpleSemHeadV2USNoEntropy(SimpleSemHeadV2US):
    '''
    using binary labels to supervise semantic segmentation, removing entropy loss
    '''
    def forward_train(self, *args, **kwargs):
        losses = super().forward_train(*args, **kwargs)
        if 'entropy_loss' in losses:
            del losses['entropy_loss']
        return losses


@HEADS.register_module()
class SimpleSemHeadV2USPlusMapPriorNoEntropy(SimpleSemHeadV2USPlusMapPrior):
    '''
    using binary labels to supervise semantic segmentation, removing entropy loss
    '''
    def forward_train(self, *args, **kwargs):
        losses = super().forward_train(*args, **kwargs)
        if 'entropy_loss' in losses:
            del losses['entropy_loss']
        return losses


@HEADS.register_module()
class SETRUPSemHead(SETRUPHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        x = inputs  # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class SETRUPSemHeadOnlyPost(SETRUPSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class UPerSemHead(UPerHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        x = inputs  # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class UPerSemHeadOnlyPost(UPerSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class FCNSemHead(FCNHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        x = inputs  # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post


@HEADS.register_module()
class FCNSemHeadOnlyPost(FCNSemHead):
    '''
    SegFormer's head for semantic segmentation.
    only supervising post-change images
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        # x = self._transform_inputs(inputs)  # mutliscale feature
        x = inputs # multiscale feature
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        # seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        # losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = losses_post['loss_seg'],
            acc_seg = losses_post['acc_seg']
        )
        return losses


@HEADS.register_module()
class LrformerSemHead(LRHead):
    '''
    SegFormer's head for semantic segmentation.
    '''
    def forward_train(self, inputs, img_metas, train_cfg, gt_semantic_seg_pre, gt_semantic_seg_post, gt_semantic_seg=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_pre = self.forward([t1_1, t1_2, t1_3, t1_4])
        losses_pre = self.losses(seg_logit=seg_logits_pre, seg_label=gt_semantic_seg_pre)

        seg_logits_post = self.forward([t2_1,t2_2,t2_3,t2_4])
        losses_post = self.losses(seg_logit=seg_logits_post, seg_label=gt_semantic_seg_post)

        losses = dict(
            loss_seg = 0.5 * (losses_pre['loss_seg'] + losses_post['loss_seg']),
            acc_seg = 0.5 * (losses_pre['acc_seg'] + losses_post['acc_seg'])
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg_pre):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32; len=3, 1/4,1/8,1/16
        s1, s2, s3, s4 = x # multiscale features
        t1_1, t2_1 = split_batches(s1) # features at same scale but different points in time
        t1_2, t2_2 = split_batches(s2)
        t1_3, t2_3 = split_batches(s3)
        t1_4, t2_4 = split_batches(s4)
        seg_logits_post = self.forward([t2_1, t2_2, t2_3, t2_4])

        return seg_logits_post