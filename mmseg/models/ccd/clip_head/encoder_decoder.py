from mmcv.runner import auto_fp16
from ..encoder_decoder import EncoderDecoderCCD
from ...cd.fhd import split_images
from ...builder import SEGMENTORS

from mmcv.runner import auto_fp16
from ....core import add_prefix
from ....ops import resize
from ..encoder_decoder import merge_tiles, split_into_tiles
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import cv2

@SEGMENTORS.register_module()
class EncoderDecoderClip(EncoderDecoderCCD):
    '''
    Overall model class for Cross-modal CD.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prompt_embed = self.backbone.clip_encoder.prompt_embed
        self.decode_head.bc_head.prompt_embed = nn.Parameter(prompt_embed, requires_grad=False) \
            if not isinstance(prompt_embed, nn.Parameter) else prompt_embed.clone()
        # from torch import nn
        # print(isinstance(self.decode_head.bc_head.prompt_embed, nn.Parameter))
        # print(isinstance(self.backbone.clip_encoder.prompt_embed, nn.Parameter))

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, gt_semantic_seg_pre=None, gt_semantic_seg_post=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        
        if return_loss:
            if img.shape[1] % 2 == 0:
                img1, img2 = split_images(img)
            else:
                img2 = img
            return self.forward_train(
                img=img2,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg_post=gt_semantic_seg_post,
                **kwargs
                )
        else:
            assert isinstance(img, list) and len(img) == 1, 'Expected a one item list!'
            if img[0].shape[1] % 2 == 0:
                img1, img2 = split_images(img[0])
            else:
                img2 = img[0]
            return self.forward_test(
                imgs=[img2],
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
            )

    '''
        Overall model class for Cross-modal CD.
        '''

    def extract_feat(self, img, img_metas, seg):
        """Extract features from images."""
        x = self.backbone(img=img, img_metas=img_metas, mask=seg)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_semantic_seg, gt_semantic_seg_pre=None, gt_semantic_seg_post=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            gt_semantic_seg_pre (Tensor): Segmentation mask for t1.
            gt_semantic_seg_post (Tensor): Segmentation mask for t2.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img=img, img_metas=img_metas, seg=gt_semantic_seg_pre)
        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x=x,
            img_metas=img_metas,
            gt_semantic_seg=gt_semantic_seg,
            gt_semantic_seg_pre=gt_semantic_seg_pre,
            gt_semantic_seg_post=gt_semantic_seg_post,
        )
        losses.update(loss_decode)

        return losses

    def encode_decode(self, img, img_metas, gt_semantic_seg_pre=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if self.tile_inference:
            img, n_h, n_w = split_into_tiles(img, self.inference_tile_size)
            gt_semantic_seg_pre, _, _ = split_into_tiles(gt_semantic_seg_pre, self.inference_tile_size)

        x = self.extract_feat(img=img, img_metas=img_metas, seg=gt_semantic_seg_pre)
        output = self._decode_head_forward_test(
            x=x,
            img_metas=img_metas,
            gt_semantic_seg_pre=gt_semantic_seg_pre
        )
        if self.tile_inference:
            output['bc'] = merge_tiles(output['bc'], n_h, n_w)
            output['sem'] = merge_tiles(output['sem'], n_h, n_w)
        output['bc'] = resize(
            input=output['bc'],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        output['sem'] = resize(
            input=output['sem'],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )
        return output


