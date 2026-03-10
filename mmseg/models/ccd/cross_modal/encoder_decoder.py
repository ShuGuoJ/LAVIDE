from mmcv.runner import auto_fp16
from ..encoder_decoder import EncoderDecoderCCD
from ...cd.fhd import split_images
from ...builder import SEGMENTORS
import torch
from ....ops import resize

@SEGMENTORS.register_module()
class EncoderDecoderCMCD(EncoderDecoderCCD):
    '''
    Overall model class for Cross-modal CD.
    '''
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
            img1, img2 = split_images(img)
            return self.forward_train(
                img=img2,
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                gt_semantic_seg_post=gt_semantic_seg_post,
                **kwargs
                )
        else:
            assert isinstance(img, list) and len(img) == 1, 'Expected a one item list!'
            img1, img2 = split_images(img[0])
            return self.forward_test(
                imgs=[img2],
                img_metas=img_metas,
                gt_semantic_seg_pre=gt_semantic_seg_pre,
                **kwargs
            )


@SEGMENTORS.register_module()
class EncoderDecoderCMCD2(EncoderDecoderCMCD):
    def forward_train(self, img, img_metas, gt_semantic_seg, gt_semantic_seg_pre=None, gt_semantic_seg_post=None):
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

        x = self.extract_feat(img, gt_semantic_seg_pre)
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

    def extract_feat(self, img, gt_semantic_seg_pre):
        """Extract features from images."""
        x = self.backbone(img, gt_semantic_seg_pre)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas, gt_semantic_seg_pre=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if self.tile_inference:
            img, n_h, n_w = split_into_tiles(img, self.inference_tile_size)
            gt_semantic_seg_pre, _, _ = split_into_tiles(gt_semantic_seg_pre, self.inference_tile_size)

        x = self.extract_feat(img, gt_semantic_seg_pre)
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


@SEGMENTORS.register_module()
class DummyEncoderDecoderCMCDWrapper(EncoderDecoderCMCD):
    """Encoder Decoder wrapper for FLOPs calculation.
    
    This wrapper accepts (img, gt_semantic_seg_pre) as input for accurate FLOPs counting.
    It directly calls the model's core computation path (extract_feat + decode_head)
    to ensure thop can properly hook into all operations.
    
    - img: Tensor of shape (B, C, H, W), single time-phase image (will be duplicated internally)
    - gt_semantic_seg_pre: Tensor of shape (B, H, W), semantic segmentation map
    """

    @auto_fp16(apply_to=('img',))
    def forward(self, img, gt_semantic_seg_pre):
        """Forward function for FLOPs calculation.
        
        Directly calls extract_feat and decode_head to ensure all operations
        are properly tracked by thop.
        
        Args:
            img: Tensor of shape (B, C, H, W), single time-phase image
            gt_semantic_seg_pre: Tensor of shape (B, H, W), semantic segmentation map
        
        Returns:
            Segmentation logits tensor
        """
        # Duplicate img to create bi-temporal input (img1, img2)
        # img_concat = torch.cat([img, img], dim=1)
        img_concat = img
        
        # Ensure gt_semantic_seg_pre has correct shape (B, H, W)
        if gt_semantic_seg_pre.dim() == 4:
            gt_semantic_seg_pre = gt_semantic_seg_pre.squeeze(1)
        
        # Step 1: Extract features (backbone + map_encoder + neck)
        x = self.extract_feat(img=img_concat)
        
        # Step 2: Decode head forward
        out = self.decode_head.bc_head.forward(x, gt_semantic_seg_pre=gt_semantic_seg_pre)
        
        return out


def split_into_tiles(tensor, tile_size):
    if tensor.ndim == 4:
        b, c, h, w = tensor.shape
        assert h % tile_size == 0 and w % tile_size == 0
        n_h = h // tile_size
        n_w = w // tile_size

        tiles = tensor.reshape(b,c,n_h,tile_size,n_w, tile_size)
        tiles = tiles.permute(0,2,4,1,3,5).reshape(b*n_h*n_w, c, tile_size, tile_size)
        return tiles, n_h, n_w
    elif tensor.ndim == 3:
        b, h, w = tensor.shape
        assert h % tile_size == 0 and w % tile_size == 0
        n_h = h // tile_size
        n_w = w // tile_size
        tiles = tensor.reshape(b, n_h, tile_size,n_w, tile_size)
        tiles = tiles.permute(0,1,3,2,4).reshape(b*n_h*n_w, tile_size, tile_size)
        return tiles, n_h, n_w
    else:
        raise ValueError('Invalid number of dimensions: ', tensor.ndim)

def merge_tiles(tiles, n_h, n_w):
    if tiles.ndim == 4:
        n_t, c, t_h, t_w = tiles.shape
        b = n_t // n_h // n_w
        tiles = tiles.reshape(b, n_h, n_w, c, t_h, t_w).permute(0,3,1,4,2,5)
        tensor = tiles.reshape(b, c, n_h * t_h, n_w * t_w)
        return tensor
    if tiles.ndim == 3:
        n_t, t_h, t_w = tiles.shape
        b = n_t // n_h // n_w
        tiles = tiles.reshape(b, n_h, n_w, t_h, t_w).permute(0,1,3,2,4)
        tensor = tiles.reshape(b, n_h * t_h, n_w * t_w)
        return tensor
    else:
        raise ValueError('Invalid number of dimensions: ', tiles.ndim)