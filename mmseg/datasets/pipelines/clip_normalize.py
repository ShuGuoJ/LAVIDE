from ..builder import PIPELINES
from transformers import CLIPModel, CLIPProcessor
from mmcv.image import imdenormalize
from mmseg.utils import split_images
import numpy as np
from PIL import Image


@PIPELINES.register_module()
class DummyClipNormalize(object):
    def __call__(self, results):
        img = results['img']
        mean = results['img_norm_cfg']['mean']
        std = results['img_norm_cfg']['std']
        if img.shape[-1] % 2 == 0:
            img1, img2 = split_images(img)
            img1_denorm = imdenormalize(img1, mean=mean, std=std)
            img2_denorm = imdenormalize(img2, mean=mean, std=std)
            img = np.concatenate((img1_denorm, img2_denorm), axis=-1)
        else:
            img = imdenormalize(img, mean=mean, std=std)
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        results['clip_img'] = img
        return results


@PIPELINES.register_module()
class ClipNormalize(DummyClipNormalize):
    def __init__(self, pretrained, img_scale=(336, 336)):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(pretrained)
        self.img_scale = img_scale

    def __call__(self, results):
        results = super().__call__(results)
        img_denorm = results['clip_img']
        if img_denorm.shape[0] % 2 == 0:
            imgs = np.split(img_denorm, 2, axis=0)
            clip_img = self.processor(images=imgs, return_tensors="pt", do_resize=True, size=self.img_scale, do_center_crop=False).pixel_values
            b, c, h, w = clip_img.shape
            clip_img = clip_img.reshape(b*c, h, w)
        else:
            clip_img = self.processor(images=[img_denorm], return_tensors="pt", do_resize=True, size=self.img_scale, do_center_crop=False).pixel_values
        results['clip_img'] = clip_img
        return results


