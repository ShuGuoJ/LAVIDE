import os.path as osp
import numpy as np

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from ..builder import DATASETS
from ..pipelines import ComposeWithVisualization
from .custom_ccd import CustomDatasetCCD
from tqdm import tqdm
from multiprocessing import Pool


MEAN = (0,0,0)
STD = (1,1,1)


@DATASETS.register_module()
class SecondCCD(CustomDatasetCCD):
    '''
    HRSCD for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['building', 'facade']
    
    def __init__(
        self,
        pipeline,
        data_root,
        split,
        ann_dir=None,
        img_suffix='.png',
        seg_map_suffix='.png',
        test_mode=False,
        ignore_index_bc=255,
        ignore_index_sem=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        if_visualize=False,
        label_map = None
    ):
        self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.data_root = data_root
        self.img_dir = osp.join(data_root, 'images')
        self.ann_dir = osp.join(data_root, 'labels')
        self.split = split
        with open(osp.join(data_root, 'splits', split + '.txt'), 'r') as f:
            sites = [s.strip() for s in f.readlines()]
        self.sites = sites
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        # load annotations

        self.img_infos = self.load_img_infos()

        self.test_mode = test_mode
        self.ignore_index_bc = ignore_index_bc
        self.ignore_index_sem = ignore_index_sem
        self.reduce_zero_label = reduce_zero_label
        self.label_map = label_map     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

    def load_img_infos(self):
        img_infos = []
        for site in self.sites:
            img_pre, img_post, seg_map_pre, seg_map_post, seg_map = site.split(' ')
            img_info = dict(
                filename=osp.join(self.data_root, img_post),
                filename_pre=osp.join(self.data_root, img_pre),
                ann=dict(
                    seg_map=osp.join(self.data_root, seg_map),
                    seg_map_pre=osp.join(self.data_root, seg_map_pre),
                    seg_map_post=osp.join(self.data_root, seg_map_post)
                )
            )
            img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} image pairs', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []

        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_bc_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_bc_maps = []
        for img_info in self.img_infos:
            bc_map_file = img_info['ann']['seg_map']
            gt_bc_map = mmcv.imread(
                bc_map_file, flag='unchanged', backend='pillow')
            gt_bc_maps.append(gt_bc_map)
        
        return gt_bc_maps

    def get_gt_sem_maps(self, efficient_test=False):
        gt_sem_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info['ann']['seg_map_post']
            gt_seg_map_post = mmcv.imread(
                seg_map_post, flag='unchanged', backend='pillow')
            # reduce zero label
            # avoid using underflow conversion
            gt_seg_map_post[gt_seg_map_post == 0] = self.ignore_index_sem
            gt_seg_map_post = gt_seg_map_post - 1
            gt_seg_map_post[gt_seg_map_post == self.ignore_index_sem - 1] = self.ignore_index_sem
            gt_sem_maps.append(gt_seg_map_post.astype(np.uint8))

        return gt_sem_maps



def post_process(x):
    x = x.squeeze().astype(np.uint8)
    return x