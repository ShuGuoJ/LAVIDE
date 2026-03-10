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
class HRSCDatasetCCD(CustomDatasetCCD):    
    '''
    HRSCD for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['artificial', 'agricultural', 'forest', 'wetland', 'water']
    
    def __init__(
        self,
        pipeline,
        data_root,
        split,
        ann_dir=None,
        img_suffix='.tif',
        seg_map_suffix='.tif',
        test_mode=False,
        ignore_index_bc=255,
        ignore_index_sem=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        if_visualize=False
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
        self.label_map = None     # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

    def load_img_infos(self):
        img_infos = []
        for site_pre in self.sites:
            d = 'D' + site_pre[:2]

            for tile in mmcv.scandir(osp.join(self.img_dir, '2006', d, site_pre), recursive=False, suffix=self.img_suffix):
                splitted = site_pre.split('-')
                splitted[1] = '2012'
                site_post = '-'.join(splitted)
                tile_seg = tile[:-len(self.img_suffix)] + self.seg_map_suffix
                
                img_info = dict(
                    filename=osp.join(self.img_dir, '2012', d, site_post, tile),
                    filename_pre=osp.join(self.img_dir, '2006', d, site_pre, tile),
                    ann=dict(seg_map=osp.join(self.ann_dir, 'change', d, site_post, tile_seg),
                                seg_map_pre=osp.join(self.ann_dir, '2006', d, site_post, tile_seg), # somehow the files are named with 2012 here as well in the original dataset
                                seg_map_post=osp.join(self.ann_dir, '2012', d, site_post, tile_seg))
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
                bc_map_file, flag='unchanged', backend='tifffile')            
            gt_bc_maps.append(gt_bc_map)
        
        return gt_bc_maps

    def get_gt_sem_maps(self, efficient_test=False):
        gt_sem_maps = []
        for img_info in self.img_infos:
            seg_map_post = img_info['ann']['seg_map_post']
            gt_seg_map_post = mmcv.imread(
                seg_map_post, flag='unchanged', backend='tifffile')
            # reduce zero label
            # avoid using underflow conversion
            gt_seg_map_post[gt_seg_map_post == 0] = self.ignore_index_sem
            gt_seg_map_post = gt_seg_map_post - 1
            gt_seg_map_post[gt_seg_map_post == self.ignore_index_sem - 1] = self.ignore_index_sem
            gt_sem_maps.append(gt_seg_map_post.astype(np.uint8))

        return gt_sem_maps


@DATASETS.register_module()
class FasterHRSCDatasetCCD(HRSCDatasetCCD):
    '''
    HRSCD for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['artificial', 'agricultural', 'forest', 'wetland', 'water']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preloading()

    @staticmethod
    def loading(file, reader, file_flag, file_type, postprocess=None):
        file_bytes = reader.get(file)
        img = mmcv.imfrombytes(file_bytes, flag=file_flag, backend=file_type)
        if postprocess is not None:
            img = postprocess(img)
        return file, img

    def multi_proc_loading(self, plist, n_proc, client, file_flag, file_type, desc=None, postprocess=None):
        print('Loading {} images with {} processes...'.format(len(plist), n_proc))
        pbar = tqdm(total=len(plist))
        if desc is None:
            pbar.set_description('Loading img')
        else:
            pbar.set_description(desc=desc)

        def update_bar(*args):
            pbar.update()

        imgs = {}
        pool = Pool(n_proc)
        outs = []
        for path in plist:
            outs.append(pool.apply_async(FasterHRSCDatasetCCD.loading, args=(path, client, file_flag, file_type, postprocess), callback=update_bar))
        pool.close()
        pool.join()
        for out in outs:
            k, v = out.get()
            imgs[k] = v
        return imgs

    def preloading(self):
        loaded_imgs = {}
        loaded_anns = {}

        img_paths, ann_paths = [], []
        for info in self.img_infos:
            img_paths.append(info['filename'])
            img_paths.append(info['filename_pre'])

            ann_paths.append(info['ann']['seg_map'])
            ann_paths.append(info['ann']['seg_map_pre'])
            ann_paths.append(info['ann']['seg_map_post'])

        img_paths = list(set(img_paths))
        ann_paths = list(set(ann_paths))

        assert 3 * len(img_paths) == 2 * len(ann_paths)

        file_client = mmcv.FileClient(backend='disk')

        import psutil
        mem = psutil.virtual_memory()
        init_mem = float(mem.used) / 1024 / 1024 / 1024
        loaded_imgs = self.multi_proc_loading(img_paths, 4, client=file_client, file_flag='color', file_type='tifffile', desc='loading img...')
        mem = psutil.virtual_memory()
        last_mem = float(mem.used) / 1024 / 1024 / 1024
        print(f'Init: {init_mem:.2f}GB\tLast: {last_mem:.2f}GB\tDelta: {last_mem - init_mem:.2f}GB')

        mem = psutil.virtual_memory()
        init_mem = float(mem.used) / 1024 / 1024 / 1024
        loaded_anns = self.multi_proc_loading(ann_paths, 4, client=file_client, file_flag='unchanged', file_type='tifffile', desc='loading ann...', postprocess=post_process)
        mem = psutil.virtual_memory()
        last_mem = float(mem.used) / 1024 / 1024 / 1024
        print(f'Init: {init_mem:.2f}GB\tLast: {last_mem:.2f}GB\tDelta: {last_mem - init_mem:.2f}GB')
        # exit(0)

        # for k, v in loaded_anns.items():
        #     loaded_anns[k] = v.squeeze().astype(np.uint8)

        # for path in tqdm(img_paths, desc='loading img...'):
        #     img_bytes = file_client.get(path)
        #     img = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
        #     loaded_imgs[path] = img
        #
        # for path in tqdm(ann_paths, desc='loading ann...'):
        #     ann_bytes = file_client.get(path)
        #     ann = mmcv.imfrombytes(ann_bytes, flag='unchanged', backend='tifffile').squeeze().astype(np.uint8)
        #     loaded_anns[path] = ann

        self.loaded_imgs, self.loaded_anns = loaded_imgs, loaded_anns


    # def preloading(self):
    #     loaded_imgs = {}
    #     loaded_anns = {}
    #
    #     img_paths, ann_paths = [], []
    #     for info in self.img_infos:
    #         img_paths.append(info['filename'])
    #         img_paths.append(info['filename_pre'])
    #
    #         ann_paths.append(info['ann']['seg_map'])
    #         ann_paths.append(info['ann']['seg_map_pre'])
    #         ann_paths.append(info['ann']['seg_map_post'])
    #
    #     img_paths = set(img_paths)
    #     ann_paths = set(ann_paths)
    #     assert 3 * len(img_paths) == 2 * len(ann_paths)
    #
    #     file_client = mmcv.FileClient(backend='disk')
    #
    #     for path in tqdm(img_paths, desc='loading img...'):
    #         img_bytes = file_client.get(path)
    #         img = mmcv.imfrombytes(img_bytes, flag='color', backend='tifffile')
    #         loaded_imgs[path] = img
    #
    #     for path in tqdm(ann_paths, desc='loading ann...'):
    #         ann_bytes = file_client.get(path)
    #         ann = mmcv.imfrombytes(ann_bytes, flag='unchanged', backend='tifffile').squeeze().astype(np.uint8)
    #         loaded_anns[path] = ann
    #
    #     self.loaded_imgs, self.loaded_anns = loaded_imgs, loaded_anns

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        img1_file, img2_file = img_info['filename_pre'], img_info['filename']
        img1, img2 = self.loaded_imgs[img1_file], self.loaded_imgs[img2_file]

        gt_semantic_seg_bc = ann_info['seg_map']
        gt_semantic_seg_post_file, gt_semantic_seg_pre_file = ann_info['seg_map_post'], ann_info['seg_map_pre']
        gt_semantic_seg_bc = self.loaded_anns[gt_semantic_seg_bc]
        gt_semantic_seg_post, gt_semantic_seg_pre = self.loaded_anns[gt_semantic_seg_post_file], self.loaded_anns[
            gt_semantic_seg_pre_file]

        results = dict(img_info=img_info, ann_info=ann_info)
        results.update({'img1': img1, 'img2': img2})
        results.update({'gt_semantic_seg_bc': gt_semantic_seg_bc, 'gt_semantic_seg_post': gt_semantic_seg_post, 'gt_semantic_seg_pre': gt_semantic_seg_pre})
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        #
        # img1_file, img2_file = img_info['filename_pre'], img_info['filename']
        # img1, img2 = self.loaded_imgs[img1_file], self.loaded_imgs[img2_file]
        #
        # gt_semantic_seg_post_file, gt_semantic_seg_pre_file = ann_info['seg_map'], ann_info['seg_map_pre']
        # gt_semantic_seg_post, gt_semantic_seg_pre = self.loaded_anns[gt_semantic_seg_post_file], self.loaded_anns[
        #     gt_semantic_seg_pre_file]
        #
        # results = dict(img_info=img_info, ann_info=ann_info)
        # results.update({'img1': img1, 'img2': img2})
        # results.update({'gt_semantic_seg_post': gt_semantic_seg_post, 'gt_semantic_seg_pre': gt_semantic_seg_pre})
        # self.pre_pipeline(results)
        # return self.pipeline(results)

        return self.prepare_train_img(idx)


@DATASETS.register_module()
class PartialFasterHRSCDatasetCCD(FasterHRSCDatasetCCD):
    '''
    HRSCD for Conditional CD.
    Inherits evaluate() from CustomDatasetCCD
    '''
    CLASSES = ['artificial', 'agricultural', 'forest', 'wetland', 'water']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preloading(self):
        loaded_imgs = {}
        loaded_anns = {}

        img_paths, ann_paths = [], []
        for info in self.img_infos:
            img_paths.append(info['filename'])
            img_paths.append(info['filename_pre'])

            ann_paths.append(info['ann']['seg_map'])
            ann_paths.append(info['ann']['seg_map_pre'])
            ann_paths.append(info['ann']['seg_map_post'])

        img_paths = list(set(img_paths))
        ann_paths = list(set(ann_paths))

        assert 3 * len(img_paths) == 2 * len(ann_paths)

        file_client = mmcv.FileClient(backend='disk')

        import psutil
        # loading imgs
        mem = psutil.virtual_memory()
        init_mem = float(mem.used) / 1024 / 1024 / 1024
        loaded_imgs = self.multi_proc_loading(img_paths, 4, client=file_client, file_flag='color', file_type='tifffile', desc='loading img...')
        mem = psutil.virtual_memory()
        last_mem = float(mem.used) / 1024 / 1024 / 1024
        print(f'Init: {init_mem:.2f}GB\tLast: {last_mem:.2f}GB\tDelta: {last_mem - init_mem:.2f}GB')

        # loading anns
        # mem = psutil.virtual_memory()
        # init_mem = float(mem.used) / 1024 / 1024 / 1024
        # loaded_anns = self.multi_proc_loading(ann_paths, 4, client=file_client, file_flag='unchanged', file_type='tifffile', desc='loading ann...', postprocess=post_process)
        # mem = psutil.virtual_memory()
        # last_mem = float(mem.used) / 1024 / 1024 / 1024
        # print(f'Init: {init_mem:.2f}GB\tLast: {last_mem:.2f}GB\tDelta: {last_mem - init_mem:.2f}GB')

        self.loaded_imgs = loaded_imgs

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        img1_file, img2_file = img_info['filename_pre'], img_info['filename']
        img1, img2 = self.loaded_imgs[img1_file], self.loaded_imgs[img2_file]

        results = dict(img_info=img_info, ann_info=ann_info)
        results.update({'img1': img1, 'img2': img2})
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        #
        # img1_file, img2_file = img_info['filename_pre'], img_info['filename']
        # img1, img2 = self.loaded_imgs[img1_file], self.loaded_imgs[img2_file]
        #
        # gt_semantic_seg_post_file, gt_semantic_seg_pre_file = ann_info['seg_map'], ann_info['seg_map_pre']
        # gt_semantic_seg_post, gt_semantic_seg_pre = self.loaded_anns[gt_semantic_seg_post_file], self.loaded_anns[
        #     gt_semantic_seg_pre_file]
        #
        # results = dict(img_info=img_info, ann_info=ann_info)
        # results.update({'img1': img1, 'img2': img2})
        # results.update({'gt_semantic_seg_post': gt_semantic_seg_post, 'gt_semantic_seg_pre': gt_semantic_seg_pre})
        # self.pre_pipeline(results)
        # return self.pipeline(results)

        return self.prepare_train_img(idx)


def post_process(x):
    x = x.squeeze().astype(np.uint8)
    return x