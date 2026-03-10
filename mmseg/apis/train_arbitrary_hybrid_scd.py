import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner, get_dist_info
from mmcv.utils import print_log

from mmseg.core.evaluation.eval_hooks import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
from mmseg.apis import single_gpu_test
from mmseg.dataloaders import JointLoader

from torch.utils.data import Subset

import math, copy


def collect_hook_msgs(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    hook_msgs = checkpoint['meta'].get('hook_msgs', {})
    return hook_msgs


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fill_short_list(A, B):
    '''
    扩充（A,B）中长度最短的列表，使最短的列表长度与最长的列表长度保持一致
    '''
    la, lb = len(A), len(B)
    new_list = []
    if la < lb:
        m = int(math.ceil(lb/la))
        for _ in range(m):
            new_list.extend(copy.deepcopy(A))
        return new_list[:lb], B
    else:
        m = int(math.ceil(la/lb))
        for _ in range(m):
            new_list.extend(copy.deepcopy(B))
        return A, new_list[:la]


def train_arbitrary_hybrid_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)


    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # split dataset
    assert len(dataset) == 2, 'Only two datasets are supported'

    # split dataset
    data_length = len(dataset[0])
    sample_indices = list(range(data_length))
    bc_length = int(data_length * cfg.split_proportion)
    sample_indices_bc, sample_indices_sem = sample_indices[:bc_length], sample_indices[bc_length:]
    logger.info(f'bc length: {len(sample_indices_bc)}, sem length: {len(sample_indices_sem)}')
    sample_indices_bc, sample_indices_sem = fill_short_list(sample_indices_bc, sample_indices_sem)
    dataset[0] = Subset(dataset[0], sample_indices_bc)
    dataset[1] = Subset(dataset[1], sample_indices_sem)
    logger.info(f'bc length: {len(dataset[0].indices)}, sem length: {len(dataset[1].indices)}')
    # dataset = [dataset[1]]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu // len(dataset),
            cfg.data.workers_per_gpu // len(dataset),
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]
    assert len(data_loaders) == 2, 'Only two dataloaders are supported'
    data_loaders = [JointLoader(data_loaders[0], data_loaders[1])]
    # for debugging JointLoader
    # from mmcv.runner.iter_based_runner import IterLoader
    # from tqdm import tqdm
    # iter_loader = IterLoader(data_loaders[0])
    # for i in tqdm(range(5000)):
    #     inputs = next(iter_loader)
    # iter_loader = iter(data_loaders[0])
    # inputs = next(iter_loader)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    for logging_hook in cfg.log_config.hooks:
        if logging_hook.type == 'WandbLoggerHook':
            logging_hook.init_kwargs.config = cfg.to_dict()
            logging_hook.init_kwargs.name = cfg.run_name
            logging_hook.init_kwargs.dir = cfg.work_dir
            logging_hook.init_kwargs.tags = cfg.work_dir.split('/')[:-1]
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook_class = DistEvalHook if distributed else EvalHook
        eval_hook = eval_hook_class(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook, priority='LOW') # https://github.com/open-mmlab/mmcv/issues/1261

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
        hook_msgs = collect_hook_msgs(cfg.resume_from)
        runner.meta['hook_msgs'] = hook_msgs
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

    # test
    rank, _ = get_dist_info()
    if cfg.data.test and rank == 0:
        print_log('============================== Testing ==============================', logger=logger)
        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        test_dataloader = build_dataloader(
            test_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        # best_ckpt_path = runner.meta['hook_msgs']['best_ckpt']
        # runner.load_checkpoint(best_ckpt_path)
        runner.load_checkpoint(eval_hook.best_ckpt_path)
        results = single_gpu_test(
            runner.model,
            test_dataloader,
            show=False,
            efficient_test=False)
        eval_res = test_dataset.evaluate(results, logger=runner.logger)
