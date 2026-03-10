import torch
from torch import Tensor
import numpy as np
from collections import OrderedDict
from .metrics import eval_metrics, total_intersect_and_union, f_score
# from mmseg.models.utils.custom_fun import get_boundary
from functools import reduce

def get_boundary(x):
    # import time
    # begin = time.time()
    # x: [H, W]
    H, W = x.shape
    x_pad = np.ones((H + 2, W + 2), dtype=x.dtype)
    x_pad[1:H + 1, 1:W + 1] = x
    x_pad[0, 1:W + 1] = x[0, :]
    x_pad[-1, 1:W + 1] = x[-1, :]
    x_pad[1:H + 1, 0] = x[:, 0]
    x_pad[1:H + 1, -1] = x[:, -1]
    is_bottom = x != x_pad[0:H, 1:W + 1]
    is_up = x != x_pad[2:H + 2, 1:W + 1]
    is_right = x != x_pad[1:H + 1, 0:W]
    is_left = x != x_pad[1:H + 1, 2:W + 2]
    b_mask = reduce(np.logical_or, [is_up, is_bottom, is_left, is_right])
    b_mask = b_mask.astype(np.uint8)
    # end = time.time()
    # print(f'time: {end-begin}')
    return b_mask


def fast_hist(a, b, n):
    k = (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def scd_eval_metrics(
        results,
        gt_bc_maps,
        gt_sem_maps,
        num_semantic_classes,
        ignore_index_bc,
        ignore_index_sem
    ):
    assert len(results) == len(gt_bc_maps) == len(gt_sem_maps)

    total_bc_intersect = np.zeros((1,), dtype=np.float64)
    total_bc_union = np.zeros((1,), dtype=np.float64)
    total_bc_pred = np.zeros((1,), dtype=np.float64)
    total_bc_gt = np.zeros((1,), dtype=np.float64)

    total_no_bc_intersect = np.zeros((1,), dtype=np.float64)
    total_no_bc_union = np.zeros((1,), dtype=np.float64)
    total_no_bc_pred = np.zeros((1,), dtype=np.float64)
    total_no_bc_gt = np.zeros((1,), dtype=np.float64)

    total_sc_intersect = np.zeros((num_semantic_classes, ), dtype=np.float64)
    total_sc_union = np.zeros((num_semantic_classes, ), dtype=np.float64)

    total_sem_intersect = np.zeros((num_semantic_classes, ), dtype=np.float64)
    total_sem_union = np.zeros((num_semantic_classes, ), dtype=np.float64)

    total_hist = np.zeros((num_semantic_classes, num_semantic_classes), dtype=np.float64)

    total_boundary_intersect = np.zeros((1,), dtype=np.float64)
    total_boundary_union = np.zeros((1,), dtype=np.float64)

    total_inst_no, total_inst_partial, total_inst_full = 0, 0, 0
    pred_inst_no, pred_inst_partial, pred_inst_full = 0, 0, 0

    for i in range(len(results)):
        pred_bc = results[i]['bc']
        pred_sem = results[i]['sem']
        gt_bc = gt_bc_maps[i]
        gt_sem = gt_sem_maps[i]

        mask_bc = (gt_bc != ignore_index_bc)
        # mask_sem = (gt_sem != ignore_index_sem)

        pred_bc_masked = pred_bc[mask_bc]
        gt_bc_masked = gt_bc[mask_bc]
        pred_sem_masked = pred_sem[mask_bc]
        gt_sem_masked = gt_sem[mask_bc]

        # BC
        intersect_bc = np.logical_and((pred_bc_masked == 1), (gt_bc_masked == 1))
        union_bc = np.logical_or((pred_bc_masked == 1), (gt_bc_masked == 1))
        total_bc_intersect = total_bc_intersect + intersect_bc.sum()
        total_bc_union = total_bc_union + union_bc.sum()
        total_bc_pred = total_bc_pred + pred_bc_masked.sum()
        total_bc_gt = total_bc_gt + gt_bc_masked.sum()

        # no BC
        intersect_no_bc = np.logical_and((pred_bc_masked == 0), (gt_bc_masked == 0))
        union_no_bc = np.logical_or((pred_bc_masked == 0), (gt_bc_masked == 0))
        total_no_bc_intersect = total_no_bc_intersect + intersect_no_bc.sum()
        total_no_bc_union = total_no_bc_union + union_no_bc.sum()
        total_no_bc_pred = total_no_bc_pred + (1 - pred_bc_masked).sum()
        total_no_bc_gt = total_no_bc_gt + (1 - gt_bc_masked).sum()
        
        # SC
        change_mask = (gt_bc == 1)
        intersect_sc = pred_sem[change_mask][pred_sem[change_mask] == gt_sem[change_mask]]
        intersect_area_sc = np.histogram(intersect_sc, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        pred_area_sc = np.histogram(pred_sem[change_mask], bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        gt_area_sc = np.histogram(gt_sem[change_mask], bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        union_area_sc = pred_area_sc + gt_area_sc - intersect_area_sc
        total_sc_intersect = total_sc_intersect + intersect_area_sc
        total_sc_union = total_sc_union + union_area_sc

        # sem
        intersect_sem = pred_sem_masked[pred_sem_masked == gt_sem_masked]
        intersect_area_sem = np.histogram(intersect_sem, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        pred_area_sem = np.histogram(pred_sem_masked, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        gt_area_sem = np.histogram(gt_sem_masked, bins=num_semantic_classes, range=(-0.5, num_semantic_classes-0.5))[0]
        union_area_sem = pred_area_sem + gt_area_sem - intersect_area_sem
        total_sem_intersect = total_sem_intersect + intersect_area_sem
        total_sem_union = total_sem_union + union_area_sem

        # sek
        total_hist = total_hist + get_hist(pred_sem_masked, gt_sem_masked, num_semantic_classes)

        # boundary
        if 'boundary' in results[i]:
            pred_boundary = results[i]['boundary']
            gt_boundary = get_boundary(gt_sem)
            intersect_boundary = np.logical_and((pred_boundary == 1), (gt_boundary == 1))
            union_boundary = np.logical_or((pred_boundary == 1), (gt_boundary == 1))
            total_boundary_intersect = total_boundary_intersect + intersect_boundary.sum()
            total_boundary_union = total_boundary_union + union_boundary.sum()

        if 'inst' in results[i]:
            # 将inst转换为one-hot编码
            assert (results[i]['inst'].max() + 1) == results[i]['inst_pred'].shape[0]
            H, W = results[i]['inst'].shape[:2]
            max_index = results[i]['inst'].max() + 1
            inst_one_hot_map = np.eye(max_index)[results[i]['inst']]
            inst_one_hot_map = inst_one_hot_map.reshape(H*W, -1).transpose(1, 0)
            gt_bc_ = gt_bc.reshape(H*W, 1)

            gt_inst_cd_sum = np.matmul(inst_one_hot_map, (gt_bc_==1).astype(inst_one_hot_map.dtype))
            gt_inst_cd_ratio = gt_inst_cd_sum / (inst_one_hot_map.sum(axis=-1, keepdims=True) + 1e-8)

            degradation_gt = np.ones((max_index, 1), dtype=inst_one_hot_map.dtype)
            degradation_gt[gt_inst_cd_ratio > 0.98] = 2
            degradation_gt[gt_inst_cd_ratio < 0.02] = 0
            invalid_mask = inst_one_hot_map.sum(axis=-1) == 0
            ignore_sum = np.matmul(inst_one_hot_map, (gt_bc_==ignore_index_bc).astype(inst_one_hot_map.dtype))
            ignore_ratio = ignore_sum / (inst_one_hot_map.sum(axis=-1, keepdims=True) + 1e-8)
            ignore_mask = ignore_ratio > 0.98
            degradation_gt[invalid_mask] = ignore_index_bc
            degradation_gt[ignore_mask] = ignore_index_bc

            total_inst_no = total_inst_no + (degradation_gt == 0).sum().item()
            total_inst_partial = total_inst_partial + (degradation_gt == 1).sum().item()
            total_inst_full = total_inst_full +(degradation_gt == 2).sum().item()

            pred_inst_ = results[i]['inst_pred'][:, None].copy()
            pred_inst_[invalid_mask] = ignore_index_bc
            pred_inst_[ignore_mask] = ignore_index_bc
            pred_inst_no = pred_inst_no + np.logical_and(pred_inst_ == 0, degradation_gt == 0).sum().item()
            pred_inst_partial = pred_inst_partial + np.logical_and(pred_inst_ == 1, degradation_gt == 1).sum().item()
            pred_inst_full = pred_inst_full + np.logical_and(pred_inst_ == 2, degradation_gt == 2).sum().item()

    # ret_metrics = OrderedDict()
    # ret_metrics['BC'] = (total_bc_intersect / total_bc_union).item()
    # ret_metrics['SC'] = (total_sc_intersect / total_sc_union).mean()
    # ret_metrics['mIoU'] = (total_sem_intersect / total_sem_union).mean()
    # ret_metrics['BC_recall'] = (total_bc_intersect / total_bc_gt).item()
    # ret_metrics['BC_precision'] = (total_bc_intersect / total_bc_pred).item()
    # ret_metrics['SCS'] = 0.5 * (ret_metrics['BC'] + ret_metrics['SC'])
    # ret_metrics['SC_per_class'] = total_sc_intersect / total_sc_union
    # ret_metrics['IoU_per_class'] = total_sem_intersect / total_sem_union

    # kappa
    kappa = cal_kappa(total_hist)

    ret_metrics = OrderedDict()
    ret_metrics['BC'] = (total_bc_intersect / (total_bc_union + 1e-8)).item()
    ret_metrics['SC'] = (total_sc_intersect / (total_sc_union + 1e-8)).mean()
    ret_metrics['mIoU'] = (total_sem_intersect / (total_sem_union +1e-8)).mean()
    ret_metrics['BC_recall'] = (total_bc_intersect / (total_bc_gt +1e-8)).item()
    ret_metrics['BC_precision'] = (total_bc_intersect / (total_bc_pred + 1e-8)).item()
    ret_metrics['SCS'] = 0.5 * (ret_metrics['BC'] + ret_metrics['SC'])
    ret_metrics['SC_per_class'] = total_sc_intersect / (total_sc_union + 1e-8)
    ret_metrics['IoU_per_class'] = total_sem_intersect / (total_sem_union + 1e-8)
    ret_metrics['SeK'] = (kappa * np.exp(ret_metrics['BC'])) / np.e
    ret_metrics['NO_BC'] = (total_no_bc_intersect / (total_no_bc_union + 1e-8)).item()
    ret_metrics['NO_BC_recall'] = (total_no_bc_intersect / (total_no_bc_gt + 1e-8)).item()
    ret_metrics['NO_BC_precision'] = (total_no_bc_intersect / (total_no_bc_pred + 1e-8)).item()
    ret_metrics['F1'] = 2 * ret_metrics['BC_recall'] * ret_metrics['BC_precision'] / (ret_metrics['BC_recall'] + ret_metrics['BC_precision'] + 1e-8)
    ret_metrics['NO_F1'] = 2 * ret_metrics['NO_BC_recall'] * ret_metrics['NO_BC_precision'] / (ret_metrics['NO_BC_recall'] + ret_metrics['NO_BC_precision'] + 1e-8)
    ret_metrics['mF1'] = (ret_metrics['F1'] + ret_metrics['NO_F1']) / 2.
    ret_metrics['mBC'] = (ret_metrics['BC'] + ret_metrics['NO_BC']) / 2.
    ret_metrics['Boundary'] = (total_boundary_intersect / (total_boundary_union + 1e-8)).item()
    #
    ret_metrics['Inst_NO'] = pred_inst_no / (total_inst_no + 1e-8)
    ret_metrics['Inst_Partial'] = pred_inst_partial / (total_inst_partial + 1e-8)
    ret_metrics['Inst_Full'] = pred_inst_full / (total_inst_full + 1e-8)
    ret_metrics['Inst_OA'] = (pred_inst_no + pred_inst_partial + pred_inst_full) / (total_inst_no + total_inst_partial + total_inst_full + 1e-8)

    return ret_metrics
