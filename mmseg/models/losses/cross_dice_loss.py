from ..builder import LOSSES, build_loss
from torch import nn

@LOSSES.register_module()
class CrossDiceLoss(nn.Module):
    def __init__(self, cross_loss_cfg, dice_loss_cfg):
        super().__init__()
        self.cross_loss = build_loss(cross_loss_cfg)
        self.dice_loss = build_loss(dice_loss_cfg)

    def forward(self, pred, target, **kwargs):
        loss = dict()
        cross_loss = self.cross_loss(pred, target, **kwargs)
        dice_loss = self.dice_loss(pred, target)
        loss['loss_seg'] = cross_loss
        loss['dice_loss'] = dice_loss
        return loss