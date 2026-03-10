from ..builder import LOSSES, build_loss
from torch import nn

@LOSSES.register_module()
class FocalDiceLoss(nn.Module):
    def __init__(self, focal_loss_cfg, dice_loss_cfg):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = build_loss(focal_loss_cfg)
        self.dice_loss = build_loss(dice_loss_cfg)

    def forward(self, pred, target, **kwargs):
        loss = dict()
        focal_loss = self.focal_loss(pred, target, **kwargs)
        dice_loss = self.dice_loss(pred, target)
        loss['focal_loss'] = focal_loss
        loss['dice_loss'] = dice_loss
        return loss