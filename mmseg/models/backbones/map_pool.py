import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import BACKBONES
import einops


class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class BasicMaskPool(nn.Module):
    def __init__(self, vis_in):
        super().__init__()
        self.ln = nn.LayerNorm(vis_in)

    def forward(self, x, mask):
        # x: [B, C, H, W]
        # m: [B, N, H, W]
        B, C, H, W = x.shape
        mask_flat = mask.reshape(B, -1, H*W)
        x_flat = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous()
        pool_feat = torch.bmm(mask_flat, x_flat) #[B, N, C]
        pool_feat_norm = self.ln(pool_feat)
        mask_flat_21 = mask.reshape(B, -1, H*W).permute(0, 2, 1).contiguous() #[B, HW, N]
        mask_feat = torch.bmm(mask_flat_21, pool_feat_norm) #[B, HW, C]
        mask_feat = mask_feat.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        return mask_feat


class BasicMaskPoolV2(nn.Module):
    def __init__(self, vis_in):
        super().__init__()
        self.ln = nn.LayerNorm(vis_in)

    def forward(self, x, mask):
        # x: [B, C, H, W]
        # m: [B, N, H, W]
        B, C, H, W = x.shape
        mask_flat = mask.reshape(B, -1, H*W)
        x_flat = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous()
        pool_feat = torch.bmm(mask_flat, x_flat) #[B, N, C]
        pool_feat_norm = self.ln(pool_feat)
        mask_flat_21 = mask.reshape(B, -1, H*W).permute(0, 2, 1).contiguous().detach() #[B, HW, N]
        mask_feat = torch.bmm(mask_flat_21, pool_feat_norm) #[B, HW, C]
        mask_feat = mask_feat.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        return mask_feat


@BACKBONES.register_module()
class BasicMapPool(nn.Module):
    def __init__(self, vis_in, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.fus_conv = nn.Sequential(
            nn.Conv2d(vis_in*2, vis_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, vis_in, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x, m):
        if m.ndim == 3:
            m_ = m.unsqueeze(1)
        else:
            m_ = m
        m_ = F.interpolate(m_.float(), size=x.shape[2:], mode='nearest')
        m_ = m_.long()
        m_ = self.process_m(m_)
        B, C, H, W = x.shape
        m_flat = m_.reshape(B, -1, H*W)
        x_flat = x.reshape(B, C, H*W).permute(0, 2, 1).contiguous()
        mask_pool_feat = torch.bmm(m_flat, x_flat) #[B, N, C]
        mask_pool_feat = mask_pool_feat / (m_flat.sum(dim=2, keepdim=True) + 1e-8)
        m_flat_21 = m_.reshape(B, -1, H*W).permute(0, 2, 1).contiguous() #[B, HW, N]
        mask_feat = torch.bmm(m_flat_21, mask_pool_feat) #[B, HW, C]
        mask_feat = mask_feat.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_fus = self.fus_conv(torch.cat([x, mask_feat], dim=1))

        return x_fus, x_fus

    def process_m(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        B, H, W = x.shape
        one_hot_channels = self.num_classes + 1
        # last index for ignore
        x_ = torch.clone(x)
        x_[x > self.num_classes] = self.num_classes
        with torch.no_grad():
            one_hot = nn.functional.one_hot(
                x_.long(), num_classes=one_hot_channels)
            one_hot = one_hot.permute(0, 3, 1, 2).reshape(
                B, one_hot_channels, H, W).float()
            one_hot = one_hot.contiguous()

        return one_hot


@BACKBONES.register_module()
class BasicMapPoolSoft(BasicMapPool):
    def __init__(self, vis_in, num_classes, **kwargs):
        super().__init__(vis_in=vis_in, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.fus_conv = nn.Sequential(
            nn.Conv2d(vis_in*2, vis_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, vis_in, kernel_size=1, stride=1, bias=False)
        )
        self.mask_pool = BasicMaskPool(vis_in=vis_in)

    def forward(self, x, m):
        if m.ndim == 3:
            m_ = m.unsqueeze(1)
        else:
            m_ = m
        m_ = F.interpolate(m_.float(), size=x.shape[2:], mode='nearest')
        m_ = m_.long()
        m_ = self.process_m(m_)
        m_ = torch.ones_like(m_)
        m_ = m_ / m_.shape[1]
        mask_feat = self.mask_pool(x, m_)
        x_fus = self.fus_conv(torch.cat([x, mask_feat], dim=1))

        return x_fus, x_fus


@BACKBONES.register_module()
class BasicImagePoolSoft(BasicMapPool):
    def __init__(self, vis_in, num_classes, **kwargs):
        super().__init__(vis_in=vis_in, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.mask_generator = nn.Sequential(
            nn.Conv2d(vis_in, vis_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, self.num_classes + 1, kernel_size=1, bias=False)
        )

        self.fus_conv = nn.Sequential(
            nn.Conv2d(vis_in*2, vis_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, vis_in, kernel_size=1, stride=1, bias=False)
        )
        self.mask_pool = BasicMaskPool(vis_in=vis_in)

    def forward(self, x, m):
        mask = self.mask_generator(x)
        mask = torch.softmax(mask, dim=1)
        mask_feat = self.mask_pool(x, mask)
        x_fus = self.fus_conv(torch.cat([x, mask_feat], dim=1))

        return x_fus, x_fus


@BACKBONES.register_module()
class BasicImagePoolHard(BasicMapPool):
    def __init__(self, vis_in, num_classes, **kwargs):
        super().__init__(vis_in=vis_in, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.mask_generator = nn.Sequential(
            nn.Conv2d(vis_in, vis_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, self.num_classes + 1, kernel_size=1, bias=False)
        )

        self.fus_conv = nn.Sequential(
            nn.Conv2d(vis_in*2, vis_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, vis_in, kernel_size=1, stride=1, bias=False)
        )
        self.mask_pool = BasicMaskPoolV2(vis_in=vis_in)

    def forward(self, x, m):
        mask = self.mask_generator(x)
        if self.training:
            mask = F.gumbel_softmax(mask, tau=0.2, hard=True, dim=1)
        else:
            mask = torch.argmax(mask, dim=1)
            mask = F.one_hot(mask, self.num_classes + 1)
            mask = mask.permute(0, 3, 1, 2).contiguous().float()
        mask_feat = self.mask_pool(x, mask)
        x_fus = self.fus_conv(torch.cat([x, mask_feat], dim=1))

        return x_fus, x_fus


@BACKBONES.register_module()
class BasicColPoolSoft(BasicMapPool):
    def __init__(self, vis_in, num_classes, **kwargs):
        super().__init__(vis_in=vis_in, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.mask_generator = nn.Sequential(
            nn.Conv2d(vis_in + self.num_classes + 1, vis_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, self.num_classes + 1, kernel_size=1, bias=False)
        )

        self.fus_conv = nn.Sequential(
            nn.Conv2d(vis_in*2, vis_in, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(vis_in),
            nn.ReLU(),
            nn.Conv2d(vis_in, vis_in, kernel_size=1, stride=1, bias=False)
        )
        self.mask_pool = BasicMaskPool(vis_in=vis_in)

    def forward(self, x, m):
        if m.ndim == 3:
            m_ = m.unsqueeze(1)
        else:
            m_ = m
        m_ = self.process_m(m_)
        m_ = F.interpolate(m_.float(), size=x.shape[2:], mode='bilinear', align_corners=False)
        mask = self.mask_generator(torch.concat([x, m_], dim=1))
        mask = torch.softmax(mask, dim=1)
        mask_feat = self.mask_pool(x, mask)
        x_fus = self.fus_conv(torch.cat([x, mask_feat], dim=1))

        return x_fus, x_fus
