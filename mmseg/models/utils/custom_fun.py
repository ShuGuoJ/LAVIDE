import torch
from functools import reduce


def get_boundary(x):
    if x.ndim == 3:
        x = x.unsqueeze(1)

    # import time
    # begin = time.time()
    # x: [H, W]
    B, C, H, W = x.shape
    x_pad = torch.ones((B, C, H + 2, W + 2), dtype=x.dtype, device=x.device)
    x_pad[..., 1:H + 1, 1:W + 1] = x
    x_pad[..., 0, 1:W + 1] = x[..., 0, :]
    x_pad[..., -1, 1:W + 1] = x[..., -1, :]
    x_pad[..., 1:H + 1, 0] = x[..., :, 0]
    x_pad[..., 1:H + 1, -1] = x[..., :, -1]
    is_bottom = x != x_pad[..., 0:H, 1:W + 1]
    is_up = x != x_pad[..., 2:H + 2, 1:W + 1]
    is_right = x != x_pad[..., 1:H + 1, 0:W]
    is_left = x != x_pad[..., 1:H + 1, 2:W + 2]
    b_mask = reduce(torch.logical_or, [is_up, is_bottom, is_left, is_right])
    b_mask = b_mask.to(dtype=torch.float32)
    # end = time.time()
    # print(f'time: {end-begin}')
    return b_mask


# def calc_boundary_dist(x):
#     # x: [H, W]
#     if x.ndim == 4:
#         x = x.squeeze(1)
#     H, W = x.shape[-2:]
#     B = x.shape[0]
#     h_axis = torch.arange(H, dtype=x.dtype, device=x.device)
#     w_axis = torch.arange(W, dtype=x.dtype, device=x.device)
#     h_axis, w_axis = torch.meshgrid(h_axis, w_axis)
#     h_axis_flat, w_axis_flat = h_axis.reshape(H*W), w_axis.reshape(H*W)
#     dist = torch.zeros((B, H*W, H*W), dtype=x.dtype, device=x.device)
#     # boundary_mask = x == 1
#     # boundary_mask = boundary_mask.reshape(H*W)
#     for b in range(B):
#         boundary_mask = x[b] == 1
#         boundary_mask = boundary_mask.reshape(H * W)
#         for h in range(H):
#             for w in range(W):
#                 if x[b, h, w] == 0:
#                     continue
#
#                 d = torch.sqrt(torch.pow(h_axis_flat - h, 2) + torch.pow(w_axis_flat - w, 2) + 1e-8)
#                 d = 1 / d
#                 i = h * W + w
#                 d[boundary_mask] = 0
#                 d[i] = 1
#                 dist[b, i] = d
#
#     return dist

# def calc_boundary_dist(x):
#     # x: [H, W]
#     if x.ndim == 4:
#         x = x.squeeze(1)
#     H, W = x.shape[-2:]
#     B = x.shape[0]
#     h_axis = torch.arange(H, dtype=x.dtype, device=x.device)
#     w_axis = torch.arange(W, dtype=x.dtype, device=x.device)
#     h_axis, w_axis = torch.meshgrid(h_axis, w_axis)
#     h_axis_flat, w_axis_flat = h_axis.reshape(H*W), w_axis.reshape(H*W)
#     dist = torch.zeros((1, H*W, H*W), dtype=x.dtype, device=x.device)
#     # boundary_mask = x == 1
#     # boundary_mask = boundary_mask.reshape(H*W)
#     # for b in range(B):
#     #     boundary_mask = x[b] == 1
#     #     boundary_mask = boundary_mask.reshape(H * W)
#     #     for h in range(H):
#     #         for w in range(W):
#     #             if x[b, h, w] == 0:
#     #                 continue
#     #
#     #             d = torch.sqrt(torch.pow(h_axis_flat - h, 2) + torch.pow(w_axis_flat - w, 2) + 1e-8)
#     #             d = 1 / d
#     #             i = h * W + w
#     #             d[boundary_mask] = 0
#     #             d[i] = 1
#     #             dist[b, i] = d
#
#     for h in range(H):
#         for w in range(W):
#             d = torch.sqrt(torch.pow(h_axis_flat - h, 2) + torch.pow(w_axis_flat - w, 2) + 1e-8)
#             d = 1 / d
#             i = h * W + w
#             dist[0, i] = d
#     dist = dist.repeat(B, 1, 1)
#     x_ = x.reshape(B, H*W)
#     dist[x_==0] = 0
#     x_ = x_[:, None].expand(-1, H*W, -1)
#     dist[x_==1] = 0
#     eye = torch.eye(H*W, dtype=x.dtype, device=x.device)
#     eye = eye[None].repeat(B, 1, 1)
#     eye = eye * x_
#     dist[eye==1] = 1
#
#     return dist

def calc_boundary_dist(x):
    # x: [H, W]
    if x.ndim == 4:
        x = x.squeeze(1)
    H, W = x.shape[-2:]
    B = x.shape[0]
    h_axis = torch.arange(H, dtype=x.dtype, device=x.device)
    w_axis = torch.arange(W, dtype=x.dtype, device=x.device)
    h_axis, w_axis = torch.meshgrid(h_axis, w_axis)
    h_axis_flat, w_axis_flat = h_axis.reshape(H*W), w_axis.reshape(H*W)
    hw_axis_flat = torch.stack([h_axis_flat, w_axis_flat], dim=1)
    hwhw_axis_flat = hw_axis_flat[None].expand(H*W, -1, -1)
    hhww_axis_flat = hw_axis_flat[:, None].expand(-1, H*W, -1)
    diff = hhww_axis_flat - hwhw_axis_flat
    dist = 1 / (diff.norm(dim=-1) + 1e-8)
    dist = dist[None]
    # dist = torch.zeros((1, H*W, H*W), dtype=x.dtype, device=x.device)
    # boundary_mask = x == 1
    # boundary_mask = boundary_mask.reshape(H*W)
    # for b in range(B):
    #     boundary_mask = x[b] == 1
    #     boundary_mask = boundary_mask.reshape(H * W)
    #     for h in range(H):
    #         for w in range(W):
    #             if x[b, h, w] == 0:
    #                 continue
    #
    #             d = torch.sqrt(torch.pow(h_axis_flat - h, 2) + torch.pow(w_axis_flat - w, 2) + 1e-8)
    #             d = 1 / d
    #             i = h * W + w
    #             d[boundary_mask] = 0
    #             d[i] = 1
    #             dist[b, i] = d

    # for h in range(H):
    #     for w in range(W):
    #         d = torch.sqrt(torch.pow(h_axis_flat - h, 2) + torch.pow(w_axis_flat - w, 2) + 1e-8)
    #         d = 1 / d
    #         i = h * W + w
    #         dist[0, i] = d
    dist = dist.repeat(B, 1, 1)
    x_ = x.reshape(B, H*W)
    dist[x_==0] = 0
    x_ = x_[:, None].expand(-1, H*W, -1)
    dist[x_==1] = 0
    eye = torch.eye(H*W, dtype=x.dtype, device=x.device)
    eye = eye[None].repeat(B, 1, 1)
    eye = eye * x_
    dist[eye==1] = 1

    return dist


# testing get_boundary
# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     # 假设语义图为一个二维数组（每个像素为类别标签），可以根据实际情况加载图像
#     # 示例：创建一个简单的语义图
#     semantic_map = torch.tensor([
#         [0, 0, 1, 1, 1],
#         [0, 0, 1, 1, 0],
#         [2, 2, 1, 1, 0],
#         [2, 2, 2, 0, 0],
#         [2, 2, 0, 0, 0]
#     ])
#
#     # 获取边界掩膜
#     # boundary_mask = extract_boundary_mask(semantic_map)
#     boundary_mask = get_boundary(semantic_map)
#
#     # 显示结果
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.title("Semantic Map")
#     plt.imshow(semantic_map, cmap='tab20b')  # 使用一个类别标签的色图来显示语义图
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.title("Boundary Mask")
#     plt.imshow(boundary_mask, cmap='gray')  # 使用灰度图显示边界掩膜
#     plt.axis('off')
#
#     plt.show()

# testing calc_boundary_dist
if __name__ == '__main__':
    B1 = [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    B2 = [[0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]]

    B = torch.tensor([B1, B2], dtype=torch.float)
    D = calc_boundary_dist(B)
    print(D)

    # D_ = calc_boundary_dist_(B)
    # print(D_)
    # print(torch.equal(D, D_))
