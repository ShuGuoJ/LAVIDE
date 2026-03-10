import torch


def collect_hook_msgs(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    return checkpoint['meta']['hook_msgs']