import os
import os.path as osp
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 10_000_000_000

parser = argparse.ArgumentParser(description='Create compressed tiles.')
parser.add_argument('--data_dir', type=str, help='Data dir')
parser.add_argument('--out_dir', type=str, help='Out dir')
parser.add_argument('--tile_size', type=int, help='Size of the final tiles')
parser.add_argument('--img_compression', type=str, default='jpeg')
parser.add_argument('--label_compression', type=str, default='tiff_lzw')
args = parser.parse_args()


def process_img(split, data):
    img_list = []
    tag = 'img'
    for x in tqdm(data, desc='processing img...'):
        img1, img2 = [i for i in x.split(' ') if i[1:1+len(tag)] == tag]
        img1, img2 = img1[1:], img2[1:]
        if split == 'test':
            img1_path = osp.join(args.data_dir, split, split, split, img1)
            img2_path = osp.join(args.data_dir, split, split, split, img2)
        else:
            img1_path = osp.join(args.data_dir, split, split, img1)
            img2_path = osp.join(args.data_dir, split, split, img2)
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))

        org_img1_root = osp.dirname(img1_path)
        org_img2_root = osp.dirname(img2_path)
        org_img1_root = org_img1_root.split('/')[-2:]
        org_img1_root = '/'.join(org_img1_root)
        org_img2_root = org_img2_root.split('/')[-2:]
        org_img2_root = '/'.join(org_img2_root)
        img1_save_root = osp.join(args.out_dir, 'images', split, org_img1_root)
        img2_save_root = osp.join(args.out_dir, 'images', split, org_img2_root)
        os.makedirs(img1_save_root, exist_ok=True)
        os.makedirs(img2_save_root, exist_ok=True)
        suffix = img1_path.split('.')[-1]

        img1_name = osp.basename(img1_path)[:-(len('.' + suffix))]
        img2_name = osp.basename(img2_path)[:-(len('.' + suffix))]

        assert img1.shape == img2.shape, f'{img1.shape} != {img2.shape}'
        for i in range(img1.shape[0] // args.tile_size):
            for j in range(img1.shape[1] // args.tile_size):
                img1_tile = img1[args.tile_size * i:args.tile_size * (i+1),
                    args.tile_size * j:args.tile_size * (j+1)]
                img2_tile = img2[args.tile_size * i:args.tile_size * (i+1),
                    args.tile_size * j:args.tile_size * (j+1)]

                img1_tile_path = osp.join(img1_save_root, f'{img1_name}_{i}_{j}.{suffix}')
                img2_tile_path = osp.join(img2_save_root, f'{img2_name}_{i}_{j}.{suffix}')
                Image.fromarray(img1_tile).save(img1_tile_path, compression=args.img_compression)
                Image.fromarray(img2_tile).save(img2_tile_path, compression=args.img_compression)
                item = f'{img1_tile_path[len(args.out_dir + "/"):]} {img2_tile_path[len(args.out_dir + "/"):]}'
                img_list.append(item)

    return img_list

def process_sem_label(split, data):
    sem_label_list = []
    tag = 'building_labels'
    for x in tqdm(data, desc='processing sem_label...'):
        sem_label1, sem_label2 = [i for i in x.split(' ') if i[1:1 + len(tag)] == tag]
        sem_label1, sem_label2 = sem_label1[1:], sem_label2[1:]
        if split == 'test':
            sem_label1_path = osp.join(args.data_dir, split, split, split, sem_label1)
            sem_label2_path = osp.join(args.data_dir, split, split, split, sem_label2)
        else:
            sem_label1_path = osp.join(args.data_dir, split, split, sem_label1)
            sem_label2_path = osp.join(args.data_dir, split, split, sem_label2)
        sem_label1 = np.array(Image.open(sem_label1_path))
        sem_label2 = np.array(Image.open(sem_label2_path))
        # change 2 to 0, ignoring building facades
        # sem_label1[sem_label1 == 2] = 0
        # sem_label2[sem_label2 == 2] = 0

        org_sem_label1_root = osp.dirname(sem_label1_path)
        org_sem_label2_root = osp.dirname(sem_label2_path)
        org_sem_label1_root = org_sem_label1_root.split('/')[-2:]
        org_sem_label1_root = '/'.join(org_sem_label1_root)
        org_sem_label2_root = org_sem_label2_root.split('/')[-2:]
        org_sem_label2_root = '/'.join(org_sem_label2_root)
        sem_label1_save_root = osp.join(args.out_dir, 'labels', split, org_sem_label1_root)
        sem_label2_save_root = osp.join(args.out_dir, 'labels', split, org_sem_label2_root)
        os.makedirs(sem_label1_save_root, exist_ok=True)
        os.makedirs(sem_label2_save_root, exist_ok=True)
        suffix = sem_label1_path.split('.')[-1]

        sem_label1_name = osp.basename(sem_label1_path)[:-(len('.' + suffix))]
        sem_label2_name = osp.basename(sem_label2_path)[:-(len('.' + suffix))]

        assert sem_label1.shape == sem_label2.shape, f'{sem_label1.shape} != {sem_label2.shape}'
        for i in range(sem_label1.shape[0] // args.tile_size):
            for j in range(sem_label1.shape[1] // args.tile_size):
                sem_label1_tile = sem_label1[args.tile_size * i:args.tile_size * (i + 1),
                            args.tile_size * j:args.tile_size * (j + 1)]
                sem_label2_tile = sem_label2[args.tile_size * i:args.tile_size * (i + 1),
                            args.tile_size * j:args.tile_size * (j + 1)]

                sem_label1_tile_path = osp.join(sem_label1_save_root, f'{sem_label1_name}_{i}_{j}.{suffix}')
                sem_label2_tile_path = osp.join(sem_label2_save_root, f'{sem_label2_name}_{i}_{j}.{suffix}')
                Image.fromarray(sem_label1_tile).save(sem_label1_tile_path, compression=args.img_compression)
                Image.fromarray(sem_label2_tile).save(sem_label2_tile_path, compression=args.img_compression)
                item = f'{sem_label1_tile_path[len(args.out_dir + "/"):]} {sem_label2_tile_path[len(args.out_dir + "/"):]}'
                sem_label_list.append(item)

    return sem_label_list

def process_change_label(split, data):
    change_label_list = []
    tag = 'labels'
    for x in tqdm(data, desc='processing change_label...'):
        [change_label,] = [i for i in x.split(' ') if i[1:1 + len(tag)] == tag]
        change_label = change_label[1:]
        if split == 'test':
            change_label_path = osp.join(args.data_dir, split, split, split, change_label)
        else:
            change_label_path = osp.join(args.data_dir, split, split, change_label)
        change_label = np.array(Image.open(change_label_path))
        if split == 'val':
            change_label[change_label == 255] = 1

        org_change_label_root = osp.dirname(change_label_path)
        org_change_label_root = org_change_label_root.split('/')[-2:]
        org_change_label_root = '/'.join(org_change_label_root)

        change_label_save_root = osp.join(args.out_dir, 'labels', 'change', split, org_change_label_root)
        os.makedirs(change_label_save_root, exist_ok=True)
        suffix = change_label_path.split('.')[-1]

        change_label_name = osp.basename(change_label_path)[:-(len('.' + suffix))]

        for i in range(change_label.shape[0] // args.tile_size):
            for j in range(change_label.shape[1] // args.tile_size):
                change_label_tile = change_label[args.tile_size * i:args.tile_size * (i + 1),
                                  args.tile_size * j:args.tile_size * (j + 1)]

                change_label_tile_path = osp.join(change_label_save_root, f'{change_label_name}_{i}_{j}.{suffix}')
                Image.fromarray(change_label_tile).save(change_label_tile_path, compression=args.img_compression)
                item = f'{change_label_tile_path[len(args.out_dir + "/"):]}'
                change_label_list.append(item)

    return change_label_list


def process_split(split):
    # split_path = osp.join(args.data_dir, f'list_BANDON_{split}_test.txt') # for testing
    split_path = osp.join(args.data_dir, f'list_BANDON_{split}.txt')
    with open(split_path, 'r') as fr:
        data_items = [l.strip() for l in fr.readlines()]

    img_items = process_img(split=split, data=data_items)
    sem_label_items = process_sem_label(split=split, data=data_items)
    change_label_items = process_change_label(split=split, data=data_items)

    assert len(img_items) == len(sem_label_items), f'{len(img_items)} != {len(sem_label_items)}'
    assert len(img_items) == len(change_label_items), f'{len(img_items)} != {len(change_label_items)}'

    data = []
    for i in range(len(img_items)):
        item = ' '.join([img_items[i], sem_label_items[i], change_label_items[i]])
        data.append(item)
    split_save_root = osp.join(args.out_dir, 'splits')
    os.makedirs(split_save_root, exist_ok=True)
    split_save_path = osp.join(split_save_root, f'{split}.txt')
    with open(split_save_path, 'w') as fw:
        fw.write('\n'.join(data))

    print(f'Have collected {len(data)} pairs')
    print('Finish processing', split, '...')


def main():
    for split in ['train', 'val', 'test']:
    # for split in ['test']: # for testing
        process_split(split)


if __name__ == '__main__':
    main()

