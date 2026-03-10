import os
import os.path as osp
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import shutil

Image.MAX_IMAGE_PIXELS = 10_000_000_000

parser = argparse.ArgumentParser(description='Create compressed tiles.')
parser.add_argument('--data_dir', type=str, default='/remote-home/share/data/CD/SECOND/SECOND_train_set', help='Data dir')
parser.add_argument('--out_dir', type=str, default='/remote-home/share/data/CD/SECOND/SECOND_train_set/tiles512', help='Out dir')
parser.add_argument('--img_compression', type=str, default='jpeg')
parser.add_argument('--label_compression', type=str, default='tiff_lzw')
args = parser.parse_args()


def process_img():
    img_root_1 = osp.join(args.data_dir, 'im1')
    img_root_2 = osp.join(args.data_dir, 'im2')

    files = os.listdir(img_root_1)
    files = sorted(files)

    img_list = []
    for f in tqdm(files, desc='processing img...'):
        img_path_1 = osp.join(img_root_1, f)
        img_path_2 = osp.join(img_root_2, f)

        new_img_path_1 = osp.join(args.out_dir, 'images', 'im1', f)
        new_img_path_2 = osp.join(args.out_dir, 'images', 'im2', f)

        os.makedirs(osp.dirname(new_img_path_1), exist_ok=True)
        os.makedirs(osp.dirname(new_img_path_2), exist_ok=True)

        shutil.copy(img_path_1, new_img_path_1)
        shutil.copy(img_path_2, new_img_path_2)

        item = f'{new_img_path_1[len(args.out_dir + "/"):]} {new_img_path_2[len(args.out_dir + "/"):]}'
        img_list.append(item)

    return img_list


def process_sem_label():
    sem_root_1 = osp.join(args.data_dir, 'building_label1')
    sem_root_2 = osp.join(args.data_dir, 'building_label2')

    files = os.listdir(sem_root_1)
    files = sorted(files)

    sem_label_list = []
    for f in tqdm(files, desc='processing sem_label...'):
        sem_path_1 = osp.join(sem_root_1, f)
        sem_path_2 = osp.join(sem_root_2 , f)

        new_sem_path_1 = osp.join(args.out_dir, 'labels', 'building_label1', f)
        new_sem_path_2 = osp.join(args.out_dir, 'labels', 'building_label2', f)

        os.makedirs(osp.dirname(new_sem_path_1), exist_ok=True)
        os.makedirs(osp.dirname(new_sem_path_2), exist_ok=True)

        shutil.copy(sem_path_1, new_sem_path_1)
        shutil.copy(sem_path_2, new_sem_path_2)

        item = f'{new_sem_path_1[len(args.out_dir + "/"):]} {new_sem_path_2[len(args.out_dir + "/"):]}'
        sem_label_list.append(item)

    return sem_label_list

def process_change_label():
    change_root = osp.join(args.data_dir, 'change')

    files = os.listdir(change_root)
    files = sorted(files)

    change_label_list = []
    for f in tqdm(files, desc='processing change_label...'):
        change_path = osp.join(change_root, f)

        new_change_path = osp.join(args.out_dir, 'labels', 'change', f)

        os.makedirs(osp.dirname(new_change_path), exist_ok=True)

        shutil.copy(change_path, new_change_path)

        item = f'{new_change_path[len(args.out_dir + "/"):]}'
        change_label_list.append(item)

    return change_label_list


def process_dataset():
    split = 'test'
    img_items = process_img()
    sem_label_items = process_sem_label()
    change_label_items = process_change_label()

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
    process_dataset()
    # for split in ['train', 'val', 'test']:
    # # for split in ['test']: # for testing
    #     process_split(split)


if __name__ == '__main__':
    main()

