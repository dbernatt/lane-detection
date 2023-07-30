import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import logging
from clrnet.utils.visualization import imshow_lanes
from .process import Process

# from mmcv.parallel import DataContainer as DC

class BaseDataset(Dataset):
    def __init__(self, cfg, split, processes):
        print('Init BaseDataset...')
        self.cfg = cfg
        self.work_dirs = self.cfg.work_dirs
        self.data_root = self.cfg.data_root
        self.training = 'train' in split

        print('BaseDataset cfg: ', self.cfg)
        print('BaseDataset processes: ', processes)
        self.processes = Process(processes)

    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.work_dirs, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes] # !!!
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})

        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        # meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})
        # print('sample = ', sample)
        return sample