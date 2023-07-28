import torch
import numpy as np
from ..registry import PROCESS

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@PROCESS.register_module
class Normalize(object):
    def __init__(self, img_norm):
        print('Normalize')
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        m = self.mean
        s = self.std
        img = sample['img']
        if len(m) == 1:
            img = img - np.array(m)  # single channel image
            img = img / np.array(s)
        else:
            img = img - np.array(m)[np.newaxis, np.newaxis, ...]
            img = img / np.array(s)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample

@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """
    def __init__(self, keys=['img', 'mask'], cfg = None):
        self.keys = keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(img, -1)
        for key in self.keys:
            if key == 'img_metas' or key == 'gt_masks' or key == 'lane_line':
                data[key] = sample[key]
                continue
            data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

@PROCESS.register_module
class RandomUDoffsetLABEL(object):
    def __init__(self, max_offset, cfg=None):
        self.max_offset = max_offset

    def __call__(self, sample):
        img = sample['img']
        label = sample['mask']
        offset = np.random.randint(-self.max_offset, self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        sample['img'] = img
        sample['mask'] = label
        return sample

def CLRTransforms(img_h, img_w):
    print('CLRTransforms')
    return [
        # dict(name='Resize',
        #      parameters=dict(size=dict(height=img_h, width=img_w)),
        #      p=1.0),
        dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
        dict(name='Affine',
             parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                    y=(-0.1, 0.1)),
                             rotate=(-10, 10),
                             scale=(0.8, 1.2)),
             p=0.7)
        # dict(name='Resize',
        #      parameters=dict(size=dict(height=img_h, width=img_w)),
        #      p=1.0),
    ]
