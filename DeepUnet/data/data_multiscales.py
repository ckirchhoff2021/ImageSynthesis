import os
import json
import random
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


def train_dataloader(path, image_file='GoProJson/train_data.json', count=1e5, batch_size=64, num_workers=0, use_transform=True, distributed=False):
    train_dataset = DeblurDatasetX(path, image_file, count=count, transform=use_transform)

    if distributed:
        import horovod.torch as hvd
        hvd.init()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        return dataloader, train_sampler
    else:
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    return dataloader


def test_dataloader(path, image_file='GoProJson/test_data.json', count=1e5, batch_size=1, num_workers=0, distributed=False):
    test_dataset = DeblurDatasetX(path, image_file, count=count, is_test=True)

    if distributed:
        import horovod.torch as hvd
        hvd.init()
        train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(),
                                                                        rank=hvd.rank())
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        return dataloader, train_sampler

    else:
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return dataloader


def valid_dataloader(path, image_file='GoProJson/test_data.json', count=1e5, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDatasetX(path, image_file, count=count, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class DeblurDatasetX(Dataset):
    def __init__(self, image_dir, data_json, count=1e5, transform=False, is_test=False):
        self.count = count
        self.image_dir = image_dir
        image_list = json.load(open(os.path.join(image_dir, data_json), 'r'))
        random.shuffle(image_list)
        cnt = len(image_list)
        self.count = self.count if self.count < cnt else cnt

        self.image_list = image_list[:self.count]
        print('==> Action: image_dir = {}, count = {}'.format(self.image_dir, self.count))
        self.transform = transform
        self.is_test = is_test

        self.datas = list()
        self.initialize()

    def get_transform(self, transform_type):
        size = int(transform_type)
        return PairCompose([
                PairRandomCrop(size),
                PairRandomHorizontalFilp(),
                PairToTensor()
                ])

    def initialize(self):
        for i in range(len(self.image_list)):
            if not self.is_test:
                self.datas.append([i, '256'])
                self.datas.append([i, '128'])
                self.datas.append([i, '64'])
            else:
                self.datas.append[i, '256']
        random.shuffle(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        index, transform_type = self.datas[idx]
        b, s = self.image_list[index]
        image = Image.open(os.path.join(self.image_dir, b))
        label = Image.open(os.path.join(self.image_dir, s))

        if self.transform:
            transform = self.get_transform(transform_type)
            x, y = transform(image, label)
            x, y = x.unsqueeze(0), y.unsqueeze(0)
    
            if transform_type == '256':
                x256, y256 = x, y
                x128 = F.interpolate(x256, scale_factor=0.5)
                y128 = F.interpolate(y256, scale_factor=0.5)
                x64 = F.interpolate(x128, scale_factor=0.5)
                y64 = F.interpolate(y128, scale_factor=0.5)

            elif transform_type == '128':
                x128, y128 = x, y
                x256 = F.interpolate(x128, scale_factor=2)
                y256 = F.interpolate(y128, scale_factor=2)
                x64 = F.interpolate(x128, scale_factor=0.5)
                y64 = F.interpolate(y128, scale_factor=0.5)
            
            else:
                x64, y64 = x, y
                x128 = F.interpolate(x64, scale_factor=2)
                y128 = F.interpolate(y64, scale_factor=2)
                x256 = F.interpolate(x128, scale_factor=2)
                y256 = F.interpolate(y128, scale_factor=2)

        else:
            x, y = F.to_tensor(image), F.to_tensor(label)
            x, y = x.unsquueze(0), y.unsqueeze(0)
            x256, y256 = x, y
            x128 = F.interpolate(x256, scale_factor=0.5)
            y128 = F.interpolate(y256, scale_factor=0.5)
            x64 = F.interpolate(x128, scale_factor=0.5)
            y64 = F.interpolate(y128, scale_factor=0.5)


        if self.is_test:
            name = self.image_list[idx][0]
            return x256[0], y256[0], x128[0], y128[0], x64[0], y64[0], name

        return x256[0], y256[0], x128[0], y128[0], x64[0], y64[0]


