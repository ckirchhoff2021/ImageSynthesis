import os
import json
import random
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from data import get_normalize, get_corrupt_function, get_transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler


def train_dataloader(path, image_file='GoProJson/train_data.json', count=1e5, batch_size=64, num_workers=0, use_transform=True, distributed=False, crop_size=256):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(crop_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    train_dataset = DeblurDataset(path, image_file, count=count, transform=transform)

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
    test_dataset = DeblurDataset(path, image_file, count=count, is_test=True)

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
        DeblurDataset(path, image_file, count=count, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, data_json, count=1e5, transform=None, is_test=False):
        self.count = count
        self.image_dir = image_dir
        self.image_list = json.load(open(os.path.join(image_dir, data_json), 'r'))
        random.shuffle(self.image_list)

        cnt = len(self.image_list)
        self.count = self.count if self.count < cnt else cnt

        print('==> Action: image_dir = {}, count = {}'.format(self.image_dir, self.count))
        self.transform = transform
        self.is_test = is_test


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        b, s = self.image_list[idx]
        # image = Image.open(os.path.join(self.image_dir, b))
        # label = Image.open(os.path.join(self.image_dir, s))

        image = Image.open(b)
        label = Image.open(s)

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx][0]
            return image, label, name
        return image, label




class DeblurFolderDataset(Dataset):
    def __init__(self, image_dir, count=1e5, transform=None, is_test=False, is_random=True):
        super(DeblurFolderDataset, self).__init__()
        self.count = count
        self.image_dir = image_dir
        
        image_list = os.listdir(self.image_dir)
        self.image_list = list()
        for image in image_list:
            if not image.endswith('.jpg') and not image.endswith('.png'):
                continue
            self.image_list.append(os.path.join(self.image_dir, image))
        random.shuffle(self.image_list)

        cnt = len(self.image_list)
        self.count = self.count if self.count < cnt else cnt

        print('==> Action: image_dir = {}, count = {}'.format(self.image_dir, self.count))
        self.transform = transform
        self.is_test = is_test
        self.is_random = is_random

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        image_file = self.image_list[idx]
        src = Image.open(image_file)
        blur = src.crop([0,0,300,300])
        sharp = src.crop([300,0,600,300])
        pix2pix = src.crop([600,0,900,300])

        images = [blur, pix2pix]
        if self.is_random:
            i = np.random.randint(2)
        else:
            i = 0
        
        if self.transform:
            image, label = self.transform(images[i], sharp)
        else:
            image = F.to_tensor(images[i])
            label = F.to_tensor(sharp)

        if self.is_test:
            name = image_file
            return image, label, name
        return image, label



def folder_dataloader(image_dir, count=1e5, batch_size=64, num_workers=0, use_transform=True, distributed=False):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    train_dataset = DeblurFolderDataset(image_dir, count=count, transform=transform)

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
