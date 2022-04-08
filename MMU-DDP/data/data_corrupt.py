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
import cv2


def corrupt_dataloader(path, image_file='GoProJson/train_data.json', count=1e5, batch_size=64, size=512, num_workers=0, corruption=True, distributed=False):
    train_dataset = CorruptDataset(path, image_file, count=count, corruption=corruption, size=size)

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



class CorruptDataset(Dataset):
    def __init__(self, image_dir, data_json, count=1e5, corruption=True, size=512, is_test=False):
        self.count = count
        self.image_dir = image_dir
        self.image_list = json.load(open(os.path.join(image_dir, data_json), 'r'))
        random.shuffle(self.image_list)

        cnt = len(self.image_list)
        self.count = self.count if self.count < cnt else cnt
       
        self.is_test = is_test
        self.corruption = corruption

        self.normalize_fn = get_normalize()
        self.corrupt_fn = get_corrupt_function()
        self.transform_fn = get_transforms(size)
        print('==> Action: image_dir = {}, count = {}'.format(self.image_dir, self.count))


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        b, s = self.image_list[idx]
        
        image = cv2.imread(b)
        label = cv2.imread(s)

        src, dst = self.transform_fn(image, label)
        if self.corruption:
            src = self.corrupt_fn(src)
        x, y = self.normalize_fn(src, dst)
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        
        if self.is_test:
            name = self.image_list[idx][0]
            return x, y, name

        return x, y



class CorruptFolderDataset(Dataset):
    def __init__(self, image_dir, count=1e5, size=256, corruption=True, is_test=False, is_random=True):
        super(CorruptFolderDataset, self).__init__()
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

        self.is_test = is_test
        self.is_random = is_random
        self.corruption = corruption

        self.normalize_fn = get_normalize()
        self.corrupt_fn = get_corrupt_function()
        self.transform_fn = get_transforms(size)
        print('==> Action: image_dir = {}, count = {}'.format(self.image_dir, self.count))


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        image_file = self.image_list[idx]
        src = cv2.imread(image_file)
        blur = src[:,:300, :]
        sharp = src[:,300:600,:]
        pix2pix = src[:,600:,:]

        images = [blur, pix2pix]
        if self.is_random:
            i = np.random.randint(2)
        else:
            i = 0

        src, dst = self.transform_fn(images[i], sharp)
        if self.corruption:
            src = self.corrupt_fn(src)
        x, y = self.normalize_fn(src, dst)
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        if self.is_test:
            name = image_file
            return x, y, name
        return x, y



def corruptFolder_dataloader(image_dir, count=1e5, batch_size=64, num_workers=0, size=256, distributed=False):
    train_dataset = CorruptFolderDataset(image_dir, count=count, size=size)

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



