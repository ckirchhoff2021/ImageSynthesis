import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import csv
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

def csv_reader(root, file):
    fstream = open(os.path.join(root, file), 'r')
    reader = csv.reader(fstream)
    datas = list()
    for x in reader:
        x1, x2 = x
        y1 = os.path.join(root, x1)
        y2 = os.path.join(root, x2)
        datas.append([y1, y2])
    return datas


class FFHQDataV1(Dataset):
    def __init__(self):
        super(FFHQDataV1, self).__init__()
        self.root = '/data/juicefs_hz_cv_v2/public_data/low_light_enhancement/esrgan/train_data/'
        self.file = 'train.csv'
        self.transform = transform
        self.datas = csv_reader(self.root, self.file)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        f = data[1]
        x = Image.open(f)
        y = self.transform(x)
        return y, f


if __name__ == '__main__':
    ffhq_datas = FFHQDataV1()
    print('==> Num: ', len(ffhq_datas))
    x, _ = ffhq_datas[12]
    print(x.size())