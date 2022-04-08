import torch
print('xxx')
print(torch.cuda.is_available())

from data import DeblurDataset
from models.MIMOUNet import *
from models.discriminator import PatchDiscriminator
from data.data_corrupt import CorruptDataset, CorruptFolderDataset

import json


def corrupt_test():
    data = CorruptDataset('/data/juicefs_hz_cv_v3/11145199/datas/', 'hq_test.json')
    print('Num: ', len(data))

    x, y = data[2]
    print(x)
    print(x.shape)
    print(y.shape)


def corrupt_test2():
    data = CorruptFolderDataset('/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen')
    print('Num: ', len(data))

    x, y = data[2]
    print(x.shape)
    print(y.shape)




if __name__ == '__main__':
    '''
    image_dir = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/'
    train_data = 'train_data.json'
    test_data = 'test_data.json'

    train_deblur = DeblurDataset(image_dir, train_data)
    print(len(train_deblur))

    test_deblur = DeblurDataset(image_dir, test_data)
    print(len(test_deblur))
    '''

    '''
    import time
    x = torch.randn(4,3,300,300)
    start = time.time()
    n2 = MIMOUNetRRDB(num_res=1)
    y2 = n2(x)
    end = time.time()
    print('time: ', end-start)

    print(y2[0].size())
    print(y2[1].size())
    print(y2[2].size())
    '''

    '''
    dp = PatchDiscriminator(3)
    x4 = torch.randn(1, 3, 16, 16)
    y4 = dp(x4)
    print('patch:', y4.size())


    gu = GradientDeblurGAN()
    gu.cuda()
    x1 = torch.randn(1, 3, 256, 256).cuda()
    x4, x2, x1, g1, g2 = gu(x1)
    print('patch:', x1.size())
    '''

    # ms4 = MS4UNet()
    # x1 = torch.randn(1,3,256,256)
    # y1 = ms4(x1)
    # k1, k2, k3, k4 = y1
    # print(k4.size())

    corrupt_test()
    # corrupt_test2()