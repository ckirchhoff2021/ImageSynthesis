import os
import torch
import argparse
from torch.backends import cudnn
from models.unet import DeblurUNet
from models.autoencoder import AutoEncoder


from train2 import _train
from eval import _eval

import torch.nn as nn


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # model = DeblurUNet()
    model = AutoEncoder(dw=True)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='AutoEncoder', choices=['DeblurUnet', 'AutoEncoder'], type=str)
    parser.add_argument('--data_dir', type=str, default='/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen')
    parser.add_argument('--train_file', type=str, default='hq_train.json')
    parser.add_argument('--test_file', type=str, default='hq_test.json')
    parser.add_argument('--model_save_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/deblur/results/')
    parser.add_argument('--result_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/deblur/results/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_count', type=int, default=1e8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])
    parser.add_argument('--vgg_weights', type=str, default='/data/juicefs_hz_cv_v3/11145199/pretrained/vgg16-397923af.pth')

    # Test
    parser.add_argument('--test_model', type=str, default='/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/11145199/results/weights/model.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    parser.add_argument('--test_count', type=int, default=64)

    args = parser.parse_args()
    args.model_save_dir = os.path.join(args.model_save_dir, args.model_name, 'weights/')
    args.result_dir = os.path.join(args.result_dir, args.model_name, 'result_image/')
    print(args)
    main(args)
