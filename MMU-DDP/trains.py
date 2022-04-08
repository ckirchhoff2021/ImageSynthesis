import os
import torch
import torch.nn as nn

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F

from models.perceptual import VGGLoss
from models.gradient import GWLoss

from models.discriminator import Discriminator256, Discriminator128, Discriminator64
from loss import *



def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.train_file, args.train_count, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter(args.result_dir)
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_vgg_adder = Adder()
    epoch_gradient_adder = Adder()

    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_vgg_adder = Adder()
    iter_gradient_adder = Adder()

    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1


    D256 = Discriminator256(3)
    d256_optimizer = torch.optim.Adam(D256.parameters(), lr=5e-4, betas=(0.9,0.999))
    D128 = Discriminator128(3)
    d128_optimizer = torch.optim.Adam(D128.parameters(), lr=5e-4, betas=(0.9, 0.999))
    D64 = Discriminator64(3)
    d64_optimizer = torch.optim.Adam(D64.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    D256.cuda()
    D128.cuda()
    D64.cuda()
    model.train()
    D256.train()
    D128.train()
    D64.train()

    vgg_criterion = VGGLoss().to(device)
    gradient_criterion = GWLoss().to(device)
    adv_criterion = nn.BCELoss().to(device)


    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            v1 = perceptural_criterion(pred_img[0], label_img4)
            v2 = perceptural_criterion(pred_img[1], label_img2)
            v3 = perceptural_criterion(pred_img[2], label_img)
            vgg_loss = v1+v2+v3

            g1 = gradient_criterion(pred_img[0], label_img4)
            g2 = gradient_criterion(pred_img[1], label_img2)
            g3 = gradient_criterion(pred_img[2], label_img)
            gradient_loss = g1+g2+g3


            label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)
            pred_fft1 = torch.rfft(pred_img[0], signal_ndim=2, normalized=False, onesided=False)
            label_fft2 = torch.rfft(label_img2, signal_ndim=2, normalized=False, onesided=False)
            pred_fft2 = torch.rfft(pred_img[1], signal_ndim=2, normalized=False, onesided=False)
            label_fft3 = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
            pred_fft3 = torch.rfft(pred_img[2], signal_ndim=2, normalized=False, onesided=False)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss = loss_content + 0.1 * loss_fft + 0.1 * vgg_loss + 0.1 * gradient_loss
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            iter_vgg_adder(vgg_loss.item())
            iter_gradient_adder(gradient_loss.item())

            epoch_vgg_adder(vgg_loss.item())
            epoch_gradient_adder(gradient_loss.item())


            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Loss VGG: %7.4f Loss Gradient %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(), iter_fft_adder.average(),
                    iter_vgg_adder.average(), iter_gradient_adder.average()))

                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                writer.add_scalar('VGG Loss', iter_vgg_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('Gradient Loss', iter_gradient_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_vgg_adder.reset()
                iter_gradient_adder.reset()


        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch VGG loss: %7.4f Epoch Gradient loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_vgg_adder.average(), epoch_gradient_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_vgg_adder.reset()
        epoch_gradient_adder.reset()

        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR', val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))

    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
