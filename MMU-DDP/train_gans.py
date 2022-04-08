import os
from sys import prefix
import torch
import torch.nn as nn

from data import train_dataloader, folder_dataloader
from utils import Adder, Timer, check_lr, save_models
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F

from models.perceptual import VGGLoss
from models.gradient import GWLoss

from models.discriminator import Discriminator256, Discriminator128, Discriminator64
from loss import *
from models.layers import freeze, unfreeze

from eval import _eval
from torch.backends import cudnn
from models.MIMOUNet import build_net

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import distributed
import torch.backends.cudnn as cudnn



def _train(model, args):
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True
    ####horovod
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    ####horovod
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(prefix='genrator'))
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    ####horovod
    # dataloader, train_sampler = train_dataloader(args.data_dir, args.train_file, args.train_count, args.batch_size, args.num_worker, distributed=True)
    dataloader, train_sampler = folder_dataloader(args.data_dir, args.train_count, args.batch_size, args.num_worker, distributed=True)
    max_iter = len(dataloader)
    print('==> max iterations: ', max_iter)
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


    writer = SummaryWriter(args.model_save_dir)
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    epoch_vgg_adder = Adder()
    epoch_gradient_adder = Adder()
    epoch_adv_adder = Adder()

    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    iter_vgg_adder = Adder()
    iter_gradient_adder = Adder()
    iter_adv_adder = Adder()

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
    D256 = nn.DataParallel(D256, device_ids=list(range(torch.cuda.device_count()))).cuda()
    D128.cuda()
    D128 = nn.DataParallel(D128, device_ids=list(range(torch.cuda.device_count()))).cuda()
    D64.cuda()
    D64 = nn.DataParallel(D64, device_ids=list(range(torch.cuda.device_count()))).cuda()

    d256_optimizer = hvd.DistributedOptimizer(d256_optimizer, named_parameters=D256.named_parameters(prefix='discriminator256'))
    hvd.broadcast_parameters(D256.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(d256_optimizer, root_rank=0)

    d128_optimizer = hvd.DistributedOptimizer(d128_optimizer, named_parameters=D128.named_parameters(prefix='discriminator128'))
    hvd.broadcast_parameters(D128.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(d128_optimizer, root_rank=0)

    d64_optimizer = hvd.DistributedOptimizer(d64_optimizer, named_parameters=D64.named_parameters(prefix='discriminator64'))
    hvd.broadcast_parameters(D64.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(d64_optimizer, root_rank=0)

    model.train()
    D256.train()
    D128.train()
    D64.train()

    vgg_criterion = VGGLoss(model_file=args.vgg_weights).to(device)
    gradient_criterion = GWLoss().to(device)
    adv_criterion = nn.BCELoss()

    for epoch_idx in range(epoch, args.num_epoch + 1):
        train_sampler.set_epoch(epoch_idx)
        epoch_timer.tic()
        iter_timer.tic()

        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            pred_img = model(input_img)
            x4, x2, x1 = pred_img
            y1 = label_img
            y2 = F.interpolate(y1, scale_factor=0.5, mode='bilinear')
            y4 = F.interpolate(y1, scale_factor=0.25, mode='bilinear')

            loss_content = content_loss(criterion, x1, x2, x4, y1, y2, y4)
            loss_fft = fft_loss(criterion, x1, x2, x4, y1, y2, y4)
            loss_vgg = perceptual_loss(vgg_criterion, x1, x2, x4, y1, y2, y4)
            loss_gradient = gradient_loss(gradient_criterion, x1, x2, x4, y1, y2, y4)

            mini_batch = input_img.size(0)
            real_label = torch.ones(mini_batch, 1).to(device)
            fake_label = torch.zeros(mini_batch, 1).to(device)

            fake_probs256 = D256(x1)
            fake_probs128 = D128(x2)
            fake_probs64 = D64(x4)

            loss_g1 = adv_criterion(fake_probs256, real_label)
            loss_g2 = adv_criterion(fake_probs128, real_label)
            loss_g4 = adv_criterion(fake_probs64, real_label)

            loss_g_adv = loss_g1 + loss_g2 + loss_g4
            loss = loss_content + 0.1 * loss_fft + loss_g_adv * 0.005 + loss_vgg + loss_gradient * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            freeze(model)
            fake_imgs = model(input_img)
            x4, x2, x1 = fake_imgs
           
            d256_optimizer.synchronize()
            fake_probs256 = D256(x1)
            real_probs256 = D256(y1)
            loss_d1 = adversarial_loss(adv_criterion, real_probs256, fake_probs256, real_label, fake_label)
            d256_optimizer.zero_grad()
            loss_d1.backward() 
            d256_optimizer.step()


            d128_optimizer.synchronize()
            fake_probs128 = D128(x2)
            real_probs128 = D128(y2)
            loss_d2 = adversarial_loss(adv_criterion, real_probs128, fake_probs128, real_label, fake_label)
            d128_optimizer.zero_grad()
            loss_d2.backward()
            d128_optimizer.step()

            d64_optimizer.synchronize()
            fake_probs64 = D64(x4)
            real_probs64 = D64(y4)
            loss_d4 = adversarial_loss(adv_criterion, real_probs64, fake_probs64, real_label, fake_label)
            d64_optimizer.zero_grad()
            loss_d4.backward()
            # with d64_optimizer.skip_synchronize():
            d64_optimizer.step()

            optimizer.synchronize()
            unfreeze(model)

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            iter_adv_adder(loss_g_adv.item())
            iter_vgg_adder(loss_vgg.item())
            iter_gradient_adder(loss_gradient.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_adv_adder(loss_g_adv.item())
            epoch_vgg_adder(loss_vgg.item())
            epoch_gradient_adder(loss_gradient.item())


            if (iter_idx + 1) % args.print_freq == 0 and hvd.rank() == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Loss VGG: %7.4f Loss Gradient %7.4f Loss adv %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(), iter_fft_adder.average(),
                    iter_vgg_adder.average(), iter_gradient_adder.average(), iter_adv_adder.average()))

                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                writer.add_scalar('VGG Loss', iter_vgg_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('Gradient Loss', iter_gradient_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('Adversarial Loss', iter_adv_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_vgg_adder.reset()
                iter_gradient_adder.reset()
                iter_adv_adder.reset()
        
        if hvd.rank() == 0:
            save_models([model, D256, D128, D64], args.model_save_dir, args.model_name)

        if epoch_idx % args.save_freq == 0 and hvd.rank() == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'name': args.model_name,
                        'epoch': epoch_idx}, save_name)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch VGG loss: %7.4f Epoch Gradient loss: %7.4f Epoch Adv loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_vgg_adder.average(), epoch_gradient_adder.average(), epoch_adv_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_vgg_adder.reset()
        epoch_gradient_adder.reset()
        epoch_adv_adder.reset()

        scheduler.step()

        '''
        if epoch_idx % args.valid_freq == 0 and hvd.rank() == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR', val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
        '''

    if hvd.rank() == 0:
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save({'model': model.state_dict()}, save_name)


