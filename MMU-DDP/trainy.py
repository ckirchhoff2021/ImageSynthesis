import os
import torch

from data.data_loadx import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import distributed
import torch.backends.cudnn as cudnn

from models.perceptual import VGGLoss
from models.gradient import GWLoss


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
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    ####horovod
    dataloader, train_sampler = train_dataloader(args.data_dir, args.train_file, args.train_count, args.batch_size,
                                                 args.num_worker, distributed=True)
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

    writer = SummaryWriter(os.path.join(args.model_save_dir, 'runs')) if hvd.rank() == 0 else None
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

    perceptural_criterion = VGGLoss().to(device)
    gradient_criterion = GWLoss().to(device)

    for epoch_idx in range(epoch, args.num_epoch + 1):
        model.train()

        epoch_timer.tic()
        iter_timer.tic()

        train_sampler.set_epoch(epoch_idx)
        for iter_idx, batch_data in enumerate(dataloader):
            x1, y1, x2, y2, x4, y4 = batch_data
            x1, y1 = x1.to(device), y1.to(device)
            x2, y2 = x2.to(device), y2.to(device)
            x4, y4 = x4.to(device), y4.to(device)

            optimizer.zero_grad()
            pred_img = model(x1, x2, x4)
        
            l1 = criterion(pred_img[0], y4)
            l2 = criterion(pred_img[1], y2)
            l3 = criterion(pred_img[2], y1)
            loss_content = l1+l2+l3

            v1 = perceptural_criterion(pred_img[0], y4)
            v2 = perceptural_criterion(pred_img[1], y2)
            v3 = perceptural_criterion(pred_img[2], y1)
            vgg_loss = v1+v2+v3

            g1 = gradient_criterion(pred_img[0], y4)
            g2 = gradient_criterion(pred_img[1], y2)
            g3 = gradient_criterion(pred_img[2], y1)
            gradient_loss = g1+g2+g3

            label_fft1 = torch.rfft(y4, signal_ndim=2, normalized=False, onesided=False)
            pred_fft1 = torch.rfft(pred_img[0], signal_ndim=2, normalized=False, onesided=False)
            label_fft2 = torch.rfft(y2, signal_ndim=2, normalized=False, onesided=False)
            pred_fft2 = torch.rfft(pred_img[1], signal_ndim=2, normalized=False, onesided=False)
            label_fft3 = torch.rfft(y1, signal_ndim=2, normalized=False, onesided=False)
            pred_fft3 = torch.rfft(pred_img[2], signal_ndim=2, normalized=False, onesided=False)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss = loss_content + 0.1 * loss_fft + vgg_loss + 0.1 * gradient_loss
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


            if (iter_idx + 1) % args.print_freq == 0 and hvd.rank()==0:
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


        if hvd.rank() == 0:
            overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0 and hvd.rank() == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        if hvd.rank() == 0:
            print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch VGG loss: %7.4f Epoch Gradient loss: %7.4f" % (
                epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_vgg_adder.average(), epoch_gradient_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_vgg_adder.reset()
        epoch_gradient_adder.reset()
        
        scheduler.step()

        '''
        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR_GOPRO', val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
        '''

    if hvd.rank() == 0:
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save({'model': model.state_dict()}, save_name)
