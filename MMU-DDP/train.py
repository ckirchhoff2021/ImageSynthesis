import os
import torch

from data import train_dataloader, folder_dataloader
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

from loss import *



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
    '''
    dataloader, train_sampler = train_dataloader(args.data_dir, args.train_file, args.train_count, args.batch_size,
                                                 args.num_worker, distributed=True)
    '''

    dataloader, train_sampler = folder_dataloader(args.data_dir, args.train_count, args.batch_size, args.num_worker, distributed=True)
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
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    vgg_criterion = VGGLoss().to(device)
    gradient_criterion = GWLoss().to(device)

    for epoch_idx in range(epoch, args.num_epoch + 1):
        model.train()

        epoch_timer.tic()
        iter_timer.tic()

        train_sampler.set_epoch(epoch_idx)
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

            loss = loss_content + 0.1 * loss_fft + loss_vgg + 0.1 * loss_gradient

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            if (iter_idx + 1) % args.print_freq == 0 and hvd.rank()==0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

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
            print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
                epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

    if hvd.rank() == 0:
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save({'model': model.state_dict()}, save_name)
