from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from dataset import DeblurDataset, train_dataloader, test_dataloader, valid_dataloader


from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import distributed
import torch.backends.cudnn as cudnn



# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--model_name', type=str, default='pix2pix')
parser.add_argument('--data_dir', type=str, default='/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro')

parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')


parser.add_argument('--train_file', type=str, default='GoProJson/train.json')
parser.add_argument('--test_file', type=str, default='GoProJson/test.json')
parser.add_argument('--model_save_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/pix2pix/weights/')
parser.add_argument('--result_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/pix2pix/results/')
parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
parser.add_argument('--num_worker', type=int, default=8)


parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--train_count', type=int, default=1e6)
 
# Test
parser.add_argument('--test_model', type=str, default='/data/juicefs_hz_cv_v2/11145199/pix2pix/weights/model.pkl')
parser.add_argument('--test_count', type=int, default=100)

opt = parser.parse_args()
opt.model_save_dir = os.path.join(opt.model_save_dir, opt.model_name, 'weights/')
opt.result_dir = os.path.join(opt.result_dir, opt.model_name, 'result_image/')
print(opt)

if not os.path.exists(opt.model_save_dir):
    os.makedirs(opt.model_save_dir)
if not os.path.exists(opt.result_dir):
    os.makedirs(opt.result_dir)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
data_path = "/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro"
train_json = 'sz300Json/train.json'
test_json = 'sz300Json/test.json'


import horovod.torch as hvd
hvd.init()
torch.cuda.set_device(hvd.local_rank())
cudnn.benchmark = True
####horovod
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic')

if torch.cuda.is_available():
    net_g.cuda()
    net_g = nn.DataParallel(net_g, device_ids=list(range(torch.cuda.device_count()))).cuda()
    
    net_d.cuda()
    net_d = nn.DataParallel(net_d, device_ids=list(range(torch.cuda.device_count()))).cuda()


criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

####horovod
optimizer_g = hvd.DistributedOptimizer(optimizer_g, named_parameters=net_g.named_parameters())
hvd.broadcast_parameters(net_g.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer_g, root_rank=0)

optimizer_d = hvd.DistributedOptimizer(optimizer_d, named_parameters=net_d.named_parameters())
hvd.broadcast_parameters(net_d.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer_d, root_rank=0)

# writer = SummaryWriter(os.path.join(opt.model_save_dir, 'runs')) if hvd.rank() == 0 else None
epoch_g_adder = Adder()
epoch_d_adder = Adder()
    
iter_g_adder = Adder()
iter_d_adder = Adder()
   
epoch_timer = Timer('m')
iter_timer = Timer('m')

training_data_loader, train_sampler = train_dataloader(
    data_path, train_json, opt.train_count, opt.batch_size, opt.num_worker, distributed=True)

max_iter = len(training_data_loader)
print('batches: ', len(training_data_loader))


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    net_g.train()
    epoch_timer.tic()
    iter_timer.tic()
    train_sampler.set_epoch(epoch)

    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################
 
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        
        optimizer_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g = loss_g_gan + loss_g_l1
        
        optimizer_g.synchronize()
        optimizer_g.zero_grad()
        loss_g.backward()
        with optimizer_g.skip_synchronize():
            optimizer_g.step()

        iter_g_adder(loss_g.item())
        iter_d_adder(loss_d.item())

        epoch_g_adder(loss_g.item())
        epoch_d_adder(loss_d.item())

        if hvd.rank() == 0 and iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), iter.item(), loss_g.item()))

            print("Time: %7.4f Epoch: %03d Iter: %4d/%4d Loss_D: %7.4f Loss_G: %7.4f" % (
                iter_timer.toc(), epoch, iteration + 1, max_iter, iter_d_adder.average(), iter_g_adder.average()))

            iter_timer.tic()
            iter_g_adder.reset()
            iter_d_adder.reset()

    if hvd.rank() == 0:
        g_name = os.path.join(opt.model_save_dir, 'model_g.pkl')
        torch.save({'model': net_g.state_dict()}, g_name)

        d_name = os.path.join(opt.model_save_dir, 'model_d.pkl')
        torch.save({'model': net_d.state_dict()}, d_name)

    if hvd.rank() == 0 and epoch % 10 == 0:
        print("EPOCH: %02d Elapsed time: %4.2f Epoch g_Loss: %7.4f Epoch d_Loss: %7.4f" % (
            epoch, epoch_timer.toc(), epoch_g_adder.average(), epoch_d_adder.average()))

    epoch_d_adder.reset()
    epoch_g_adder.reset()
            
    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)


    #checkpoint
    if hvd.rank() == 0:
        g_name = os.path.join(opt.model_save_dir, 'FinalG.pkl')
        torch.save({'model': net_g.state_dict()}, g_name)

        d_name = os.path.join(opt.model_save_dir, 'FinalD.pkl')
        torch.save({'model': net_d.state_dict()}, d_name)
