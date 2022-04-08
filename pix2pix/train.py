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
from dataset import DeblurDataset, train_dataloader, test_dataloader, valid_dataloader, DeblurFolderDataset

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
parser.add_argument('--model_save_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/deblur')
parser.add_argument('--result_dir', type=str, default='/data/juicefs_hz_cv_v2/11145199/deblur')
parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)


parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_count', type=int, default=1e6)
 
# Test
parser.add_argument('--test_model', type=str, default='/data/juicefs_hz_cv_v2/11145199/pix2pix/weights/model.pkl')
parser.add_argument('--test_count', type=int, default=128)

parser.add_argument('--resume', type=bool, default=False)


opt = parser.parse_args()
opt.model_save_dir = os.path.join(opt.model_save_dir, opt.model_name, 'all/weights/')
opt.result_dir = os.path.join(opt.result_dir, opt.model_name, 'result_image/')
print(opt)


if not os.path.exists(opt.model_save_dir):
    os.makedirs(opt.model_save_dir)
if not os.path.exists(opt.result_dir):
    os.makedirs(opt.result_dir)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
data_path = "/data/juicefs_hz_cv_v3/11145199/datas/pix2pixGen"

training_data_loader = train_dataloader(data_path, count=opt.train_count, batch_size=opt.batch_size)
testing_data_loader = valid_dataloader(data_path, count=opt.test_count, batch_size=opt.batch_size//2) 

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

if opt.resume:
    g_model_file = os.path.join(opt.model_save_dir, 'netG.pth')
    d_model_file = os.path.join(opt.model_save_dir, 'netD.pth')
    g_state = torch.load(g_model_file)
    net_g.load_state_dict(g_state)
    d_state = torch.load(d_model_file)
    net_d.load_state_dict(d_state)


criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)


print('batches: ', len(training_data_loader))
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
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

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)


    g_model_file = os.path.join(opt.model_save_dir, 'netG.pth')
    d_model_file = os.path.join(opt.model_save_dir, 'netD.pth')
    torch.save(net_g.state_dict(), g_model_file)
    torch.save(net_d.state_dict(), d_model_file)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 5 == 0:
        g_model_file = os.path.join(opt.model_save_dir, 'netG_epoch_{}.pth'.format(epoch))
        d_model_file = os.path.join(opt.model_save_dir, 'netD_epoch_{}.pth'.format(epoch))
        torch.save(net_g.state_dict(), g_model_file)
        torch.save(net_d.state_dict(), d_model_file)

        print("Checkpoint saved to {}".format("checkpoint-" + str(epoch)))
