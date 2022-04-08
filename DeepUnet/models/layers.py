import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            # padding = kernel_size // 2 -1
            # layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))

            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=True))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResidualDenseBlock5(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock5, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x



class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5(nf, gc)
        self.RDB2 = ResidualDenseBlock5(nf, gc)
        self.RDB3 = ResidualDenseBlock5(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock5V2(nn.Module):
    def __init__(self, nf, out_chn, gc=32, bias=True):
        super(ResidualDenseBlock5V2, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv6 = nn.Conv2d(nf, out_chn, 3, 1, 1, bias=bias)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = x5 * 0.2 + x
        x7 = self.relu(self.conv6(x6))
        return x7


class FFTResBlock(nn.Module):
    def __init__(self, in_channel): 
        super(FFTResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, in_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channel, in_channel, kernel_size=3, stride=1, relu=False)
        )
        self.fft = nn.Sequential(
            BasicConv(in_channel * 2, in_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv(in_channel * 2, in_channel * 2, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.rfft(x, 2, normalized=False, onesided=False)
        
        y_real = y[:,:,:,:,0]
        y_imag = y[:,:,:,:,1]
        y1 = torch.cat([y_real, y_imag], dim=1)
        y_fft = self.fft(y1)
        
        y_real, y_imag = torch.chunk(y_fft, 2, dim=1)
        y_real = y_real.unsqueeze(4)
        y_imag = y_imag.unsqueeze(4)

        y2 = torch.cat([y_real, y_imag], dim=4)
        y3 = torch.irfft(y2, 2, normalized=False, onesided=False, signal_sizes=(H, W))
        return self.main(x) + x + y3



def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False