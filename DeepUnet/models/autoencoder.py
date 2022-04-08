import torch
import torch.nn as nn


def depthwise_conv(in_chn, out_chn, kernel_size=3, stride=1):
   return nn.Sequential(
       nn.Conv2d(in_chn, in_chn, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_chn, bias=True),
       nn.ReLU(inplace=True),
       nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False),
       nn.ReLU(inplace=True)
   )

def conv(in_chn, out_chn, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.ReLU(inplace=True)
    )

class Resblock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(Resblock, self).__init__()
        self.conv1 = conv(in_chn, out_chn)
        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        return x + y2


class DepthwiseResblock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(DepthwiseResblock, self).__init__()
        self.conv1 = depthwise_conv(in_chn, out_chn)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        return x + y2


class EncoderBlocks(nn.Module):
    def __init__(self, in_chn, out_chn, res_num=3, dw=False):
        super(EncoderBlocks, self).__init__()
        if dw:
            layers = [DepthwiseResblock(in_chn, in_chn) for i in range(res_num)]
            conv_out = depthwise_conv(in_chn, out_chn, kernel_size=4, stride=2)
        else:
            layers = [Resblock(in_chn, in_chn) for i in range(res_num)]
            conv_out = conv(in_chn, out_chn, kernel_size=4, stride=2)

        self.main = nn.Sequential(*layers)
        self.conv_out = conv_out

    def forward(self, x):
        y = self.main(x)
        return self.conv_out(y)


class DecoderBlocks(nn.Module):
    def __init__(self, in_chn, out_chn, res_num=3, dw=False):
        super(DecoderBlocks, self).__init__()
        if dw:
            layers = [DepthwiseResblock(in_chn, in_chn) for i in range(res_num)]
            conv_out = depthwise_conv(in_chn, out_chn)
        else:
            layers = [Resblock(in_chn, in_chn) for i in range(res_num)]
            conv_out = conv(in_chn, out_chn)

        self.upsample = nn.Upsample(scale_factor=2)
        self.main = nn.Sequential(*layers)
        self.conv_out = conv_out

    def forward(self, x):
        y = self.main(x)
        y1 = self.upsample(y)
        y2 = self.conv_out(y1)
        return y2


class AutoEncoder(nn.Module):
    def __init__(self, dw=True):
        super(AutoEncoder, self).__init__()
        self.dw = dw
        if self.dw:
            self.conv0 = depthwise_conv(3, 64)
        else:
            self.conv0 = conv(3, 64)

        self.encoders = nn.ModuleList([
            EncoderBlocks(64, 128, dw=self.dw),
            EncoderBlocks(128, 256, dw=self.dw),
            EncoderBlocks(256, 512, dw=self.dw),
            EncoderBlocks(512, 512, dw=self.dw)
        ])

        self.decoders = nn.ModuleList([
            DecoderBlocks(512, 512, dw=self.dw),
            DecoderBlocks(512, 256, dw=self.dw),
            DecoderBlocks(256, 128, dw=self.dw),
            DecoderBlocks(128, 64, dw=self.dw)
        ])

        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y0 = self.conv0(x)
        e0 = self.encoders[0](y0)
        e1 = self.encoders[1](e0)
        e2 = self.encoders[2](e1)
        e3 = self.encoders[3](e2)

        d0 = self.decoders[0](e3)
        d1 = self.decoders[1](d0)
        d2 = self.decoders[2](d1)
        d3 = self.decoders[3](d2)

        y1 = self.conv1(d3)
        return x + y1


if __name__ == '__main__':

    import thop
    x = torch.randn(1, 3, 256, 256)
    net = AutoEncoder(dw=True)
    flops, params = thop.profile(net, inputs=(x, ))
    print(flops, params)
