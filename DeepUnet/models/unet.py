import torch
import torch.nn as nn
import torch.nn.functional as F


class Resblock(nn.Module):
    def __init__(self, in_chn):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x) + x


class DownSample(nn.Module):
    def __init__(self, in_chn):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_chn, out_chn, method='upsample'):
        super(UpSample, self).__init__()
        if method == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_chn, in_chn, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_chn, in_chn * 4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_chn, res_num=8):
        super(EncoderBlock, self).__init__()
        resblocks = [Resblock(in_chn) for i in range(res_num)]
        self.encoder = nn.Sequential(*resblocks)

    def forward(self, x):
        return self.encoder(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_chn, res_num=8):
        super(DecoderBlock, self).__init__()
        resblocks = [Resblock(in_chn) for i in range(res_num)]
        self.decoder = nn.Sequential(*resblocks)

    def forward(self, x):
        return self.decoder(x)


class BasicConv(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class DeblurUNet(nn.Module):
    def __init__(self, in_chn=3, n_fea=32):
        super(DeblurUNet, self).__init__()
        self.Extractors = nn.ModuleList([
            BasicConv(in_chn, n_fea),
            BasicConv(in_chn, n_fea),
            BasicConv(in_chn, n_fea * 2)
        ])

        self.Encoder = nn.ModuleList([
            EncoderBlock(n_fea),
            EncoderBlock(n_fea * 2),
            EncoderBlock(n_fea * 4)
        ])

        self.Decoder = nn.ModuleList([
            DecoderBlock(n_fea),
            DecoderBlock(n_fea * 2),
            DecoderBlock(n_fea * 4)
        ])

        self.downs = nn.ModuleList([
            DownSample(n_fea),
            DownSample(n_fea * 2)
        ])

        self.ups = nn.ModuleList([
            UpSample(n_fea * 2, n_fea),
            UpSample(n_fea * 4, n_fea * 2)
        ])

        self.bottom = BasicConv(n_fea * 4, n_fea * 4)
        self.convs = nn.ModuleList([
            nn.Conv2d(n_fea, in_chn, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_fea * 2, in_chn, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_fea * 4, in_chn, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=0.5)
        x4 = F.interpolate(x, scale_factor=0.25)

        f1 = self.Extractors[0](x)
        f2 = self.Extractors[1](x2)
        f4 = self.Extractors[2](x4)

        e1 = self.Encoder[0](f1)
        d1 = self.downs[0](e1)

        z1 = torch.cat([f2, d1], dim=1)
        e2 = self.Encoder[1](z1)

        d2 = self.downs[1](e2)
        z2 = torch.cat([f4, d2], dim=1)
        e3 = self.Encoder[2](z2)

        z3 = self.bottom(e3)
        r3 = self.Decoder[2](z3 + e3)
        y4 = self.convs[2](r3)

        u2 = self.ups[1](r3)
        r2 = self.Decoder[1](u2 + e2)
        y2 = self.convs[1](r2)

        u1 = self.ups[0](r2)
        r1 = self.Decoder[0](u1 + e1)
        y1 = self.convs[0](r1)

        return x4 + y4, x2 + y2, x + y1


if __name__ == '__main__':
    net = DeblurUNet(3, 64)
    x = torch.randn(1,3,256,256)
    y = net(x)
    y1, y2, y3 = y
    print(y1.size())
    print(y2.size())
    print(y3.size())

    import thop
    x = torch.randn(1, 3, 256, 256)
    flops, params = thop.profile(net, inputs=(x, ))
    print(flops, params)


