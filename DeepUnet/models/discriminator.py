import torch
import torch.nn as nn
from .layers import initialize_weights


class Discriminator256(nn.Module):
    def __init__(self, num_channel=3):
        super(Discriminator256, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.02)

        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)   # 128
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)   # 64
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)   # 32
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)   # 16
        self.bn8 = nn.BatchNorm2d(512)

        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8], 0.1)

    def forward(self, x):
        y_conv1 = self.leaky_relu(self.conv1(x))
        y_conv2 = self.bn2(self.leaky_relu(self.conv2(y_conv1)))
        y_conv3 = self.bn3(self.leaky_relu(self.conv3(y_conv2)))
        y_conv4 = self.bn4(self.leaky_relu(self.conv4(y_conv3)))
        y_conv5 = self.bn5(self.leaky_relu(self.conv5(y_conv4)))
        y_conv6 = self.bn6(self.leaky_relu(self.conv6(y_conv5)))
        y_conv7 = self.bn7(self.leaky_relu(self.conv7(y_conv6)))
        y_conv8 = self.bn8(self.leaky_relu(self.conv8(y_conv7)))

        y_avg = self.avg_pool(y_conv8)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_fc1 = self.leaky_relu(self.fc1(y_avg))
        y_fc2 = self.fc2(y_fc1)
        y_out = self.sigmoid(y_fc2)

        return y_out


class Discriminator128(nn.Module):
    def __init__(self, num_channel=3):
        super(Discriminator128, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.02)

        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)  # 64
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)  # 32
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)  # 16
        self.bn6 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6], 0.1)

    def forward(self, x):
        y_conv1 = self.leaky_relu(self.conv1(x))
        y_conv2 = self.bn2(self.leaky_relu(self.conv2(y_conv1)))
        y_conv3 = self.bn3(self.leaky_relu(self.conv3(y_conv2)))
        y_conv4 = self.bn4(self.leaky_relu(self.conv4(y_conv3)))
        y_conv5 = self.bn5(self.leaky_relu(self.conv5(y_conv4)))
        y_conv6 = self.bn6(self.leaky_relu(self.conv6(y_conv5)))

        y_avg = self.avg_pool(y_conv6)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_fc1 = self.leaky_relu(self.fc1(y_avg))
        y_fc2 = self.fc2(y_fc1)
        y_out = self.sigmoid(y_fc2)

        return y_out


class Discriminator64(nn.Module):
    def __init__(self, num_channel=3):
        super(Discriminator64, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.02)

        self.conv1 = nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)  # 32
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)  # 16
        self.bn4 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        y_conv1 = self.leaky_relu(self.conv1(x))
        y_conv2 = self.bn2(self.leaky_relu(self.conv2(y_conv1)))
        y_conv3 = self.bn3(self.leaky_relu(self.conv3(y_conv2)))
        y_conv4 = self.bn4(self.leaky_relu(self.conv4(y_conv3)))

        y_avg = self.avg_pool(y_conv4)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_fc1 = self.leaky_relu(self.fc1(y_avg))
        y_fc2 = self.fc2(y_fc1)
        y_out = self.sigmoid(y_fc2)

        return y_out



class PatchDiscriminator(nn.Module):
    '''
    1, in_chn, 16, 16
    '''
    def __init__(self, in_chn):
        super(PatchDiscriminator, self).__init__() 
        self.feature = nn.Sequential(
            nn.Conv2d(in_chn, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.feature(x)
        y2 = y1.view(y1.size(0), -1)
        y = self.fc(y2)
        return y



if __name__ == '__main__':
    d256 = Discriminator256()
    x1 = torch.randn(1,3,256,256)
    y1 = d256(x1)
    print(y1.size())

    d128 = Discriminator128()
    x2 = torch.randn(1, 3, 128, 128)
    y2 = d256(x2)
    print(y2.size())

    d64 = Discriminator64()
    x3 = torch.randn(1, 3, 64, 64)
    y3 = d64(x3)
    print(y3.size())

    dp = PatchDiscriminator(3)
    x4 = torch.randn(1, 3, 16, 16)
    y4 = dp(x4)
    print('patch:', y4.size())