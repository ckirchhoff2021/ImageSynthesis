import torch
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator


if __name__ == '__main__':
    G = ESRGAN(3,3)
    D = Discriminator()

    x = torch.randn(1,3,256,256)
    y1 = G(x)
    print(y1.size())
    y2 = D(x)
    print(y2.size())
