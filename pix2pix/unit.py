import torch
from networks import *

def main():
    pass


if __name__ == '__main__':
    net_g = define_G(3, 3, 64, 'batch', False, 'normal', 0.02)
    net_d = define_D(3 + 3, 64, 'basic')

    x = torch.randn(1, 3, 300, 300)
    y = net_g(x)
    print(y.size())
