import torch
import pickle
import cv2
import numpy as np


def main():
    with open('pretrained/ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    print(G)
    print('dim: ', G.z_dim)
    # G = G.float()
    z = torch.randn([1, G.z_dim]).cuda()
    c = None
    img = G(z, c, force_fp32=True)
    # img = G(z, c)
    print(img.size())

    x = img * 0.5 + 0.5
    y = torch.clamp(x[0], 0, 1)
    y = y.numpy() * 255.0
    y = y.transpose(1,2,0)
    y = y.astype(np.uint8)
    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs/rand.png', y)



if __name__ == '__main__':
    main()