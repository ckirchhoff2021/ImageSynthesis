import torch
import torch.nn as nn
from networks import define_G

from torchvision.transforms import functional as F

from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio

from collections import OrderedDict


import cv2
import os
import numpy as np
import torch.nn.functional as F1
from tqdm import tqdm


# blur_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/input/'
# sharp_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/target/'

blur_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/11145199/szface_300/input/'
sharp_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/11145199/szface_300/target/'


def main():   
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/pix2pix/weights/netG_epoch_190.pth'
    save_path = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/pix2pixGen'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:0")
    net_g = define_G(3, 3, 64, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_g.cuda()

    state = torch.load(model_file)
    net_g.load_state_dict(state)
    net_g.eval()
    
    files = os.listdir(blur_folder)

    psnrs = list()

    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(blur_folder, file_name)
        sharp_file = os.path.join(sharp_folder, file_name)
        
        blur = Image.open(blur_file)
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()

        sharp = Image.open(sharp_file)
        sharp_tensor = F.to_tensor(sharp).unsqueeze(0)

        merge = Image.new('RGB', (900, 300))
        merge.paste(blur, (0, 0, 300, 300))
        merge.paste(sharp, (300, 0, 600, 300))

        with torch.no_grad():
            output = net_g(blur_tensor)
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
            label_numpy = sharp_tensor.squeeze(0).numpy()

            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (600, 0, 900, 300))

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            print('==> PSNR: %.2f ' % psnr)

            merge.save(os.path.join(save_path, file_name))
            psnrs.append(psnr)

    print('average base: ', np.mean(psnrs))





if __name__ == '__main__':
    main()