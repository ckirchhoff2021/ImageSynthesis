import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import Generator

import torch
import torch.nn as nn

from torchvision.transforms import functional as F
from PIL import Image

from torchvision.utils import save_image
from torchvision import transforms


def infer_faceHQ():
    
    netG_A2B = Generator(3, 3)
    netG_B2A = Generator(3, 3)

    a2b_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/cycleGANHQ/netG_A2B.pth'
    b2a_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/cycleGANHQ/netG_B2A.pth'

    state = torch.load(a2b_file)
    netG_A2B.load_state_dict(state)

    state = torch.load(b2a_file)
    netG_B2A.load_state_dict(state)

    netG_A2B.eval()
    netG_B2A.eval()

    outdir = 'output/runHQ'
    os.makedirs(outdir, exist_ok=True)

    netG_A2B.cuda()
    netG_B2A.cuda()

    import json
    test_file = '/data/juicefs_hz_cv_v3/11145199/datas/ffhq/ffhq_test.json'
    files = json.load(open(test_file, 'r'))
    print('Test Num: ', len(files))


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    from tqdm import tqdm
    for file in tqdm(files):
        b, s = file
        name = b.split('/')[-1]
        A = Image.open(b)
        B = Image.open(s)

        a1 = transform(A)
        b1 = transform(B)
        a1, b1 = a1.unsqueeze(0).cuda(), b1.unsqueeze(0).cuda()
        
        b_out = netG_A2B(a1)
        a_out = netG_B2A(b1)

        b_out += 0.5 / 255.0
        a_out += 0.5 / 255.0

        b_out = torch.clamp(b_out, 0, 1).cpu()
        a_out = torch.clamp(a_out, 0, 1).cpu()

        _, _, w, h = b_out.size()
        out = Image.new('RGB', (w*2, h*2))

        A = A.resize((w, h))
        B = B.resize((w, h))
        
        out.paste(A, (0, 0, w, h))
        out.paste(B, (0, h, w, h*2))

        A2B = F.to_pil_image(b_out.squeeze(), 'RGB')
        B2A = F.to_pil_image(a_out.squeeze(), 'RGB')

        out.paste(A2B, (w, 0, w*2, h))
        out.paste(B2A, (w, h, w*2, h*2))

        out.save(os.path.join(outdir, name))

    print('Done ...')



def simplify_gen2():
    netG_A2B = Generator(3, 3)
    a2b_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/cycleGANHQ/netG_A2B.pth'
    state = torch.load(a2b_file)
    netG_A2B.load_state_dict(state)
    netG_A2B.cuda()
    netG_A2B.eval()

    outdir = 'output/run'
    os.makedirs(outdir, exist_ok=True)
    
    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    files = os.listdir(folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        file = os.path.join(folder, file_name)
        x = Image.open(file)
        a = F.to_tensor(x).unsqueeze(0)
        a = a.cuda()
        y = netG_A2B(a)

        y += 0.5 / 255.0
        y = torch.clamp(y, 0, 1).cpu()
        A2B = F.to_pil_image(y.squeeze(), 'RGB')

        w, h = A2B.size
        out = Image.new('RGB', (w*2, h))
        x = x.resize((w, h))
        out.paste(x, (0, 0, w, h))
        A2B = F.to_pil_image(y.squeeze(), 'RGB')
        out.paste(A2B, (w, 0, w*2, h))
      
        out.save(os.path.join(outdir, file_name))


    print('Done ......')



if __name__ == '__main__':
    infer_faceHQ()
    # simplify_gen2()