import torch
import torch.nn as nn
from model.ESRGAN import ESRGAN

import os
import cv2
import numpy as np

from PIL import Image
import torch.nn.functional as F1

from torchvision.transforms import functional as F
 

def generate():
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/esrganHQ/generator_170.pth'
    model = ESRGAN(3, 3, 64)
    model.cuda()

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.eval()

    file = '/data/juicefs_hz_cv_v3/11145199/work/ffhq.jpg'
    name = file.split('/')[-1]

    x = cv2.imread(file)
    _, w, _ = x.shape
    b = x[:,:w//2,:]
    t = x[:,w//2:,:]

    blur = (b / 255.0 - 0.5) / 0.5
    blur = b / 255.0
    blur = np.transpose(blur, (2, 0, 1))
    blur_tensor = torch.from_numpy(blur).unsqueeze(0).float()
    blur_tensor = blur_tensor.cuda()

    with torch.no_grad():  
        pred = model(blur_tensor)
        y = pred[0]
        # y = y * 0.5 + 0.5
        y = torch.clamp(y, 0, 1).cpu().squeeze().numpy()
        y = y * 255.0
        g = np.transpose(y, (1, 2, 0)).astype(np.uint8)
        merge = np.concatenate([b, t, g], axis=1)
        cv2.imwrite(os.path.join('outputs/hq', name), merge)


@torch.no_grad()
def generateHQ(outdir):
    import json
    test_file = '/data/juicefs_hz_cv_v3/11145199/datas/ffhq/ffhq_test.json'
    files = json.load(open(test_file, 'r'))
    print('Test Num: ', len(files))

    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/esrganHQ/generator_299.pth'
    model = ESRGAN(3, 3, 64)
    model.cuda()

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    from tqdm import tqdm
    for file in tqdm(files):
        b, s = file
        name = b.split('/')[-1]
        A = Image.open(b)
        B = Image.open(s)
       
        x = transform(A)
        x = x.unsqueeze(0).cuda()
        pred = model(x)
        y = pred[0]
        y += 0.5/255
        y = torch.clamp(y, 0, 1).cpu()

        _, w, h = y.size()
        out = Image.new('RGB', (w*3, h))

        A = A.resize((w, h))
        B = B.resize((w, h))
        
        out.paste(A, (0, 0, w, h))
        out.paste(B, (w, 0, w*2, h))

        C = F.to_pil_image(y.squeeze(), 'RGB')
        out.paste(C, (w*2, 0, w*3, h))

        out.save(os.path.join(outdir, name))

    print('Done ...')


def generate2(save_folder):
    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'

    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/esrganHQ/generator_170.pth'
    model = ESRGAN(3, 3, 64)
    model.cuda()

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.eval()

    files = os.listdir(folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(folder, file_name)
        blur = Image.open(blur_file)

        w, h = blur.size
        dw = 4 - w % 4
        dh = 4 - h % 4
    
        blur = F.pad(blur, (0, 0, dw, dh))
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()
        print('dw, dh, w ,h, size: ', dw, dh, w, h, blur.size)
    
        w, h = blur.size
        merge = Image.new('RGB', (w * 2, h))
        merge.paste(blur, (0, 0, w, h))

        with torch.no_grad():
            pred = model(blur_tensor)
            output = pred
            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (w, 0, w*2, h))
           
            merge.save(os.path.join(save_folder, 'x'+ file_name))



if __name__ == '__main__':
    # generate()
    save_folder = '/data/juicefs_hz_cv_v3/11145199/work/ESRGAN/outputs'
    generateHQ(save_folder)
    # generate2(save_folder)
