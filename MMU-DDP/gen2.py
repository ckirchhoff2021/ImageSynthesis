from re import X
import torch
import torch.nn as nn
from torchvision.transforms import functional as F

from PIL import Image
from models.MIMOUNet import build_net
from models.unet import DeblurUNet
from models.face_model.face_gan import FaceGAN
from skimage.metrics import peak_signal_noise_ratio


import cv2
import os
import numpy as np
import torch.nn.functional as F1
from tqdm import tqdm


test_folder = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_noglass'

base = ['MIMO-UNet', '/data/juicefs_hz_cv_v3/11145199/pretrained/model_lite_face_512.pkl']
mimo_3gan = ['MIMO-UNet', '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/MIMO-UNet/weights/model_500.pkl']
mimo_pgan = ['MIMO-UNet', '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/patch/MIMO-UNet/weights/model_500.pkl']

grad_gan = ['GradientDeblurGAN', '/data/juicefs_hz_cv_v2/11145199/deblur/finetune/0207/GradientDeblurGAN/weights/model_440.pkl']
mimo_s4unet = ['MS4UNet', '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0126/MS4UNet/weights/model_500.pkl']
normal_unet = ['DeblurUNet', '/data/juicefs_hz_cv_v2/11145199/deblur/results/DeblurUnet/weights/model_320.pkl']


def load_facegan():
    base_dir = '/data/juicefs_hz_cv_v3/11145199/work/GPEN'
    size = 512
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    facegan = FaceGAN(base_dir, size, model['name'], model['channel_multiplier'], model['narrow'], device='cuda')
    return facegan


def load_model(model_info):
    name, weights = model_info
    if name == 'DeblurUNet':
        model = DeblurUNet()
    else:
        model = build_net(name)

    model.cuda()
    model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))

    state_dict = torch.load(weights)
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


@torch.no_grad()
def generate(sample_folder, save_folder):
    model_infos = [base, mimo_3gan, mimo_pgan, grad_gan, mimo_s4unet, normal_unet]
    models = list()
    for model_info in model_infos:
        net = load_model(model_info)
        models.append([model_info[0], net])

    facegan = load_facegan()
    os.makedirs(save_folder, exist_ok=True)

    files = os.listdir(sample_folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(sample_folder, file_name)
        blur = Image.open(blur_file)    
        x1 = F.to_tensor(blur).unsqueeze(0).cuda()
        x2 = np.array(blur)
        x2 = x2[:,:,::-1]

        w, h = blur.size
        print('==> [%s]-[%d][%d]'%(file_name, w, h))
        merge = Image.new('RGB', (w * 4, h * 2))
        merge.paste(blur, (0, 0, w, h))

        for i, model in enumerate(models):
            name, net = model
            pred = net(x1)
            if name == 'MS4UNet':
                output = pred[3]
            else:
                output = pred[2]

            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            k1 = int((i+1) / 4)
            k2 = (i+1) % 4
            merge.paste(pred_image, (k2*w, k1*h, (k2+1)*w, (k1+1)*h))

        y2 = facegan.process(x2)
        y1 = Image.fromarray(y2[:,:,::-1])  
        merge.paste(y1, (3*w, h, 4*w, 2*h))
        merge.save(os.path.join(save_folder, file_name))
    
    print('done ...')


def gpen_test():
    facegan = load_facegan()
    file = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_noglass/IMG_20220119_171001.jpg'
    x = cv2.imread(file, cv2.IMREAD_COLOR)
    y = facegan.process(x)
    print(x.shape)
    print(y.shape)
    cv2.imwrite('x.png', y)



if __name__ == '__main__':
    # gpen_test()

    sample_folder = test_folder
    save_folder ='/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_noglass_gen'

    # sample_folder = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220217/faces'
    # save_folder = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220217/faces_gen'
    generate(sample_folder, save_folder)
