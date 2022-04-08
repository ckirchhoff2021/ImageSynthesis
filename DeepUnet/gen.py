import torch
import torch.nn as nn
from torchvision.transforms import functional as F

from PIL import Image, ImageFilter
from models.MIMOUNet import build_net
from skimage.metrics import peak_signal_noise_ratio

from collections import OrderedDict
from models.MIMOUNet import MIMOUNet, MIMOUNetRRDB, MIMOUNetRRDBEnhanced
from models.unet import DeblurUNet
from models.layers import FFTResBlock

import os
import numpy as np
import torch.nn.functional as F1


blur_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/input/'
sharp_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/target/'


model_dict = {
    # 'MIMO-UNet': '/data/juicefs_hz_cv_v2/11145199/deblur/results/MIMO-UNet/weights/model_500.pkl',
    'MIMO-UNet': '/data/juicefs_hz_cv_v2/11145199/deblur/results/base/MIMO-UNet/weights/model_1000.pkl',
    'MIMO-UNetRRDB': '/data/juicefs_hz_cv_v2/11145199/deblur/results/MIMO-UNetRRDB/weights/model_500.pkl',
    'MIMO-UNetRRDBEnhanced': '/data/juicefs_hz_cv_v2/11145199/deblur/results/MIMO-UNetRRDBEnhanced/weights/model_500.pkl',
    'MIMO-UNetFFT': '/data/juicefs_hz_cv_v2/11145199/deblur/results/MIMO-UNetFFT/weights/model_0.pkl'
}


def load_model(model_name):
    model_file = model_dict[model_name]
    model = build_net(model_name)

    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    '''
    state = OrderedDict()
    for k, v in state_dict['model'].items():
        name = k[7:]
        state_dict[name] = v
    model.load_state_dict(state)
    model.eval()
    '''
    return model


def main():
    base = load_model('MIMO-UNet')
    model = load_model('MIMO-UNetRRDBEnhanced')
    save_path = os.path.join('outputs', 'rrdbEnhanced')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    files = os.listdir(blur_folder)

    base_psnrs = list()
    model_psnrs = list()

    for file_name in files:
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(blur_folder, file_name)
        sharp_file = os.path.join(sharp_folder, file_name)
        
        blur = Image.open(blur_file)
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()

        x2 = F1.interpolate(blur_tensor, scale_factor=0.5)
        x4 = F1.interpolate(blur_tensor, scale_factor=0.25)
        x2, x4 = x2.cuda(), x4.cuda()

        sharp = Image.open(sharp_file)
        sharp_tensor = F.to_tensor(sharp).unsqueeze(0)

        merge = Image.new('RGB', (1200, 300))
        merge.paste(blur, (0, 0, 300, 300))
        merge.paste(sharp, (300, 0, 600, 300))


        with torch.no_grad():
            pred = base(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
            label_numpy = sharp_tensor.squeeze(0).numpy()

            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (600, 0, 900, 300))

            base_psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
          
            pred = model(blur_tensor, x2, x4)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
        
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (900, 0, 1200, 300))

            model_psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            print('==> base PSNR: %.2f:, model PSNR: %.2f' % (base_psnr, model_psnr))

            merge.save(os.path.join(save_path, file_name))
            base_psnrs.append(base_psnr)
            model_psnrs.append(model_psnr)


    print('average base: ', np.mean(base_psnrs))
    print('average model: ', np.mean(model_psnrs))



def generate():
    model = load_model('MIMO-UNetRRDB')
    save_path = os.path.join('outputs', 'badcase')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    data_folder = '/data/juicefs_hz_cv_v3/11145199/work/badcase'
    files = os.listdir(data_folder)
    for file_name in files:
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(data_folder, file_name)
        blur = Image.open(blur_file)
        print(blur_file, type(blur))

        w, h = blur.size
        dw = 4 - w % 4
        dh = 4 - h % 4
    
        blur = F.pad(blur, (0, 0, dw, dh))
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()
        print('dw, dh, w ,h, size: ', dw, dh, w, h, blur.size)
    

        w, h = blur.size
        merge = Image.new('RGB', (w * 4, h))
        merge.paste(blur, (0, 0, w, h))

        half_b = F1.interpolate(blur_tensor, scale_factor=0.5)
        hhalf_b = F1.interpolate(half_b, scale_factor=0.5)       

        with torch.no_grad():
            pred = model(blur_tensor, half_b, hhalf_b)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (w, 0, w*2, h))
           
            half_output = pred[1]
            half_output = torch.clamp(half_output, 0, 1).cpu()
            half_pred = F.to_pil_image(half_output.squeeze(0), 'RGB')
            half_pred = half_pred.resize((w, h))
            merge.paste(half_pred, (w*2, 0, w*3, h))

            hhalf_output = pred[1]
            hhalf_output = torch.clamp(hhalf_output, 0, 1).cpu()
            hhalf_pred = F.to_pil_image(hhalf_output.squeeze(0), 'RGB')
            hhalf_pred = hhalf_pred.resize((w, h))
            merge.paste(hhalf_pred, (w*3, 0, w*4, h))

            merge.save(os.path.join(save_path, file_name))




def mimo_Gan_test():
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/MIMO-UNet/weights/model_100.pkl'
    model = build_net('MIMO-UNet')
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    base = load_model('MIMO-UNet')
    save_path = os.path.join('outputs', 'gan')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    files = os.listdir(blur_folder)

    base_psnrs = list()
    model_psnrs = list()

    for file_name in files:
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(blur_folder, file_name)
        sharp_file = os.path.join(sharp_folder, file_name)
        
        blur = Image.open(blur_file)
        blur_tensor = F.to_tensor(blur).unsqueeze(0)

        sharp = Image.open(sharp_file)
        sharp_tensor = F.to_tensor(sharp).unsqueeze(0)

        merge = Image.new('RGB', (1200, 300))
        merge.paste(blur, (0, 0, 300, 300))
        merge.paste(sharp, (300, 0, 600, 300))


        with torch.no_grad():
            pred = base(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
            label_numpy = sharp_tensor.squeeze(0).numpy()

            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (600, 0, 900, 300))

            base_psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
          
            pred = model(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
        
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (900, 0, 1200, 300))

            model_psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            print('==> base PSNR: %.2f:, model PSNR: %.2f' % (base_psnr, model_psnr))

            merge.save(os.path.join(save_path, file_name))
            base_psnrs.append(base_psnr)
            model_psnrs.append(model_psnr)


    print('average base: ', np.mean(base_psnrs))
    print('average model: ', np.mean(model_psnrs))



def mimo_pixel_test():
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/base/MIMO-UNet/weights/model_200.pkl'
    model = build_net('MIMO-UNet')
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    save_path = os.path.join('outputs', 'pixel')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    files = os.listdir(blur_folder)

    model_psnrs = list()

    for file_name in files:
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(blur_folder, file_name)
        sharp_file = os.path.join(sharp_folder, file_name)
        
        blur = Image.open(blur_file)
        blur_tensor = F.to_tensor(blur).unsqueeze(0)

        sharp = Image.open(sharp_file)
        sharp_tensor = F.to_tensor(sharp).unsqueeze(0)

        merge = Image.new('RGB', (1200, 300))
        merge.paste(blur, (0, 0, 300, 300))
        merge.paste(sharp, (300, 0, 600, 300))


        with torch.no_grad():
            pred = model(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            pred_numpy = output.squeeze(0).numpy()
            label_numpy = sharp_tensor.squeeze(0).numpy()
        
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (900, 0, 1200, 300))

            model_psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            print('==> model PSNR: %.2f' % model_psnr)

            merge.save(os.path.join(save_path, file_name))
            model_psnrs.append(model_psnr)

    print('average model: ', np.mean(model_psnrs))




def fft_test():
    x = torch.randn(1, 3, 12, 12)
    print(x)

    _, _, H, W = x.shape
    y = torch.rfft(x, 2, normalized=False, onesided=False)
    print(y.size())

    y1 = torch.irfft(y, 2, normalized=False, onesided=False, signal_sizes=(H,W))
    print(y1.size())   
    print(y1)

    net = FFTResBlock(3)
    y = net(x)
    print(y.size())



def infer_face(model_file):
    model = DeblurUNet()
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    file = '/data/juicefs_hz_cv_v3/11145199/datas/x1.jpg'

    blur = Image.open(file)
    blur_tensor = F.to_tensor(blur).unsqueeze(0)

    merge = Image.new('RGB', (512, 256))
    merge.paste(blur, (0, 0, 256, 256))

    with torch.no_grad():  
        pred = model(blur_tensor)
        output = pred[2]
        output = torch.clamp(output, 0, 1).cpu()
    
        output += 0.5 / 255
        pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
        merge.paste(pred_image, (256, 0, 512, 256))

        merge.save(os.path.join('outputs', 'x1.jpg'))


 
if __name__ == '__main__':
 
    model_name = 'GradientDeblurGAN'
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/0120/model_400.pkl'
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/DeblurUnet/weights/model_100.pkl'
    infer_face(model_file)

