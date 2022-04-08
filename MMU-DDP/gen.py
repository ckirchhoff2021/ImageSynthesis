import torch
import torch.nn as nn
from torchvision.transforms import functional as F

from PIL import Image, ImageFilter
from models.MIMOUNet import build_net
from skimage.metrics import peak_signal_noise_ratio

from collections import OrderedDict
from models.MIMOUNet import MIMOUNet, MIMOUNetRRDB, MIMOUNetRRDBEnhanced
from models.layers import FFTResBlock

import cv2
import os
import numpy as np
import torch.nn.functional as F1
from tqdm import tqdm


blur_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/input/'
sharp_folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/public_dataset/GoPro/samples/target/'


model_dict = {
    'base': '/data/juicefs_hz_cv_v3/11145199/pretrained/model_lite_face_512.pkl',
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
    base = load_model('MIMO-UNetRRDB')
    model = load_model('MIMO-UNetRRDBEnhanced')
    save_path = os.path.join('../results', 'rrdbEnhanced')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    data_folder = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_noglass'
    files = os.listdir(data_folder)

    from tqdm import tqdm

    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(data_folder, file_name)
        blur = Image.open(blur_file)
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()

        merge = Image.new('RGB', (1536, 512))
        merge.paste(blur, (0, 0, 512, 512))
    
        with torch.no_grad():
            pred = base(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()

            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (512, 0, 1024, 512))

            x2 = F1.interpolate(blur_tensor, scale_factor=0.5)
            x4 = F1.interpolate(blur_tensor, scale_factor=0.25)
            pred = model(blur_tensor, x2, x4)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (1024, 0, 1536, 512))

            merge.save(os.path.join(save_path, file_name))




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



def infer_face(model_file, model_name):
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/HQ/MIMO-UNet-GAN/weights/model_200.pkl'
    model = build_net(model_name)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    #model = torch.load(model_file)
    # torch.save({'model': model.state_dict()}, '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/GradientDeblurGAN/weights/model-43.pkl')
    model.eval()

    file = '/data/juicefs_hz_cv_v3/11145199/datas/x1.jpg'
    
    import json
    test_file = '/data/juicefs_hz_cv_v3/11145199/datas/ffhq/ffhq_test.json'
    files = json.load(open(test_file, 'r'))
    print('Test Num: ', len(files))

    for file in tqdm(files):
        b, s = file
        name = b.split('/')[-1]
        blur = Image.open(b)
        sharp = Image.open(s)
        w, h = blur.size

        blur_tensor = F.to_tensor(blur).unsqueeze(0)

        merge = Image.new('RGB', (w*3, h))
        merge.paste(blur, (0, 0, w, h))
        merge.paste(sharp, (w, 0, w*2, h))

        with torch.no_grad():  
            pred = model(blur_tensor)
            output = pred[3]
            output = torch.clamp(output, 0, 1).cpu()
        
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (w*2, 0, w*3, h))

            merge.save(os.path.join('outputs/ms4HQ2', name))



def infer_face2(model_file, model_name):
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/HQ/MIMO-UNet-GAN/weights/model_200.pkl'
    model = build_net(model_name)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    #model = torch.load(model_file)
    # torch.save({'model': model.state_dict()}, '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/GradientDeblurGAN/weights/model-43.pkl')
    model.eval()

    file = '/data/juicefs_hz_cv_v3/11145199/datas/x1.jpg'
    file = '/data/juicefs_hz_cv_v3/11145199/work/ffhq.jpg'
    
    import json
    import cv2
    test_file = '/data/juicefs_hz_cv_v3/11145199/datas/ffhq/ffhq_test.json'
    files = json.load(open(test_file, 'r'))
    print('Test Num: ', len(files))

    save_folder = '/data/juicefs_hz_cv_v3/11145199/work/results/outputs/mimo4d1000'
    os.makedirs(save_folder, exist_ok=True)
  
    for file in tqdm(files):
        b, s = file
        name = b.split('/')[-1]

        x = cv2.imread(b)
        t = cv2.imread(s)
        _, w, _ = x.shape
        # blur = (x / 255.0 - 0.5) / 0.5
        blur = x / 255.0
        blur = np.transpose(blur, (2, 0, 1))
        blur_tensor = torch.from_numpy(blur).unsqueeze(0).float()

        with torch.no_grad():  
            pred = model(blur_tensor)
            y = pred[2]
            # y = y * 0.5 + 0.5
            y = torch.clamp(y, 0, 1).cpu().squeeze().numpy()
            y = y * 255.0
            g = np.transpose(y, (1, 2, 0)).astype(np.uint8)
            merge = np.concatenate([x, t, g], axis=1)
            cv2.imwrite(os.path.join(save_folder, name), merge)



def generate(base_file, base_name, model_file, model_name, save_folder):
    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    base = build_net(base_name)
    base.cuda()
    base = nn.DataParallel(base,device_ids=list(range(torch.cuda.device_count()))) 
    state_dict = torch.load(base_file)
    base.load_state_dict(state_dict['model'])
    base.eval()

    model = build_net(model_name)
    model.cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) 
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    os.makedirs(save_folder, exist_ok=True)

    files = os.listdir(folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(folder, file_name)
        blur = Image.open(blur_file)

        w, h = blur.size
        # dw = 4 - w % 4
        # dh = 4 - h % 4

        dw = 8 - w % 8
        dh = 8 - h % 8
    
        blur = F.pad(blur, (0, 0, dw, dh))
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()
        print('dw, dh, w ,h, size: ', dw, dh, w, h, blur.size)
    
        w, h = blur.size
        merge = Image.new('RGB', (w * 3, h))
        merge.paste(blur, (0, 0, w, h))

        with torch.no_grad():
            pred = base(blur_tensor)
            output = pred[2]
            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (w, 0, w*2, h))

            pred = model(blur_tensor)
            output = pred[3]
            output = torch.clamp(output, 0, 1).cpu()
            output += 0.5 / 255
            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')
            merge.paste(pred_image, (w*2, 0, w*3, h))
           
            merge.save(os.path.join(save_folder, file_name))





def generate_grads(model_file, model_name, save_folder):
    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    model = build_net(model_name)
    model.cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) 
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['model'])
    model.eval()

    files = os.listdir(folder)
    print('Num: ', len(files))

    os.makedirs(save_folder, exist_ok=True)

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
      
        w, h = blur.size
        merge = Image.new('RGB', (w * 2, h * 2))
        merge.paste(blur, (0, 0, w, h))

        with torch.no_grad():
            pred = model(blur_tensor)
            grads_gt = model.module.get_gradient(blur_tensor)
            grads_gt = grads_gt.cpu()

            output = pred[2]
            grads_out = pred[4]

            output += 0.5 / 255
            output = torch.clamp(output, 0, 1).cpu()

            grads_out += 0.5/255
            grads_out = torch.clamp(grads_out, 0, 1).cpu()

            pred_image = F.to_pil_image(output.squeeze(0), 'RGB')

            grad1 = F.to_pil_image(grads_gt.squeeze(0), 'RGB')
            grad2 = F.to_pil_image(grads_out.squeeze(0), 'RGB')

            merge.paste(pred_image, (w, 0, w*2, h))
            merge.paste(grad1, (0, h, w, h*2))
            merge.paste(grad2, (w, h, w*2, h*2))

            merge.save(os.path.join(save_folder, file_name))

 
if __name__ == '__main__':
    # main()
    # generate()
    # mimo_Gan_test()
    # fft_test()
    # mimo_pixel_test()

    model_name =  'MIMO-UNet'  # 'MS4UNet' # 'GradientDeblurGAN' 
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0119/GradientDeblurGAN/weights/model_50.pkl'

    '''
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/patchGAN/MIMO-UNet/weights/model_150.pkl'
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/patch/MIMO-UNet/weights/model_550.pkl'
    '''
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/MIMO-UNet/weights/model_1000.pkl'
   

    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0119/patch/MIMO-UNet/weights/model-0.pkl'
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0125/MS4UNet/weights/model_500.pkl'
   
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0119/GradientDeblurGAN/weights/model_50.pkl'
    # infer_face2(model_file, model_name)

    
    base_file = '/data/juicefs_hz_cv_v3/11145199/pretrained/model_lite_face_512.pkl'
    base_name = 'MIMO-UNet'

    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0114/patch/MIMO-UNet/weights/model_500.pkl'
    # model_name = 'MIMO-UNet'

    model_name =  'GradientDeblurGAN'
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/finetune/0207/GradientDeblurGAN/weights/model_500.pkl'

    model_name = 'MS4UNet'
    model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0126/MS4UNet/weights/model_500.pkl'
    # model_file = '/data/juicefs_hz_cv_v2/11145199/deblur/results/GANs/0125/MS4UNet/weights/model_350.pkl'

    save_folder = '/data/juicefs_hz_cv_v3/11145199/work/results/outputs/ms4-500'
    generate(base_file, base_name, model_file, model_name, save_folder)
    # generate_grads(model_file, model_name, save_folder)