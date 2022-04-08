import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt


from face_enhancement import FaceEnhancement
from sr_model.MIMOUNet import build_net
import torch
import torch.nn as nn

from torchvision.transforms import functional as F
from PIL import Image


def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('GPEN output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)


def detect_and_process():
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    # model = {'name':'GPEN-BFR-256', 'size':256, 'channel_multiplier':1, 'narrow':0.5}

    indir = 'examples/imgs'
    outdir = 'examples/run'

    indir = 'examples/test'
    outdir = 'examples/test'
    os.makedirs(outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(use_sr=True, size=model['size'], model=model['name'],
                                   channel_multiplier=model['channel_multiplier'], narrow=model['narrow'])

    # file = os.path.join(indir, 'Solvay_conference_1927.png')
    file = os.path.join(indir, 'IMG_20211223_194917.jpg')
    filename = os.path.basename(file)
    im = cv2.imread(file, cv2.IMREAD_COLOR)  # BGR
    # im = cv2.resize(im, (0,0), fx=2, fy=2) #optional

    img, orig_faces, enhanced_faces = faceenhancer.process(im)

    im = cv2.resize(im, img.shape[:2][::-1])
    cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_COMP.jpg'), np.hstack((im, img)))
    cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_GPEN.jpg'), img)

    for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
        of = cv2.resize(of, ef.shape[:2])
        cv2.imwrite(os.path.join(outdir, '.'.join(filename.split('.')[:-1]) + '_face%02d' % m + '.jpg'),
                    np.hstack((of, ef)))

    display(im, img)



def generate():
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    # model = {'name':'GPEN-BFR-256', 'size':256, 'channel_multiplier':1, 'narrow':0.5}

    indir = 'examples/imgs'
    outdir = 'examples/run'
    os.makedirs(outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(use_sr=True, size=model['size'], model=model['name'],
                                   channel_multiplier=model['channel_multiplier'], narrow=model['narrow'])

    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    files = os.listdir(folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(folder, file_name)
        im = cv2.imread(blur_file, cv2.IMREAD_COLOR)
        _, orig_faces, enhanced_faces = faceenhancer.process(im)

        ef, of = orig_faces[0], enhanced_faces[0]
        of = cv2.resize(of, ef.shape[:2])
        cv2.imwrite(os.path.join(outdir,  file_name + '.jpg'), np.hstack((of, ef)))


def simplify_gen():
    from face_model.face_gan import FaceGAN
    base_dir = './'
    outdir = 'examples/run2'
    os.makedirs(outdir, exist_ok=True)

    size = 512
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    facegan = FaceGAN(base_dir, size, model['name'], model['channel_multiplier'], model['narrow'], device='cuda')

    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    files = os.listdir(folder)
    print('Num: ', len(files))

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(folder, file_name)
        x = cv2.imread(blur_file, cv2.IMREAD_COLOR)
        y = facegan.process(x)
        x = cv2.resize(x, y.shape[:2])
        cv2.imwrite(os.path.join(outdir,  file_name), np.hstack((x, y)))

    print('Done ...')




def simplify_gen2():
    from face_model.face_gan import FaceGAN
    base_dir = './'
    outdir = 'examples/run3'
    os.makedirs(outdir, exist_ok=True)

    size = 512
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    facegan = FaceGAN(base_dir, size, model['name'], model['channel_multiplier'], model['narrow'], device='cuda')

    folder = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/crop_face/medium_light'
    files = os.listdir(folder)
    print('Num: ', len(files))

    base_file = '/data/juicefs_hz_cv_v3/11145199/pretrained/model_lite_face_512.pkl'
    base_name = 'MIMO-UNet'
    base = build_net(base_name)
    base = nn.DataParallel(base,device_ids=list(range(torch.cuda.device_count()))).cuda()
    state_dict = torch.load(base_file)
    base.load_state_dict(state_dict['model'])
    base.eval()

    from tqdm import tqdm
    for file_name in tqdm(files):
        if not file_name.endswith('.jpg'):
            continue

        blur_file = os.path.join(folder, file_name)
        x = cv2.imread(blur_file, cv2.IMREAD_COLOR)
        y = facegan.process(x)
        x = cv2.resize(x, y.shape[:2])
       
        blur_file = os.path.join(folder, file_name)
        blur = Image.open(blur_file)

        w, h = blur.size
        dw = 4 - w % 4
        dh = 4 - h % 4

        blur = F.pad(blur, (0, 0, dw, dh))
        blur_tensor = F.to_tensor(blur).unsqueeze(0).cuda()
      
        pred = base(blur_tensor)
        output = pred[2]
        output = torch.clamp(output, 0, 1).cpu()
        output += 0.5 / 255
        z = F.to_pil_image(output.squeeze(0), 'RGB')
        z = np.array(z)
        z = z[:,:,::-1]
        z = cv2.resize(z, y.shape[:2])

        cv2.imwrite(os.path.join(outdir,  file_name), np.hstack((x, z, y)))



def infer_faceHQ():
    from face_model.face_gan import FaceGAN
    base_dir = './'
    outdir = 'examples/runHQ'
    os.makedirs(outdir, exist_ok=True)

    size = 512
    model = {'name': 'GPEN-BFR-512', 'size': 512, 'channel_multiplier': 2, 'narrow': 1}
    facegan = FaceGAN(base_dir, size, model['name'], model['channel_multiplier'], model['narrow'], device='cuda')

    import json
    test_file = '/data/juicefs_hz_cv_v3/11145199/datas/ffhq/ffhq_test.json'
    files = json.load(open(test_file, 'r'))
    print('Test Num: ', len(files))

    from tqdm import tqdm
    for file in tqdm(files):
        b, s = file
        name = b.split('/')[-1]
        t = cv2.imread(s)

        x = cv2.imread(b, cv2.IMREAD_COLOR)
        y = facegan.process(x)
        x = cv2.resize(x, y.shape[:2])
        t = cv2.resize(t, y.shape[:2])
        cv2.imwrite(os.path.join(outdir,  name), np.hstack((x, t, y)))

    print('Done ...')



if __name__ == '__main__':
    detect_and_process()
    # generate()
    # simplify_gen()
    # simplify_gen2()
    # infer_faceHQ()