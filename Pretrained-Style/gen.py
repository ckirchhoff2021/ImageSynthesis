import os
import torch
import torchvision
from torch import nn
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import *
from tqdm import tqdm as tqdm
import pickle
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # get rid of interpolation warning

from torchvision.utils import save_image
from util import *
import scipy
import cv2


device = 'cuda' #@param ['cuda', 'cpu']


def panorama():
    model_type = 'models/landscape.pt'          # @param ['church', 'face', 'landscape']
    num_im =  5                                 # @param {type:"number"}
    random_seed =  90                           # @param {type:"number"}

    generator = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)
    truncation = 0.7
    mean_latent = load_model(generator, model_type)

    pad = 512//4
    all_im = []
    random_state = np.random.RandomState(random_seed)

    with torch.no_grad():
        z = random_state.randn(num_im, 512).astype(np.float32)
        z = scipy.ndimage.gaussian_filter(z, [.7, 0], mode='wrap')
        z /= np.sqrt(np.mean(np.square(z)))
        z = torch.from_numpy(z).to(device)

        source = generator.get_latent(z, truncation=truncation, mean_latent=mean_latent)
            
        for i in range(num_im-1):
            source1 = index_layers(source, i)
            source2 = index_layers(source, i+1)
            all_im.append(generator.merge_extension(source1, source2))


    b,c,h,w = all_im[0].shape
    panorama_im = torch.zeros(b,c,h,512+(num_im-2)*256)

    coord = 256+pad
    panorama_im[..., :coord] = all_im[0][..., :coord]

    for im in all_im[1:]:
        panorama_im[..., coord:coord+512-2*pad] = im[..., pad:-pad]
        coord += 512-2*pad
    panorama_im[..., coord:] = all_im[-1][..., 512-pad:]

    print(panorama_im.shape)
    save_image(panorama_im, 'outputs/panorama.png')



def disney():
    generator = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()
    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()

    mean_latent1 = load_model(generator, 'models/face.pt')
    mean_latent2 = load_model(generator2, 'models/disney.pt')

    truncation = .5
    face_seed =  19261
    disney_seed =  20131
    
    with torch.no_grad():
        torch.manual_seed(face_seed)
        source_code = torch.randn([1, 512]).to(device)
        latent1 = generator2.get_latent(source_code, truncation=truncation, mean_latent=mean_latent2)
        source_im, _ = generator(latent1)

        torch.manual_seed(disney_seed)
        reference_code = torch.randn([1, 512]).to(device)
        latent2 = generator2.get_latent(reference_code, truncation=truncation, mean_latent=mean_latent2)
        reference_im, _ = generator2(latent2)

        disney_image = torch.cat([source_im, reference_im], -1)
        save_image(disney_image, 'outputs/disney.png')



def transfer():
    generator = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()
    model_type = 'models/face.pt' 
    random_seed =  428290
    truncation = 0.7

    torch.manual_seed(random_seed)
    mean_latent = load_model(generator, f'{model_type}.pt')

    with torch.no_grad():
        code = torch.randn([1, 512]).to(device)
        code2 = torch.randn([1,512]).to(device)

        source = generator.get_latent(code, truncation=truncation, mean_latent=mean_latent)
        source2 = generator.get_latent(code2, truncation=truncation, mean_latent=mean_latent)
    
        source_im, _ = generator(source)
        target_im, _ = generator(source2)
        image = torch.cat([source_im, target_im], -1)
        save_image(image, 'outputs/transfer.png')




if __name__ == '__main__':
    # panorama()
    # disney()
    transfer()