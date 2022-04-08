import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19, vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_file):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16()
        state = torch.load(vgg_file)
        vgg.load_state_dict(state)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False

        '''
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        '''

        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss
