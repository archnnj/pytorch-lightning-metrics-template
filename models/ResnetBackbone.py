import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import transforms as TF
import pytorch_lightning as pl


class ResnetBackbone(pl.LightningModule):
    _resnet_model_mapping = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, in_ch=3, resnet_type='resnet18', pretrained=True, resize_shape=224):
        super(ResnetBackbone, self).__init__()
        assert resnet_type in ResnetBackbone._resnet_model_mapping, \
            f"{self.__class__.__name__}: please specify a valid ResNet architecture, received {resnet_type}."
        resnet = ResnetBackbone._resnet_model_mapping[resnet_type](weights='DEFAULT' if pretrained else None)
        assert in_ch in [1, 3], f"{self.__class__.__name__}: in_ch can be either 1 or 3, received {in_ch}."
        self.save_hyperparameters('resnet_type', 'pretrained', 'in_ch', 'resize_shape')
        if in_ch == 1:
            # adapt for 1 channel input
            conv_3cto1 = nn.Conv2d(1, 3, 1)
            conv_3cto1.weight = nn.Parameter(torch.ones_like(conv_3cto1.weight))
            resnet.conv1 = nn.Sequential(conv_3cto1, resnet.conv1)  # adapt to 1-channel input
        layers = list(resnet.children())[:-1]  # except fc layer
        self.model = nn.Sequential(*layers)
        self.num_features_out = resnet.fc.in_features
        self.resize_shape = resize_shape

    def forward(self, x):
        if self.resize_shape != 224:
            x = TF.Resize((self.resize_shape, self.resize_shape))
        return self.model(x)
