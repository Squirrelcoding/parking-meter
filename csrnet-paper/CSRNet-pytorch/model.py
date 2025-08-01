import torch.nn as nn
from torchvision import models

VGG_MODEL = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        # Found on page 5 of the paper

        super(CSRNet, self).__init__()
        self.seen = 0

        # Features for frontend (VGG-16) and backend. M stands for a max-pooling layer.
        self.frontend_feat = VGG_MODEL
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        # Make the layers
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            mod_state_items = list(mod.state_dict().items())
            frontend_state_items = list(self.frontend.state_dict().items())

            for i in range(len(frontend_state_items)):
                frontend_state_items[i][1].data[:] = mod_state_items[i][1].data[:]


    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2 
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
