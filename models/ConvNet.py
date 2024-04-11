import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_channels, num_classes, net_width=128, net_depth=3, net_norm='instancenorm', net_pooling='avgpooling', img_size = (32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._get_layers(num_channels, net_width, net_depth, net_norm, net_pooling, img_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out

    def _get_layers(self, channel, net_width, net_depth, net_norm, net_pooling, im_size):
        layers = []
        in_channels = channel

        if im_size[0] == 28:
            im_size = (32, 32)

        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width

            if net_norm != 'none':
                layers += [nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)]

            layers += [nn.ReLU(inplace=True)]

            in_channels = net_width

            if net_pooling != 'none':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat
