# coding: utf-8
import math
import torch
import torch.nn as nn
from torch.functional import F
from torchvision import models as built_in_models


class Model(nn.Module):
    def __init__(self, num_classes=1000, *models):
        """
        models: string names of models (lookup for the global namespace)
        """
        super(Model, self).__init__()
        imgDim = 3
        if len(models) == 0:
            print('switching to the default combination')
            models = [
                # 'IceResNet'
                # , 'LeNet'
                'densenet161'
            ]
        self.models_str = models
        for model in models:
            if model == 'IceResNet':
                setattr(self, model, IceResNet(IceSEBasicBlock, 1, num_classes, 3, 32))
            elif model == 'LeNet':
                setattr(self, model, LeNet(num_classes, imgDim))
            # elif model == 'MiniDenseNet':
            #     setattr(self, model,
            #             MiniDenseNet(growthRate=48, depth=20, reduction=0.5, bottleneck=True, nClasses=num_classes,
            #                          n_dim=imgDim))
            elif model == 'densenet161':
                setattr(self, model, built_in_models.densenet161())
            elif model == 'vgg19':
                setattr(self, model, built_in_models.vgg19_bn())
            else:
                raise ValueError('unrecognized model: %s' % str(model))
        self.fc = nn.Linear(len(models) * num_classes, num_classes)
        self.softmax = nn.Softmax()
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        indiv_proj = []
        for model in self.models_str:
            _x = x.clone()
            _m = getattr(self, model)
            _h = _m(_x)
            indiv_proj.append(_h)
        x1 = torch.cat(indiv_proj, dim=1)
        x1 = F.relu(x1)
        x2 = self.fc(x1)
        return x2


class LeNet(nn.Module):
    def __init__(self, num_classes=12, num_rgb=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_rgb, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # print(out.data.size())
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), -1)
        # print(out.data.size())
        out = self.fc3(out)
        # out = self.sig(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

REDUCTION=16

class SELayer(nn.Module):
    def __init__(self, channel, reduction=REDUCTION):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IceSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, reduction=REDUCTION):
        super(IceSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class IceResNet(nn.Module):
    def __init__(self, block, n_size=1, num_classes=1, num_rgb=2, base=32):
        super(IceResNet, self).__init__()
        self.base = base
        self.num_classes = num_classes
        self.inplane = self.base  # 45 epochs
        # self.inplane = 16 # 57 epochs
        self.conv1 = nn.Conv2d(num_rgb, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.inplane, blocks=2 * n_size, stride=2)
        self.layer2 = self._make_layer(block, self.inplane * 2, blocks=2 * n_size, stride=2)
        self.layer3 = self._make_layer(block, self.inplane * 4, blocks=2 * n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(int(8 * self.base), num_classes)
        nn.init.kaiming_normal(self.fc.weight)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride):

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print (x.data.size())
        x = self.fc(x)

        if self.num_classes == 1:  # BCE Loss,
            x = self.sig(x)
        return x

