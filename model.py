import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.functional import F
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torchvision import models as built_in_models
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import time
import os
import gc
import copy


class Model(nn.Module):
    num_classes = 1000
    imgDim = 3
    
    
    def __init__(self):
        super(Model, self).__init__()
        self.models_str = [
            'resnet'
            , 'densenet'
             , 'inception'
        ]
        self.resnet = built_in_models.resnet34()
        self.densenet = built_in_models.densenet161()
        self.inception = built_in_models.inception_v3(aux_logits=False)
        self.do_1 = nn.Dropout(p=0.35)
        self.fc = nn.Linear(len(self.models_str) * Model.num_classes, Model.num_classes)
        # Load pre-trained model
        state_dict = torch.load('weights.pth')
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k]=v
        
        self.load_state_dict(new_state_dict)

        
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
            _x = self.do_1(x.clone())
            _m = getattr(self, model)
            _h = _m(_x)
            indiv_proj.append(_h)
        x1 = torch.cat(indiv_proj, dim=1)
        x1 = F.relu(x1)
        x2 = self.fc(x1)
        return x2
