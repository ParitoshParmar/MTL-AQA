# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import torch
import torch.nn as nn
import random
import numpy as np
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True


class C3D_dilated_body(nn.Module):
    def __init__(self):
        super(C3D_dilated_body, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(128)
        self.conv3b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(256)
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(256)

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), dilation=(2, 2, 2))
        self.bn5a = nn.BatchNorm3d(256)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), dilation=(2, 2, 2))
        self.bn5b = nn.BatchNorm3d(256)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # model backbone
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)

        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h)

        h = self.relu(self.bn3a(self.conv3a(h)))
        h = self.relu(self.bn3b(self.conv3b(h)))
        h = self.pool3(h)

        h = self.relu(self.bn4a(self.conv4a(h)))
        h = self.relu(self.bn4b(self.conv4b(h)))

        h = self.relu(self.bn5a(self.conv5a(h)))
        h = self.relu(self.bn5b(self.conv5b(h)))
        h = self.dropout(h)
        return h