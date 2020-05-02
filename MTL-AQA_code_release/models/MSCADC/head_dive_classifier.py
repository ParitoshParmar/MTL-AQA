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


class C3D_dilated_head_classifier(nn.Module):
    def __init__(self):
        super(C3D_dilated_head_classifier, self).__init__()
        btlnk = 12
        # model head
        self.head_conv1 = nn.Conv3d(256, btlnk, kernel_size=(1, 1, 1), padding=(1, 1, 1))
        # context net
        self.cntxt_pad = nn.ReplicationPad3d(padding=(9, 9, 9, 9, 9, 9))
        self.cntxt_conv1 = nn.Conv3d(btlnk, btlnk * 2, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        self.cntxt_bn1 = nn.BatchNorm3d(btlnk * 2)

        self.cntxt_conv2 = nn.Conv3d(btlnk * 2, btlnk * 2, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        self.cntxt_bn2 = nn.BatchNorm3d(btlnk * 2)

        self.cntxt_conv3 = nn.Conv3d(btlnk * 2, btlnk * 4, kernel_size=(3, 3, 3), padding=(0, 0, 0), dilation=(2, 2, 2))
        self.cntxt_bn3 = nn.BatchNorm3d(btlnk * 4)

        self.cntxt_conv4 = nn.Conv3d(btlnk * 4, btlnk * 8, kernel_size=(3, 3, 3), padding=(0, 0, 0), dilation=(4, 4, 4))
        self.cntxt_bn4 = nn.BatchNorm3d(btlnk * 8)

        self.cntxt_conv5 = nn.Conv3d(btlnk * 8, btlnk * 8, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        self.cntxt_bn5 = nn.BatchNorm3d(btlnk * 8)

        self.cntxt_conv6 = nn.Conv3d(btlnk * 8, btlnk, kernel_size=(1, 1, 1), padding=(0, 0, 0))

        # resuming mode head
        self.head_bn1 = nn.BatchNorm3d(btlnk)
        self.head_pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.head_conv2 = nn.Conv3d(btlnk, int(btlnk), kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_bn2 = nn.BatchNorm3d(int(btlnk))

        self.head_conv3_position = nn.Conv3d(int(btlnk), 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3_position = nn.AvgPool3d(kernel_size=(2, 11, 11), stride=(1, 1, 1))
        self.head_conv3_armstand = nn.Conv3d(int(btlnk), 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3_armstand = nn.AvgPool3d(kernel_size=(2, 11, 11), stride=(1, 1, 1))
        self.head_conv3_rot_type = nn.Conv3d(int(btlnk), 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3_rot_type = nn.AvgPool3d(kernel_size=(2, 11, 11), stride=(1, 1, 1))
        self.head_conv3_ss_no = nn.Conv3d(int(btlnk), 10, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3_ss_no = nn.AvgPool3d(kernel_size=(2, 11, 11), stride=(1, 1, 1))
        self.head_conv3_tw_no = nn.Conv3d(int(btlnk), 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3_tw_no = nn.AvgPool3d(kernel_size=(2, 11, 11), stride=(1, 1, 1))

        self.relu = nn.ReLU()


    def forward(self, x):
        # model head
        h = self.head_conv1(x)
        h = self.cntxt_pad(h)
        h = self.relu(self.cntxt_bn1(self.cntxt_conv1(h)))
        h = self.relu(self.cntxt_bn2(self.cntxt_conv2(h)))
        h = self.relu(self.cntxt_bn3(self.cntxt_conv3(h)))
        h = self.relu(self.cntxt_bn4(self.cntxt_conv4(h)))
        h = self.relu(self.cntxt_bn5(self.cntxt_conv5(h)))
        h = self.cntxt_conv6(h)

        h = self.relu(self.head_bn1(h))
        h = self.head_pool1(h)

        h = self.relu(self.head_bn2(self.head_conv2(h)))

        position = self.head_pool3_position(self.head_conv3_position(h)).squeeze_()
        armstand = self.head_pool3_armstand(self.head_conv3_armstand(h)).squeeze_()
        rot_type = self.head_pool3_rot_type(self.head_conv3_rot_type(h)).squeeze_()
        ss_no = self.head_pool3_ss_no(self.head_conv3_ss_no(h)).squeeze_()
        tw_no = self.head_pool3_tw_no(self.head_conv3_tw_no(h)).squeeze_()
        return position, armstand, rot_type, ss_no, tw_no