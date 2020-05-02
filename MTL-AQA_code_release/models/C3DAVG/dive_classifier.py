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
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class dive_classifier(nn.Module):
    def __init__(self):
        super(dive_classifier, self).__init__()
        self.fc_position = nn.Linear(4096,3)
        self.fc_armstand = nn.Linear(4096,2)
        self.fc_rot_type = nn.Linear(4096,4)
        self.fc_ss_no = nn.Linear(4096,10)
        self.fc_tw_no = nn.Linear(4096,8)


    def forward(self, x):
        position = self.fc_position(x)
        armstand = self.fc_armstand(x)
        rot_type = self.fc_rot_type(x)
        ss_no = self.fc_ss_no(x)
        tw_no = self.fc_tw_no(x)
        return position, armstand, rot_type, ss_no, tw_no