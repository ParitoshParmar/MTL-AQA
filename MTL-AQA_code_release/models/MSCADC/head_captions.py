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
#
# utils_1.py is based on the following implementation: https://github.com/xiadingZ/video-caption.pytorch
# Thanks to the author!

import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=1200, sos_id=1, eos_id=0,
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn1 = self.rnn_cell(dim_vid, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = self.rnn_cell(dim_hidden + dim_word, dim_hidden, n_layers,
                                  batch_first=True, dropout=rnn_dropout_p)
        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        #########
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

        self.head_conv3 = nn.Conv3d(int(btlnk), int(btlnk), kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.head_pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1))

        self.relu = nn.ReLU()


    def forward(self, vid_feats_in, target_variable=None, mode='train', opt={}):
        # model head
        vid_feats = self.head_conv1(vid_feats_in)
        vid_feats = self.cntxt_pad(vid_feats)
        vid_feats = self.relu(self.cntxt_bn1(self.cntxt_conv1(vid_feats)))
        vid_feats = self.relu(self.cntxt_bn2(self.cntxt_conv2(vid_feats)))
        vid_feats = self.relu(self.cntxt_bn3(self.cntxt_conv3(vid_feats)))
        vid_feats = self.relu(self.cntxt_bn4(self.cntxt_conv4(vid_feats)))
        vid_feats = self.relu(self.cntxt_bn5(self.cntxt_conv5(vid_feats)))
        vid_feats = self.cntxt_conv6(vid_feats)

        vid_feats = self.relu(self.head_bn1(vid_feats))
        vid_feats = self.head_pool1(vid_feats)

        vid_feats = self.relu(self.head_bn2(self.head_conv2(vid_feats)))

        vid_feats = self.head_pool3(self.head_conv3(vid_feats))

        vid_feats = vid_feats.view(-1, 1200)
        vid_feats = torch.unsqueeze(vid_feats,1)

        batch_size, n_frames, _ = vid_feats.shape

        padding_words = Variable(vid_feats.data.new(batch_size, n_frames, self.dim_word)).zero_()
        padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
        state1 = None
        state2 = None
        output1, state1 = self.rnn1(vid_feats, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)

        seq_probs = []
        seq_preds = []
        if mode == 'train':
            for i in range(self.max_length - 1):
                target_variable = target_variable.type(torch.LongTensor).cuda()
                current_words = self.embedding(target_variable[:, i])
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
        else:
            current_words = self.embedding(Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
            for i in range(self.max_length - 1):
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds