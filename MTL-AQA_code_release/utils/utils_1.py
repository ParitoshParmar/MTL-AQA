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
import torch.nn as nn

torch.manual_seed(1); torch.cuda.manual_seed(1)


def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()


    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(), mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)


    def forward(self, logits, target, mask):
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output