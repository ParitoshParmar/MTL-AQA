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

import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from opts import *

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True


def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class VideoDataset(Dataset):
    def get_vocab_size(self):
        return len(self.get_vocab())


    def get_vocab(self):
        return self.ix_to_word


    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode # train or test
        # loading annotations
        self.annotations = pkl.load(open(os.path.join(anno_n_splits_dir, 'final_annotations_dict.pkl'), 'rb'))
        print('Annotations: ', self.annotations)
        if self.mode == 'train':
            # loading train set keys
            self.keys = pkl.load(
                open(os.path.join(anno_n_splits_dir, 'train_split_' + str(randomseed) + '.pkl'), 'rb'))
            print('train set keys: ', self.keys)
        elif self.mode == 'test':
            # loading test set keys
            self.keys = pkl.load(
                open(os.path.join(anno_n_splits_dir, 'test_split_' + str(randomseed) + '.pkl'), 'rb'))
            print('test set keys: ', self.keys)
        else:
            print('Currently supporting only 2 modes: train and test. Choose one of them.')

        if with_caption:
            ################ loading captions ######################
            self.captions = pkl.load(open(os.path.join(anno_n_splits_dir, 'final_captions_dict.pkl'), 'rb'))
            info = json.load(open(os.path.join(anno_n_splits_dir, 'vocab.json'), 'rb'))
            self.ix_to_word = info['ix_to_word']
            self.word_to_ix = info['word_to_ix']
            print('Vocab size: ', len(self.ix_to_word))
            self.max_cap_len = max_cap_len
            print('Max seq. len in data is: ', self.max_cap_len)


    def __getitem__(self, ix):
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image_list = sorted((glob.glob(os.path.join(dataset_frames_dir,
                                                    str('{:02d}'.format(self.keys[ix][0])), '*.jpg'))))
        end_frame = self.annotations.get(self.keys[ix]).get('end_frame')
        # temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = random.randint(temporal_aug_min, temporal_aug_max)
            end_frame = end_frame + temporal_aug_shift
        start_frame = end_frame - sample_length # presently using sample_length number of frames

        # # spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0,1)

        images = torch.zeros(16, C, H, W)
        image_no = 0
        for i in np.arange(start_frame, end_frame, 6.5):
            i = int(i)
            if self.mode == 'train':
                images[image_no] = load_image_train(image_list[i], hori_flip, transform)
            if self.mode == 'test':
                images[image_no] = load_image(image_list[i], transform)
            image_no += 1

        label_final_score = self.annotations.get(self.keys[ix]).get('final_score')
        label_position = self.annotations.get(self.keys[ix]).get('position')
        label_armstand = self.annotations.get(self.keys[ix]).get('armstand')
        label_rot_type = self.annotations.get(self.keys[ix]).get('rotation_type')
        label_ss_no = self.annotations.get(self.keys[ix]).get('ss_no')
        label_tw_no = self.annotations.get(self.keys[ix]).get('tw_no')
        if with_caption:
            ########## loading captions ############
            label_captions = np.zeros(self.max_cap_len)
            label_captions_mask = np.zeros(self.max_cap_len)
            captions = self.captions.get(self.keys[ix])
            if captions is None:
                print('Fault in caps for: ', self.keys[ix])
            if len(captions) > self.max_cap_len:
                captions = captions[:self.max_cap_len]
            captions[-1] = '<eos>'
            for j, w in enumerate(captions):
                label_captions[j] = self.word_to_ix[w]
            label_captions_non_zero = (label_captions == 0).nonzero()
            label_captions_mask[:int(label_captions_non_zero[0][0]) + 1] = 1

        data = {}
        data['video'] = images
        data['label_position'] = label_position; data['label_armstand'] = label_armstand
        data['label_rot_type'] = label_rot_type; data['label_ss_no'] = label_ss_no; data['label_tw_no'] = label_tw_no
        data['label_final_score'] = label_final_score/final_score_std
        if with_caption:
            ########## caption stuff #############
            data['label_captions'] = torch.from_numpy(label_captions).type(torch.LongTensor)
            data['label_captions_mask'] = torch.from_numpy(label_captions_mask).type(torch.FloatTensor)
        return data


    def __len__(self):
        return len(self.keys)