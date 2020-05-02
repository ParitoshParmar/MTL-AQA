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

# declaring random seed
randomseed = 0

# directory containing dataset annotation files; this anno_n_splits_dir make the full path
dataset_dir = '...'

# directory tp store train/test split lists and annotations
anno_n_splits_dir = dataset_dir + '...'

# directory containing extracted frames
dataset_frames_dir = '...'

# sample length in terms of no of frames
sample_length = 103

# input data dims; C3D-AVG:112; MSCADC: 180
C, H, W = 3,180,180#3,112,112#
# image resizing dims; C3D-AVG: 171,128; MSCADC: 640,360
input_resize = 640,360#171,128#
# temporal augmentation range
temporal_aug_min = -3; temporal_aug_max = 3

# score std
final_score_std = 17

# maximum caption length
max_cap_len = 100

vocab_size = 5779

caption_lstm_dim_hidden = 512
caption_lstm_dim_word = 512
caption_lstm_dim_vid = 1200#8192# C3D-AVG: 8192; MSCADC: 1200
caption_lstm_cell_type = 'gru'
caption_lstm_num_layers = 2
caption_lstm_dropout = 0.5
caption_lstm_lr = 0.0001

# task 2 include
with_dive_classification = True
with_caption = True

max_epochs = 100

train_batch_size = 3
test_batch_size = 5

model_ckpt_interval = 1 # in epochs

base_learning_rate = 0.0001

temporal_stride = 16