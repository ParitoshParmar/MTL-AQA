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

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_MSCADC import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.MSCADC.body import C3D_dilated_body
from models.MSCADC.head_fs_2 import C3D_dilated_head_fs
from models.MSCADC.head_dive_classifier import C3D_dilated_head_classifier
from models.MSCADC.head_captions import S2VTModel
from opts import *
from utils import utils_1
import numpy as np

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def train_phase(train_dataloader, optimizer, criterions, epoch):
    criterion_final_score = criterions['criterion_final_score']
    penalty_final_score = criterions['penalty_final_score']
    if with_dive_classification:
        criterion_dive_classifier = criterions['criterion_dive_classifier']
    if with_caption:
        criterion_caption = criterions['criterion_caption']

    model_CNN.train()
    model_score_regressor.train()
    if with_dive_classification:
        model_classifier.train()
    if with_caption:
        model_caption.train()

    iteration = 0
    for data in train_dataloader:
        true_final_score = data['label_final_score'].type(torch.FloatTensor).cuda()
        if with_dive_classification:
            true_postion = data['label_position'].cuda()
            true_armstand = data['label_armstand'].cuda()
            true_rot_type = data['label_rot_type'].cuda()
            true_ss_no = data['label_ss_no'].cuda()
            true_tw_no = data['label_tw_no'].cuda()
        if with_caption:
            true_captions = data['label_captions'].cuda()
            true_captions_mask = data['label_captions_mask'].cuda()
        video = data['video'].transpose_(1, 2).cuda()
        batch_size, C, frames, H, W = video.shape

        sample_feats_fc6 = model_CNN(video)

        if with_dive_classification:
            pred_position, pred_armstand, pred_rot_type, pred_ss_no, pred_tw_no = model_classifier(sample_feats_fc6)

        if with_caption:
            seq_probs, _= model_caption(sample_feats_fc6, true_captions, 'train')

        pred_final_score = model_score_regressor(sample_feats_fc6)

        loss_final_score = (criterion_final_score(pred_final_score, true_final_score)
                            + penalty_final_score(pred_final_score, true_final_score))

        loss = 0
        loss += loss_final_score
        if with_dive_classification:
            loss_position = criterion_dive_classifier(pred_position, true_postion)
            loss_armstand = criterion_dive_classifier(pred_armstand, true_armstand)
            loss_rot_type = criterion_dive_classifier(pred_rot_type, true_rot_type)
            loss_ss_no = criterion_dive_classifier(pred_ss_no, true_ss_no)
            loss_tw_no = criterion_dive_classifier(pred_tw_no, true_tw_no)
            loss_cls = loss_position + loss_armstand + loss_rot_type + loss_ss_no + loss_tw_no
            loss += loss_cls
        if with_caption:
            loss_caption = criterion_caption(seq_probs, true_captions[:, 1:], true_captions_mask[:, 1:])
            loss += loss_caption*0.01

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            print('Epoch:', epoch, '    Iter:', iteration, '    Loss:',
                  loss.data.cpu().numpy(), '    FS Loss:', loss_final_score.data.cpu().numpy(), end="")
            if with_dive_classification:
                  print('   Cls Loss:', loss_cls.data.cpu().numpy(), end="")
            if with_caption:
                  print('   Cap Loss:', loss_caption.data.cpu().numpy(), end="")
            print(' ')
        iteration += 1


def test_phase(test_dataloader):
    with torch.no_grad():
        pred_scores = []; true_scores = []
        if with_dive_classification:
            pred_position = []; pred_armstand = []; pred_rot_type = []; pred_ss_no = []; pred_tw_no = []
            true_position = []; true_armstand = []; true_rot_type = []; true_ss_no = []; true_tw_no = []

        model_CNN.eval()
        model_score_regressor.eval()
        if with_dive_classification:
            model_classifier.eval()
        if with_caption:
            model_caption.eval()

        for data in test_dataloader:
            true_scores.extend(data['label_final_score'].data.numpy())
            if with_dive_classification:
                true_position.extend(data['label_position'].numpy())
                true_armstand.extend(data['label_armstand'].numpy())
                true_rot_type.extend(data['label_rot_type'].numpy())
                true_ss_no.extend(data['label_ss_no'].numpy())
                true_tw_no.extend(data['label_tw_no'].numpy())
            video = data['video'].transpose_(1, 2).cuda()

            batch_size, C, frames, H, W = video.shape

            sample_feats_fc6 = model_CNN(video)

            if with_caption:
                seq_probs, _ = model_caption(sample_feats_fc6, mode = 'inference')

            if with_dive_classification:
                temp_position, temp_armstand, temp_rot_type, temp_ss_no, temp_tw_no = model_classifier(sample_feats_fc6)
                softmax_layer = nn.Softmax(dim=1)
                temp_position = softmax_layer(temp_position).data.cpu().numpy()
                temp_armstand = softmax_layer(temp_armstand).data.cpu().numpy()
                temp_rot_type = softmax_layer(temp_rot_type).data.cpu().numpy()
                temp_ss_no = softmax_layer(temp_ss_no).data.cpu().numpy()
                temp_tw_no = softmax_layer(temp_tw_no).data.cpu().numpy()

                for i in range(len(temp_position)):
                    pred_position.extend(np.argwhere(temp_position[i] == max(temp_position[i]))[0])
                    pred_armstand.extend(np.argwhere(temp_armstand[i] == max(temp_armstand[i]))[0])
                    pred_rot_type.extend(np.argwhere(temp_rot_type[i] == max(temp_rot_type[i]))[0])
                    pred_ss_no.extend(np.argwhere(temp_ss_no[i] == max(temp_ss_no[i]))[0])
                    pred_tw_no.extend(np.argwhere(temp_tw_no[i] == max(temp_tw_no[i]))[0])

            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend(temp_final_score.data.cpu().numpy())

        if with_dive_classification:
            position_correct = 0; armstand_correct = 0; rot_type_correct = 0; ss_no_correct = 0; tw_no_correct = 0
            for i in range(len(pred_position)):
                if pred_position[i] == true_position[i]:
                    position_correct += 1
                if pred_armstand[i] == true_armstand[i]:
                    armstand_correct += 1
                if pred_rot_type[i] == true_rot_type[i]:
                    rot_type_correct += 1
                if pred_ss_no[i] == true_ss_no[i]:
                    ss_no_correct += 1
                if pred_tw_no[i] == true_tw_no[i]:
                    tw_no_correct += 1
            position_accu = position_correct / len(pred_position) * 100
            armstand_accu = armstand_correct / len(pred_armstand) * 100
            rot_type_accu = rot_type_correct / len(pred_rot_type) * 100
            ss_no_accu = ss_no_correct / len(pred_ss_no) * 100
            tw_no_accu = tw_no_correct / len(pred_tw_no) * 100
            print('Accuracies: Position: ', position_accu, ' Armstand: ', armstand_accu, ' Rot_type: ', rot_type_accu,
                  ' SS_no: ', ss_no_accu, ' TW_no: ', tw_no_accu)

        rho, p = stats.spearmanr(pred_scores, true_scores)
        mse = ((np.subtract(pred_scores, true_scores) * final_score_std) ** 2).mean()
        print('Predicted scores: ', pred_scores)
        print('True scores: ', true_scores)
        print('Correlation: ', rho, '   |   MSE: ', mse)


def main():
    parameters_2_optimize = (list(model_CNN.parameters()) + list(model_score_regressor.parameters()))
    parameters_2_optimize_named = (list(model_CNN.named_parameters()) + list(model_score_regressor.named_parameters()))
    if with_dive_classification:
        parameters_2_optimize = (parameters_2_optimize + list(model_classifier.parameters()))
        parameters_2_optimize_named = (parameters_2_optimize_named + list(model_classifier.parameters()))
    if with_caption:
        parameters_2_optimize = parameters_2_optimize + list(model_caption.parameters())
        parameters_2_optimize_named = parameters_2_optimize_named + list(model_caption.named_parameters())

    learning_rate = base_learning_rate
    optimizer = optim.Adam(parameters_2_optimize, lr=learning_rate)
    print('Parameters that will be learnt: ', parameters_2_optimize_named)

    criterions = {}
    criterion_final_score = nn.MSELoss()
    penalty_final_score = nn.L1Loss()
    criterions['criterion_final_score'] = criterion_final_score
    criterions['penalty_final_score'] = penalty_final_score
    if with_dive_classification:
        criterion_dive_classifier = nn.CrossEntropyLoss()
        criterions['criterion_dive_classifier'] = criterion_dive_classifier
    if with_caption:
        criterion_caption = utils_1.LanguageModelCriterion()
        criterions['criterion_caption'] = criterion_caption

    train_dataset = VideoDataset('train')
    test_dataset = VideoDataset('test')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    print('Length of train loader: ', len(train_dataloader))
    print('Length of test loader: ', len(test_dataloader))
    print('Training set size: ', len(train_dataset.keys), ';    Test set size: ', len(test_dataset.keys))

    # actual training, testing loops
    for epoch in range(max_epochs):
        saving_dir = '...'
        print('-------------------------------------------------------------------------------------------------------')
        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])

        train_phase(train_dataloader, optimizer, criterions, epoch)
        test_phase(test_dataloader)

        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_score_regressor, 'model_score_regressor', epoch, saving_dir)
            if with_dive_classification:
                save_model(model_classifier, 'model_position_classifier', epoch, saving_dir)
            if with_caption:
                save_model(model_caption, 'model_caption', epoch, saving_dir)



if __name__ == '__main__':
    # loading C3D backbone
    model_CNN_pretrained_dict = torch.load('C3D_small_PyTorch_Trained_12.pth')
    model_CNN = C3D_dilated_body()
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    model_CNN = model_CNN.cuda()

    # loading our score regressor
    model_score_regressor = C3D_dilated_head_fs()
    model_score_regressor = model_score_regressor.cuda()
    print('Using Final Score Loss')

    if with_dive_classification:
        # loading our dive classifier
        model_classifier = C3D_dilated_head_classifier()
        model_classifier = model_classifier.cuda()
        print('Using Dive Classification Loss')

    if with_caption:
        # loading our caption model
        model_caption = S2VTModel(vocab_size, max_cap_len, caption_lstm_dim_hidden,
                                  caption_lstm_dim_word, caption_lstm_dim_vid,
                                  rnn_cell=caption_lstm_cell_type, n_layers=caption_lstm_num_layers,
                                  rnn_dropout_p=caption_lstm_dropout)
        model_caption = model_caption.cuda()
        print('Using Captioning Loss')

    main()