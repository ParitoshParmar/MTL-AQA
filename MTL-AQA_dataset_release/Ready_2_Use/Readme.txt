@inproceedings{parmar2019and,
  title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={304--313},
  year={2019}
}

All the directories contain following files:
	1. 'final_annotations_dict.pkl': dictionary containing processed annotations
		It has following keys: {'primary_view',
                                'start_frame',
                                'end_frame',
                                'position',
                                'difficulty',
                                'armstand',
                                'rotation_type',
                                'ss_no',
                                'tw_no',
                                'final_score'}
	2. 'final_captions_dict.pkl': dictionary containing processed captions
	3. 'train_split_0.pkl': list of train samples
	4. 'test_split_0.pkl': list of test samples
	5. 'vocab.json': contains all the unique words


MTL-AQA_split_0_data directory contains above mentioned files related to our dataset and main train/test split.

smaller_training_sets directory contains above mentioned files for smaller training sets; test list file remains same as main split. Number in the name is the number of training samples.

UNLV_Dive_split_4 directory contains above mentioned files for UNLV-Dive dataset [1].
[1] @inproceedings{parmar2017learning,
  title={Learning to score olympic events},
  author={Parmar, Paritosh and Tran Morris, Brendan},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={20--28},
  year={2017}
}