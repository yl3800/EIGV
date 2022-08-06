import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../')
from utils import load_file
import os.path as osp
import numpy as np
import nltk
import pandas as pd
import json
import string
import h5py
import pickle as pkl

class VideoQADataset(Dataset):
    """load the dataset in dataloader
    app+mot_feat:
                [ids] :vid     (3870,)
                [feat]:feature (3870,16,4096) app:mot
    qas_bert_feat:
                [feat]:feature (34132, 5, 37, 768)
    """

    def __init__(self, sample_list_path,video_feature_path, mode):
        self.video_feature_path = video_feature_path
        self.sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(self.sample_list_file)
        self.max_qa_length = 37

        self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))

        vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))
        print('Load {}...'.format(vid_feat_file))
        self.feats = {}
        self.vid2idx={}

        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat'] 
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                self.feats[str(vid)] = feat  # (16, 2048)
                self.vid2idx[str(vid)] = id

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        cur_sample = self.sample_list.iloc[idx]
        video_name, qns, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                    int(cur_sample['answer']), str(cur_sample['qid'])

        # index to embedding
        with h5py.File(self.bert_file, 'r') as fp:
            temp_feat = fp['feat'][idx]
            candidate_qas = torch.from_numpy(temp_feat).type(torch.float32) # (5,37,768)
            qa_lengths=((candidate_qas.sum(-1))!=0.0).sum(-1)

        video_feature = torch.from_numpy(self.feats[video_name]).type(torch.float32)
        qns_key = video_name + '_' + qid

        # get video idx in vid_feat.h5
        vid_idx=self.vid2idx[video_name]

        return video_feature, candidate_qas, qa_lengths, vid_idx,ans, qns_key

if __name__ == "__main__":

    video_feature_path = '/storage_fast/ycli/vqa/qa_feat/next-qa'
    sample_list_path = '/storage_fast/ycli/vqa/qa_dataset/next-qa'
    train_dataset=VideoQADataset(video_feature_path, sample_list_path, 'train')

    train_loader = DataLoader(dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8
        )

    for sample in train_loader:
        video_feature, candidate_qas, qa_lengths, vid_idx,ans, qns_key = sample
        print(video_feature.shape)
        print(candidate_qas.shape)
        print(qa_lengths.shape)
        print(ans.shape)
        print(qns_key.shape)
        print(vid_idx.shape)
        break