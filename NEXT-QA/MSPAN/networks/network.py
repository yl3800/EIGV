from operator import index
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import random
import sys
sys.path.append('../')
from networks.SSP import *
from networks.mem_bank import AttentionScore, MemBank
from utils import pause

class Linguistic_embedding(nn.Module):
    def __init__(self, word_dim=768, module_dim=512, dropout=0.2):
        super(Linguistic_embedding, self).__init__()
        self.encoder = nn.LSTM(word_dim, module_dim // 2, num_layers=3, batch_first=True, bidirectional=True)
        self.dropout_global = nn.Dropout(dropout)

    def forward(self, questions, question_len):
        embed = nn.utils.rnn.pack_padded_sequence(questions, question_len, batch_first=True, enforce_sorted=False)
        self.encoder.flatten_parameters()
        _, (question_global, _) = self.encoder(embed)
        question_global = torch.cat([question_global[0], question_global[1]], dim=-1)  # bs, hidden_size
        question_global = self.dropout_global(question_global)

        return question_global


class VideoQANetwork(nn.Module):
    def __init__(self, **kwargs):
        super(VideoQANetwork, self).__init__()
        self.app_pool5_dim = kwargs.pop("app_pool5_dim")
        self.num_frames = kwargs.pop("num_frames")
        self.word_dim = kwargs.pop("word_dim")
        self.module_dim = kwargs.pop("module_dim")
        self.num_answers = kwargs.pop("num_answers")
        self.num_scale = kwargs.pop("num_scale")
        self.dropout = kwargs.pop("dropout")

        self.GCN_adj = kwargs.pop("GCN_adj")
        self.GAT_adj = kwargs.pop("GAT_adj")
        self.K = kwargs.pop("K")
        self.T = kwargs.pop("T")
        self.num_neg = kwargs.pop("neg")
        self.tau_gumbel = kwargs.pop("tau_gumbel")

        self.agg_pools = nn.Sequential(*[nn.MaxPool1d(kernel_size=i+1, stride=1) for i in range(self.num_scale)])
        self.linguistic = Linguistic_embedding(self.word_dim, self.module_dim, self.dropout)
        self.visual_dim = self.app_pool5_dim

        self.fg_att = AttentionScore(self.module_dim)
        self.bg_att = AttentionScore(self.module_dim)
        self.mem_swap = MemBank()

        self.fc_cat = nn.Linear(self.visual_dim * 2, self.module_dim)
        self.ssp_offer = MSPAN(self.module_dim, self.module_dim, self.num_frames, self.num_scale,
                               self.GCN_adj, self.GAT_adj, self.T, self.K, self.dropout)

        self.que_visual = Fusion(self.module_dim, self.module_dim, use_bias=True, dropout=self.dropout)
        
        self. project_head = nn.Sequential(nn.ReLU(), nn.Linear(self.module_dim, self.module_dim))
        self.classifier = nn.Sequential(
            nn.Linear(self.module_dim, self.module_dim),
            nn.ELU(),
            nn.BatchNorm1d(self.module_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.module_dim, 1))


    def forward(self, v, q, q_len, *args): #videos, qas, qas_lengths, vid_idx, lam_1, lam_2, index, ans
        """
        visual_feat: bs, 16, 4096
        question:   bs,5,37,768
        q_len:      bs,5 
        """
        B = q.size(0)

        # for mc-qa
        q = q.view(-1, q.size(-2), q.size(-1)) # bsx5,37,768
        q_len = q_len.view(-1)
        q_feat = self.linguistic(q, q_len) # bsx5, 512        
        v_feat = self.fc_cat(v)
        
        if not self.training:
            out = self.fusion_predict(torch.repeat_interleave(v_feat, 5, dim=0), q_feat, mode='xe') # bsx5,1 
            return out.view(B, 5)

        # in training
        vid_idx, lam_1, lam_2, index, ans = args

        # get gt qas
        q_feat_gt = q_feat.view(B, 5, -1)[torch.arange(B).cuda(),ans,:] # bs, 512
        fg_mask_, bg_mask_ = self.frame_att(q_feat_gt, v_feat) #[bs, 16]

        v1 = lam_1*v*(fg_mask_.unsqueeze(-1)) + lam_2*v*(bg_mask_.unsqueeze(-1)) 
        v2 = (1-lam_1)*v[index]*(fg_mask_[index].unsqueeze(-1)) + (1-lam_2)*v[index]*(bg_mask_[index].unsqueeze(-1)) 
        
        # mix v,q
        v = v1 + v2
        q_feat_gt = lam_1*q_feat_gt + (1-lam_1)*q_feat_gt[index]
        vid_idx = torch.stack([vid_idx, vid_idx[index]], -1)

        v_feat = self.fc_cat(v) 
        # prediction in training
        # q_feat = torch.cat([lam_1*q_feat.view(B, 5, -1), (1-lam_1)*q_feat.view(B, 5, -1)[index]], 1) # bs, 5x2, 512
        q_feat = torch.cat([q_feat.view(B, 5, -1), q_feat.view(B, 5, -1)[index]], 1) # bs, 5x2, 512
        out = self.fusion_predict(torch.repeat_interleave(v_feat, 5*2, dim=0), q_feat.view(-1, q_feat.size(-1)), mode='xe') # bsx5x2,1 

        ### compute on mixed data ###
        # anchor
        out_anc = self.fusion_predict(v_feat, q_feat_gt, mode='cl')    # bs,1853
        fg_mask, bg_mask =self.frame_att(q_feat_gt, v_feat)            # bs,16

        # pos1: sample bg frame, add to original video
        v_pos1 = self.mem_swap(bg_mask, v, vid_idx) # bs, 16, d
        v_feat_pos1 = self.fc_cat(v_pos1)
        out_pos = self.fusion_predict(v_feat_pos1, q_feat_gt) #bs,1853

        # neg1: rand_V(swap fg) + Q
        v_neg = self.mem_swap(fg_mask.repeat(self.num_neg,1), v.repeat(self.num_neg,1,1), vid_idx.repeat(self.num_neg,1)) # bsxnum_neg, 16, 4096
        v_feat_neg = self.fc_cat(v_neg)                                                                                   # bsxnum_neg, 16, d
        out_neg1 = self.fusion_predict(v_feat_neg, q_feat_gt.repeat(self.num_neg,1))                                      # bsxnum_neg, 512

        #neg3: V + randQ(swap gt_qas)
        index_q = torch.randperm(B*self.num_neg)
        q_feat_neg = q_feat_gt.repeat(self.num_neg,1)[index_q,:]
        out_neg3 = self.fusion_predict(v_feat.repeat(self.num_neg,1,1), q_feat_neg) 

        out_neg = torch.cat((out_neg1,out_neg3), dim=0).view(B, -1, out_neg1.shape[-1])    # bsx(n_neg+n_neg), 512
        return out.view(B,5*2), out_anc, out_pos, out_neg


    def frame_att(self, q_global, v_local):

        fg_score = self.fg_att(q_global.unsqueeze(1), v_local) #[bs, 1, 16]
        # bg_score = 1-fg_score
        bg_score = self.bg_att(q_global.unsqueeze(1), v_local)

        # gumbel_softmax, try tau 1-10
        score=torch.cat((fg_score,bg_score),1)#[bs, 2, 16]
        score=F.gumbel_softmax(score, tau=self.tau_gumbel, hard=True, dim=1) #[bs, 2, 16]

        fg_mask=score[:,0,:]#[bs, 16]
        bg_mask=score[:,1,:]#[bs, 16]

        return fg_mask, bg_mask


    def fusion_predict(self, v_feat, q_feat, mode='cl'):
        graphs = list(map(lambda pool: pool(v_feat.permute(0, 2, 1)).permute(0, 2, 1), self.agg_pools))
        v_feat = self.ssp_offer(graphs, q_feat)
        out = self.que_visual(v_feat, q_feat)  
        out_cl=self.project_head(out)
        out_xe=self.classifier(out)  

        if mode =='xe':
            return out_xe
        elif mode =='cl':
            return out_cl
        else:
            return out_xe, out_cl          


if __name__ == "__main__":
    from utils import make_adjacency
    import argparse

    parser = argparse.ArgumentParser(description="network logger")
    parser.add_argument('-features_type', default=['appearance_pool5_16', 'motion_16'], type=str)
    parser.add_argument('-ans_num', default=1852, type=int)

    # hyper-parameters
    parser.add_argument('-num_scale', default=2, type=int)
    parser.add_argument('-T', default=2, type=int)
    parser.add_argument('-K', default=3, type=int)

    parser.add_argument('-num_frames', default=16, type=int)
    parser.add_argument('-word_dim', default=768, type=int)
    parser.add_argument('-module_dim', default=512, type=int)
    parser.add_argument('-app_pool5_dim', default=2048, type=int)
    parser.add_argument('-motion_dim', default=2048, type=int)

    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument("-neg", type=int, help="#neg_sample", default=5) 
    parser.add_argument("-tau_gumbel", type=float, help="temperature for gumbel_softmax", default=0.9) 
    args = parser.parse_args()

    GCN_adj, GAT_adj = make_adjacency(args)
    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'module_dim': args.module_dim,
        'num_answers': args.ans_num,
        'dropout': args.dropout,
        'GCN_adj': GCN_adj,
        'GAT_adj': GAT_adj,
        "K": args.K,
        "T": args.T,
        'num_scale': args.num_scale,
        'neg': args.neg,
        'tau_gumbel': args.tau_gumbel
    }

    model = VideoQANetwork(**model_kwargs).cuda()

    videos=torch.rand(2,16,4096).cuda()
    qas=torch.cat((torch.rand(2,5,17,768),torch.zeros(2, 5, 20,768)), dim=2).cuda()
    qas_lengths=torch.tensor([7,10,20,7,10],dtype=torch.int64).unsqueeze(0).repeat(2,1).cuda() # bs 
    answers=torch.tensor([0,1],dtype=torch.int64).cuda()
    qns_keys=None
    vid_idx=torch.tensor([7,7],dtype=torch.int64).cuda()
    index = torch.randperm(videos.size(0))
    out, out_anc, out_pos, out_neg = model(videos, qas, qas_lengths, vid_idx,0.7,0.6, index, answers)
    print(out.shape, out_anc.shape, out_pos.shape, out_neg.shape)