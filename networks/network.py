import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../')
from networks.q_v_transformer import CoAttention, PositionalEncoding
from networks.gcn import AdjLearner, GCN #, GAT
from networks.mem_bank import AttentionScore, MemBank
from networks.util import length_to_mask
from block import fusions #pytorch >= 1.1.0


class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, is_gru=False):

        super(EncoderQns, self).__init__()
        self.is_gru= is_gru
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.q_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)
        self.input_dropout = nn.Dropout(input_dropout_p)
        if is_gru:
            self.rnn_cell = nn.GRU
        else:
            self.rnn_cell = nn.LSTM
            
        # self.embedding = nn.Linear(768, dim_embed)
        self.embedding = nn.Sequential(nn.Linear(768, dim_embed),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding[0].weight) 


    def forward(self, qns, qns_lengths):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """

        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths.cpu(), batch_first=True, enforce_sorted=False)
        if self.is_gru:
            packed_output, hidden = self.rnn(packed)
        else:
            packed_output, (hidden, _) = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # hidden = torch.squeeze(hidden)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1) #hidden.reshape(hidden.size()[1], -1)
        output = self.q_input_ln(output) # bs,q_len,hidden_dim
        return output, hidden



class EncoderVidHGA(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, is_gru=False):
        """
        """
        self.is_gru=is_gru
        super(EncoderVidHGA, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.v_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)

        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))
        if is_gru:
            self.rnn_cell = nn.GRU
        else:
            self.rnn_cell = nn.LSTM

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight) 


    def forward(self, vid_feats, fg_mask =None):
        """
        vid_feats: (bs, 16, 4096)
        fg_mask: (bs, 16,) bool mask
        """
        
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats_trans = self.vid2hid(vid_feats.view(-1, self.dim_vid))
        vid_feats = vid_feats_trans.view(batch_size, seq_len, -1)

        # self.rnn.flatten_parameters() # for parallel
        if self.is_gru:
            foutput, fhidden = self.rnn(vid_feats)
        else:
            foutput, (fhidden,_) = self.rnn(vid_feats)

        if fg_mask is not None:
            foutput, _ = pad_packed_sequence(foutput, batch_first=True)

        # fhidden = torch.squeeze(fhidden)
        fhidden = torch.cat([fhidden[0], fhidden[1]], dim=-1) #fhidden.reshape(fhidden.size()[1], -1)
        foutput = self.v_input_ln(foutput) # bs,16,hidden_dim
        return foutput, fhidden


class VideoQANetwork(nn.Module):
    def __init__(self, **kwargs):
        super(VideoQANetwork, self).__init__()
        self.app_pool5_dim = kwargs.pop("app_pool5_dim")
        self.num_frames = kwargs.pop("num_frames")
        self.word_dim = kwargs.pop("word_dim")
        self.module_dim = kwargs.pop("module_dim")
        self.dropout = kwargs.pop("dropout")
        self.num_neg = kwargs.pop("neg")
        self.tau_gumbel = kwargs.pop("tau_gumbel")
        hidden_dim = self.module_dim
        
        vid_dim = self.app_pool5_dim*2
        self.vid_encoder = EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=self.dropout,bidirectional=True, is_gru=False)
        self.qns_encoder = EncoderQns(self.word_dim, hidden_dim, n_layers=1,rnn_dropout_p=0, input_dropout_p=self.dropout, bidirectional=True, is_gru=False)

        hidden_size = self.vid_encoder.dim_hidden*2 

        self.fg_att = AttentionScore(hidden_size)
        self.bg_att = AttentionScore(hidden_size)

        self.mem_swap = MemBank()
        
        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=self.dropout)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=1,
            dropout=self.dropout)

        self.atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-2))

        # self.agg_pools = nn.Sequential(*[nn.MaxPool1d(kernel_size=i+2, stride=1) for i in range(n_pos-1)])

        self.global_fusion = fusions.Block([hidden_size, hidden_size], hidden_size, dropout_input=self.dropout)
        self.fusion = fusions.Block([hidden_size, hidden_size], hidden_size)
        self. project_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size,hidden_size))
        self.decoder=nn.Linear(hidden_size, 1) # ans_num+<unk>

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
        q_local_, q_global_ = self.qns_encoder(q, q_len) # bsx5, 512        
        v_local_, v_global_ = self.vid_encoder(v)

        if not self.training:
            out = self.fusion_predict(q_global_,torch.repeat_interleave(v_global_, 5, dim=0), q_local_, torch.repeat_interleave(v_local_, 5, dim=0), q_len, mode='xe') # bsx5,1 
            return out.view(B, 5)

        # in training
        vid_idx, lam_1, lam_2, index, ans = args

        # get gt qas
        q_len_gt = q_len.view(B,5)[torch.arange(B).cuda(),ans] # B,
        q_feat_gt_global = q_global_.view(B, 5, -1)[torch.arange(B).cuda(),ans,:] # bs, 512
        q_feat_gt_local = q_local_.view(B, 5, q_local_.size(-2), q_local_.size(-1))[torch.arange(B).cuda(),ans,:, :] # bs, 37, 512
        fg_mask_, bg_mask_ = self.frame_att(q_feat_gt_global, v_local_) #[bs, 16]

        v1 = lam_1*v*(fg_mask_.unsqueeze(-1)) + lam_2*v*(bg_mask_.unsqueeze(-1)) 
        v2 = (1-lam_1)*v[index]*(fg_mask_[index].unsqueeze(-1)) + (1-lam_2)*v[index]*(bg_mask_[index].unsqueeze(-1)) 
        
        # mix v,q
        v = v1 + v2
        q_feat_gt_global = lam_1*q_feat_gt_global + (1-lam_1)*q_feat_gt_global[index]
        q_feat_gt_local = lam_1*q_feat_gt_local + (1-lam_1)*q_feat_gt_local[index]
        vid_idx = torch.stack([vid_idx, vid_idx[index]], -1)
        
        # prediction in training
        v_local, v_global = self.vid_encoder(v)      
        q_len = torch.cat([q_len.view(B, 5), q_len.view(B, 5)[index]], 1) # bs, 5x2
        q_global = torch.cat([q_global_.view(B, 5, -1), q_global_.view(B, 5, -1)[index]], 1) # bs, 5x2, 512
        q_local = torch.cat([q_local_.view(B, 5, q_local_.size(-2), q_local_.size(-1)), q_local_.view(B, 5, q_local_.size(-2), q_local_.size(-1))[index]], 1) # bs, 5x2, 512
        
        out = self.fusion_predict(q_global.view(-1, q_global.size(-1)), torch.repeat_interleave(v_global, 5*2, dim=0),\
                                   q_local.view(-1, q_local.size(-2), q_local.size(-1)), torch.repeat_interleave(v_local, 5*2, dim=0), q_len.view(-1), mode='xe') # bsx5x2,1 
        
        ### compute on mixed data ###
        # anchor
        out_anc = self.fusion_predict(q_feat_gt_global, v_global, q_feat_gt_local, v_local, q_len_gt, mode='cl')    # bs,1853
        fg_mask, bg_mask =self.frame_att(q_feat_gt_global, v_local)            # bs,16

        # pos1: sample bg frame, add to original video
        v_pos = self.mem_swap(bg_mask, v, vid_idx) # bs, 16, d
        v_local_pos1, v_global_pos1 = self.vid_encoder(v_pos)
        out_pos = self.fusion_predict(q_feat_gt_global, v_global_pos1, q_feat_gt_local, v_local_pos1, q_len_gt) #bs,1853

        # neg1: rand_V(swap fg) + Q
        v_neg = self.mem_swap(fg_mask.repeat(self.num_neg,1), v.repeat(self.num_neg,1,1), vid_idx.repeat(self.num_neg,1)) # bsxnum_neg, 16, 4096
        v_local_neg, v_global_neg = self.vid_encoder(v_neg)                                                                                  # bsxnum_neg, 16, d
        out_neg1 = self.fusion_predict(q_feat_gt_global.repeat(self.num_neg,1), v_global_neg, q_feat_gt_local.repeat(self.num_neg,1,1), v_local_neg, q_len_gt.repeat(self.num_neg))                                      # bsxnum_neg, 512

        #neg3: V + randQ(swap gt_qas)
        index_q = torch.randperm(B*self.num_neg)
        q_global_neg = q_feat_gt_global.repeat(self.num_neg,1)[index_q,:]
        q_local_neg = q_feat_gt_local.repeat(self.num_neg,1,1)[index_q,:,:]
        q_len_neg = q_len_gt.repeat(self.num_neg)[index_q]
        out_neg3 = self.fusion_predict(q_global_neg, v_global.repeat(self.num_neg,1), q_local_neg, v_local.repeat(self.num_neg,1,1), q_len_neg) 

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


    def fusion_predict(self, q_global, v_global, q_local, v_local, q_len, mode='cl'):
        # ## gcn
        adj = self.adj_learner(q_local, v_local)
        q_v_inputs = torch.cat((q_local, v_local), dim=1)
        q_v_local=self.gcn(q_v_inputs, adj)

        ## attention pool with mask (if applicable)
        local_attn = self.atten_pool(q_v_local) # [bs, 23, 1]
        q_mask = length_to_mask(q_len, q_local.size(1))
        pool_mask=torch.cat((q_mask, v_local.new_ones(v_local.size()[:2])), dim=-1).unsqueeze(-1) # bs,len_q+16,1
        local_out = torch.sum(q_v_local * pool_mask * local_attn, dim=1) # bs, hidden

        ## fusion
        global_out = self.global_fusion((q_global, v_global))
        out = self.fusion((global_out, local_out)).squeeze() # bs x hidden
        
        out_cl=self.project_head(out)
        out_xe=self.decoder(out)

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
    parser.add_argument('-ans_num', default=1, type=int)

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

    model = VideoQANetwork(**model_kwargs)
    # model.eval()

    videos=torch.rand(2,16,4096)
    qas=torch.cat((torch.rand(2,5,17,768),torch.zeros(2, 5, 20,768)), dim=2)
    qas_lengths=torch.tensor([7,10,18,7,10],dtype=torch.int64).unsqueeze(0).repeat(2,1)# bs 
    answers=torch.tensor([0,1],dtype=torch.int64)
    qns_keys=None
    vid_idx=torch.tensor([7,7],dtype=torch.int64)
    index = torch.randperm(videos.size(0))
    out, ou = model(videos, qas, qas_lengths, vid_idx,0.7,0.6, index, answers)
    print(out.shape)