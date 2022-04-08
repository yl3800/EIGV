import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../')
from utils.util import pause
from networks.q_v_transformer import CoAttention, PositionalEncoding
from networks.gcn import AdjLearner, GCN #, GAT
from networks.mem_bank import AttentionScore, MemBank
from networks.util import length_to_mask
from block import fusions #pytorch >= 1.1.0
from utils.util import pause


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

        if fg_mask is not None:
            fg_mask_ = fg_mask.clone()
            # stack left 
            temp=vid_feats.new_zeros(vid_feats.size())
            for i, (vid_feat_i, fg_mask_i)  in enumerate(zip(vid_feats, fg_mask)):
                fg_len_i=fg_mask_i.sum(-1)
                # if no fg frame, manualy set allframe to be fg
                if fg_len_i == 0:
                    fg_len_i = fg_mask_i.size(0)
                    fg_mask_i = fg_mask_i.new_ones(fg_mask_i.size()) 
                    fg_mask_[i,:]=fg_mask_i
                temp[i, :fg_len_i, :] = vid_feat_i[fg_mask_i, :] # assemble value to left, [1,0,2,0,3]-->[1,2,3,0,0]
            vid_feats = pack_padded_sequence(temp, fg_mask_.cpu().sum(-1), batch_first=True, enforce_sorted=False)

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

        if fg_mask is not None:
            return foutput, fhidden, fg_mask_.to(foutput.dtype)
        else:
            return foutput, fhidden
        # return foutput, fhidden


class HGA(nn.Module):
    def __init__(self, n_pos, n_neg, vocab_num, mot_feat, hidden_dim = 512,  word_dim = 512, input_dropout_p=0.5, tau=1, num_layers=1, is_gru=False):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        """
        super(HGA, self).__init__()
        vid_dim = 2048 + 2048
        self.num_pos = n_pos
        self.num_neg = n_neg
        self.tau=tau
        self.vid_encoder = EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=input_dropout_p,bidirectional=True, is_gru=is_gru)
        self.qns_encoder = EncoderQns(word_dim, hidden_dim, n_layers=1,rnn_dropout_p=0, input_dropout_p=input_dropout_p, bidirectional=True, is_gru=is_gru)

        hidden_size = self.vid_encoder.dim_hidden*2 
        input_dropout_p = self.vid_encoder.input_dropout_p

        self.fg_att = AttentionScore(hidden_size)
        self.bg_att = AttentionScore(hidden_size)

        self.mem_swap = MemBank(mot_feat)
        
        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=input_dropout_p)

        self.atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-2))

        self.agg_pools = nn.Sequential(*[nn.MaxPool1d(kernel_size=i+2, stride=1) for i in range(n_pos-1)])

        self.global_fusion = fusions.Block([hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)
        self.fusion = fusions.Block([hidden_size, hidden_size], hidden_size)
        self. project_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size,hidden_size))
        self.decoder=nn.Linear(hidden_size, vocab_num+1) # ans_num+<unk>

    def forward(self, v, q, q_len, vid_idx, *args):
        """

        :param vid_feats:[bs, 16, 4096]
        :param qns: [bs, 20, 768]
        :param qns_lengths:[bs,]
        :return:
        """
        
        B=v.shape[0]

        ## encode q,v
        q_local_, q_global_ = self.qns_encoder(q, q_len)
        v_local_, v_global_ = self.vid_encoder(v)
        
        if not self.training:
            out = self.fusion_predict(q_global_, v_global_, q_local_, v_local_, q_len) #bs,1853
            return out
            
        # in training
        lam_1, lam_2, index = args
        ## fg/bg att
        fg_mask_, bg_mask_ =self.frame_att(q_global_, v_local_) #[bs, 16]
        
        v1 = lam_1*v*(fg_mask_.unsqueeze(-1)) + lam_2*v*(bg_mask_.unsqueeze(-1)) 
        v2 = (1-lam_1)*v[index]*(fg_mask_[index].unsqueeze(-1)) + (1-lam_2)*v[index]*(bg_mask_[index].unsqueeze(-1)) 

        # mix v,q
        vid_feats = v1+v2
        qns = lam_1*q + (1-lam_1)*q[index]
        qns_lengths = q_len
        vid_idx = torch.stack([vid_idx, vid_idx[index]], -1)
        q_local, q_global = self.qns_encoder(qns, qns_lengths)
        v_local, v_global = self.vid_encoder(vid_feats)

        fg_mask, bg_mask =self.frame_att(q_global, v_local) #[bs, 16]
        out, out_anc = self.fusion_predict(q_global, v_global, q_local, v_local, qns_lengths, require_xe=True) #bs,1853

        ######## pos
        # pos1: sample bg frame, add to original video
        vid_feats_pos1 = self.mem_swap(bg_mask, vid_feats, vid_idx) # bs, 16, d
        v_local_pos1, v_global_pos1 = self.vid_encoder(vid_feats_pos1)
        out_pos1 = self.fusion_predict(q_global, v_global_pos1, q_local, v_local_pos1, qns_lengths) #bs,1853

        # pos2: multi scale
        vid_feats_pos2 = vid_feats.permute(0, 2, 1)     # [bs, 2048, 16] , pool on temporal
        vid_feats_pos2 = list(map(lambda pool: pool(vid_feats_pos2).permute(0, 2, 1), self.agg_pools))
        vid_feats_pos2 = list(map(lambda x: self.vid_encoder(x), vid_feats_pos2))
        out_pos2 = list(map(lambda x:self.fusion_predict(q_global, x[1], q_local, x[0], qns_lengths), vid_feats_pos2))
        out_pos = torch.stack([out_pos1]+out_pos2, 1)

        ######## negs
        # neg1: rand_V(swap fg) + Q
        vid_feats_neg = self.mem_swap(fg_mask.repeat(self.num_neg,1), vid_feats.repeat(self.num_neg,1,1), vid_idx.repeat(self.num_neg,1)) # bsxnum_neg, 16, d
        v_local_neg, v_global_neg = self.vid_encoder(vid_feats_neg)
        out_neg1 = self.fusion_predict(q_global.repeat(self.num_neg,1), v_global_neg, q_local.repeat(self.num_neg,1,1), v_local_neg, qns_lengths.repeat(self.num_neg)) # bsxnum_neg, 1853

        # neg2: V + rand_Q(swap question)
        index_q = torch.randperm(B*self.num_neg)
        q_local_neg, q_global_neg, qns_lengths_neg = q_local.repeat(self.num_neg,1,1)[index_q,:,:], q_global.repeat(self.num_neg,1)[index_q,:], qns_lengths.repeat(self.num_neg)[index_q]
        out_neg2 = self.fusion_predict(q_global_neg, v_global.repeat(self.num_neg,1), q_local_neg, v_local.repeat(self.num_neg,1,1), qns_lengths_neg)

        # # rand_v(index) + Q
        # index_ = torch.randperm(B*self.num_neg)
        # v_local_neg_, v_global_neg_ = v_local.repeat(self.num_neg,1,1)[index_,:,:], v_global.repeat(self.num_neg,1)[index_,:]
        # out_neg3=self.fusion_predict(q_global.repeat(self.num_neg,1), v_global_neg_, q_local.repeat(self.num_neg,1,1), v_local_neg_, qns_lengths.repeat(self.num_neg))

        out_neg = torch.cat((out_neg1,out_neg2), dim=0).view(B, -1, out_neg1.shape[-1])
        return out, out_anc, out_pos, out_neg



    def frame_att(self, q_global, v_local):

        fg_score = self.fg_att(q_global.unsqueeze(1), v_local) #[bs, 1, 16]
        # bg_score = 1-fg_score
        bg_score = self.bg_att(q_global.unsqueeze(1), v_local)

        # gumbel_softmax, try tau 1-10
        score=torch.cat((fg_score,bg_score),1)#[bs, 2, 16]
        score=F.gumbel_softmax(score, tau=self.tau, hard=True, dim=1) #[bs, 2, 16]

        fg_mask=score[:,0,:]#[bs, 16]
        bg_mask=score[:,1,:]#[bs, 16]

        return fg_mask, bg_mask



    def fusion_predict(self, q_global, v_global, q_local, v_local, q_len, require_xe=False, v_len=None, **kwargs):
        '''
            q_global: bs, hidden
            q_local: bs, max_seq_len_in_this_batch, hidden
            v_global: bs, hidden
            q_local: bs, max_seg_num_in_this_batch, hidden
            v_len: [bs,] number of fg/bg frame in each sample 
            q_len: [bs,] number of token in each sample 
            '''

        # ## gcn
        adj = self.adj_learner(q_local, v_local)
        q_v_inputs = torch.cat((q_local, v_local), dim=1)
        q_v_local=self.gcn(q_v_inputs, adj)

        ## attention pool with mask (if applicable)
        local_attn = self.atten_pool(q_v_local) # [bs, 23, 1]
        q_mask = length_to_mask(q_len, q_local.size(1))
        if v_len is not None: # both qv need mask
            v_mask = length_to_mask(v_len, v_local.size(1))
            pool_mask = torch.cat((q_mask, v_mask), dim=-1).unsqueeze(-1) # bs,20+16,1
        else: # only q need mask
            pool_mask=torch.cat((q_mask, v_local.new_ones(v_local.size()[:2])), dim=-1).unsqueeze(-1) # bs,len_q+16,1
        local_out = torch.sum(q_v_local * pool_mask * local_attn, dim=1) # bs, hidden

        ## fusion
        global_out = self.global_fusion((q_global, v_global))
        out = self.fusion((global_out, local_out)).squeeze() # bs x hidden
        
        out_cl=self.project_head(out)
        out_xe=self.decoder(out)

        if not self.training:
            return out_xe
        
        if not require_xe:
            return out_cl
        
        return out_xe, out_cl

        
        
if __name__ == "__main__":
    videos=torch.rand(2,16,4096).cuda()
    qas=torch.cat((torch.rand(2,7,768),torch.zeros(2,13,768)), dim=1).cuda()
    qas_lengths=torch.tensor([7,7],dtype=torch.int64).cuda() #torch.randint(5, 20, (32,))
    vid_idx=torch.tensor([7,7],dtype=torch.int64).cuda()
    lam_1 = np.random.beta(1,1)
    lam_2 = np.random.beta(1,1)
    index = torch.randperm(videos.size(0))

    model=HGA(3,2,1852,'resnext',4).cuda()
    out, out_anc, out_pos, out_neg = model(videos, qas, qas_lengths, vid_idx, lam_1, lam_2, index)
    print(out_anc.shape, out_pos.shape, out_neg.shape)