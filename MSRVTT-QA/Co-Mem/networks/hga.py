from cgi import print_environ
import numpy as np
import torch
import torch.nn as nn
# import random as rd
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../')
from networks.q_v_transformer import CoAttention
from networks.gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0
from networks.memory_module import EpisodicMemory
from networks.mem_bank import AttentionScore, MemBank

class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):

        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # self.embedding = nn.Linear(768, dim_embed)
        self.embedding = nn.Sequential(nn.Linear(768, dim_embed),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)
                                
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding[0].weight) 

    def forward(self, qns, qns_lengths, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed, hidden) 
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return  hidden


class EncoderVidCoMem(nn.Module):
    def __init__(self, dim_app, dim_motion, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidCoMem, self).__init__()
        self.dim_app = dim_app
        self.dim_motion = dim_motion
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn_app_l1 = self.rnn_cell(self.dim_app, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_app_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn_motion_l1 = self.rnn_cell(self.dim_motion, dim_hidden, n_layers, batch_first=True,
                                            bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_motion_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                           bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):
        """
        two separate LSTM to encode app and motion feature
        :param vid_feats:
        :return:
        """
        vid_app = vid_feats[:, :, 0:self.dim_app]
        vid_motion = vid_feats[:, :, self.dim_app:]

        app_output_l1, _ = self.rnn_app_l1(vid_app)
        app_output_l2, _ = self.rnn_app_l2(app_output_l1)


        motion_output_l1, _ = self.rnn_motion_l1(vid_motion)
        motion_output_l2, _ = self.rnn_motion_l2(motion_output_l1)
        
        outputs_app = torch.cat((app_output_l1, app_output_l2), dim=-1)
        outputs_motion = torch.cat((motion_output_l1, motion_output_l2), dim=-1)
        return outputs_app, outputs_motion



class HGA(nn.Module):
    def __init__(self,n_pos, n_neg, vocab_num, mot_feat, hidden_dim = 512,  word_dim = 512, input_dropout_p=0.5, tau=1, num_layers=1, is_gru=False):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI2020)
        """
        super(HGA, self).__init__()
        app_dim = 2048
        motion_dim = 2048
        self.tau=tau
        self.num_pos = n_pos
        self.num_neg = n_neg
        self.vid_encoder = EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=input_dropout_p,bidirectional=False, rnn_cell='gru')
        self.qns_encoder = EncoderQns(word_dim, hidden_dim, n_layers=2,rnn_dropout_p=0.5, input_dropout_p=input_dropout_p, bidirectional=False, rnn_cell='gru')

        self.fg_att = AttentionScore(hidden_dim*2)
        self.bg_att = AttentionScore(hidden_dim*2)

        self.mem_swap = MemBank(mot_feat)

        self.epm_app = EpisodicMemory(hidden_dim*2)
        self.epm_mot = EpisodicMemory(hidden_dim*2)

        self.linear_ma = nn.Linear(hidden_dim*2*3, hidden_dim*2)
        self.linear_mb = nn.Linear(hidden_dim*2*3, hidden_dim*2)
        self.vq2word = nn.Linear(hidden_dim*2*2, hidden_dim*2*2)

        self. project_head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim*2*2, hidden_dim*2))
        self.decoder=nn.Linear(hidden_dim*2*2, vocab_num+1) # ans_num+<unk>

    def forward(self, v, q, q_len, vid_idx, *args):
        """

        :param vid_feats:[bs, 16, 4096]
        :param qns: [bs, 20, 768]
        :param qns_lengths:[bs,]
        :return:
        """

        B = v.size(0)

        ## encode q,v
        q = self.qns_encoder(q, q_len)
        app, mot = self.vid_encoder(v)
        
        if not self.training:    
            out = self.fusion_predict(q, app, mot)
            return out

        # in training
        lam_1, lam_2, index = args
        
        ## fg/bg att
        fg_mask_, bg_mask_ =self.frame_att(q, app+mot) #[bs, 16]

        v1 = lam_1*v*(fg_mask_.unsqueeze(-1)) + lam_2*v*(bg_mask_.unsqueeze(-1)) 
        v2 = (1-lam_1)*v[index]*(fg_mask_[index].unsqueeze(-1)) + (1-lam_2)*v[index]*(bg_mask_[index].unsqueeze(-1)) 
        
        # mix v,q
        v_feat = v1+v2
        q_feat = lam_1*q + (1-lam_1)*q[index]
        vid_idx = torch.stack([vid_idx, vid_idx[index]], -1)

        # anchor
        app_anc, mot_anc = self.vid_encoder(v_feat)
        out, out_anc = self.fusion_predict(q_feat, app_anc, mot_anc, require_xe=True) #bs,1853
        fg_mask, bg_mask =self.frame_att(q_feat, app_anc+mot_anc) #[bs, 16]
        
        ######## pos
        # pos1: sample bg frame, add to original video
        v_feat_pos1 = self.mem_swap(bg_mask, v_feat, vid_idx) # bs, 16, d
        app_pos, mot_pos = self.vid_encoder(v_feat_pos1)
        out_pos = self.fusion_predict(q_feat, app_pos, mot_pos) #bs,1853

        ######## negs
        # neg1: rand_V(swap fg) + Q
        v_feat_neg = self.mem_swap(fg_mask.repeat(self.num_neg,1), v_feat.repeat(self.num_neg,1,1), vid_idx.repeat(self.num_neg,1)) # bsxnum_neg, 16, d
        app_neg, mot_neg = self.vid_encoder(v_feat_neg)
        out_neg1 = self.fusion_predict(q_feat.repeat(self.num_neg,1), app_neg, mot_neg) # bsxnum_neg, 1853

        # neg2: V + rand_Q(swap question)
        index_q = torch.randperm(B*self.num_neg)
        q_feat_neg = q_feat.repeat(self.num_neg,1)[index_q,:]
        out_neg2 = self.fusion_predict(q_feat_neg, app_anc.repeat(self.num_neg,1,1), mot_anc.repeat(self.num_neg,1,1))
        
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

 
    def fusion_predict(self, qns_hidden, outputs_app, outputs_motion, iter_num=3, require_xe=False):

        # batch_size = qns_hidden.size(1)
        # qns_hidden = qns_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)
        m_app = outputs_app[:, -1, :]
        m_mot = outputs_motion[:, -1, :]
        ma, mb = m_app.detach(), m_mot.detach()
        m_app = m_app.unsqueeze(1)
        m_mot = m_mot.unsqueeze(1)
        
        for _ in range(iter_num):
            mm = ma + mb
            m_app = self.epm_app(outputs_app, mm, m_app)
            m_mot = self.epm_mot(outputs_motion, mm, m_mot)
            ma_q = torch.cat((ma, m_app.squeeze(1), qns_hidden), dim=1)
            mb_q = torch.cat((mb, m_mot.squeeze(1), qns_hidden), dim=1)
            ma = torch.tanh(self.linear_ma(ma_q))
            mb = torch.tanh(self.linear_mb(mb_q))

        mem = torch.cat((ma, mb), dim=1)
        out = self.vq2word(mem)

        ## decoder 
        out_cl=self.project_head(out)
        out_xe=self.decoder(out)
        
        if not self.training:
            return out_xe
        
        if not require_xe:
            return out_cl
        
        return out_xe, out_cl  




        
if __name__ == "__main__":
    videos=torch.rand(3,16,4096).cuda()
    qas=torch.cat((torch.rand(3,7,768),torch.zeros(3,13,768)), dim=1).cuda()
    qas_lengths=torch.tensor([7,7,7],dtype=torch.int64).cuda() #torch.randint(5, 20, (32,))
    vid_idx=torch.tensor([7,7,7],dtype=torch.int64).cuda()
    lam_1 = np.random.beta(1,1)
    lam_2 = np.random.beta(1,1)
    index = torch.randperm(videos.size(0))

    model=HGA(2,2,1852,'resnext').cuda()
    out, out_anc, out_pos, out_neg = model(videos, qas, qas_lengths, vid_idx, lam_1, lam_2, index)
    print(out_anc.shape, out_pos.shape, out_neg.shape)