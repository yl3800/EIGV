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
from networks.memory_module import EpisodicMemory
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        return hidden.squeeze(0)


        
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
        app_dim = 2048
        motion_dim = 2048
        v_hidden_dim=256
        mem_hidden_dim = 256
        hidden_dim = self.module_dim
        

        self.fg_att = AttentionScore(self.module_dim)
        self.bg_att = AttentionScore(self.module_dim)
        self.mem_swap = MemBank()

        self.vid_encoder = EncoderVidCoMem(app_dim, motion_dim, v_hidden_dim, input_dropout_p=self.dropout,bidirectional=False, rnn_cell='gru')
        self.qns_encoder = EncoderQns(self.word_dim, hidden_dim, n_layers=1,rnn_dropout_p=0, input_dropout_p=self.dropout, bidirectional=False, rnn_cell='gru')

        self.epm_app = EpisodicMemory(mem_hidden_dim*2)
        self.epm_mot = EpisodicMemory(mem_hidden_dim*2)

        self.linear_ma = nn.Linear(mem_hidden_dim*2*3, mem_hidden_dim*2)
        self.linear_mb = nn.Linear(mem_hidden_dim*2*3, mem_hidden_dim*2)
        self.vq2word = nn.Linear(mem_hidden_dim*2*2, mem_hidden_dim*2*2)

        self.decoder=nn.Linear(mem_hidden_dim*2*2, 1) # ans_num+<unk>
        
        self. project_head = nn.Sequential(nn.ReLU(), nn.Linear(mem_hidden_dim*2*2, mem_hidden_dim*2*2))


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
        q_feat = self.qns_encoder(q, q_len) # bsx5, 512        
        v_local_a, v_local_m = self.vid_encoder(v)

        if not self.training:
            out = self.fusion_predict(q_feat,torch.repeat_interleave(v_local_a, 5, dim=0),torch.repeat_interleave(v_local_m, 5, dim=0), mode='xe') # bsx5,1 
            return out.view(B, 5)

        # in training
        vid_idx, lam_1, lam_2, index, ans = args

        # get gt qas
        q_feat_gt = q_feat.view(B, 5, -1)[torch.arange(B).cuda(),ans,:] # bs, 512
        fg_mask_, bg_mask_ = self.frame_att(q_feat_gt, v_local_a+v_local_m) #[bs, 16]

        v1 = lam_1*v*(fg_mask_.unsqueeze(-1)) + lam_2*v*(bg_mask_.unsqueeze(-1)) 
        v2 = (1-lam_1)*v[index]*(fg_mask_[index].unsqueeze(-1)) + (1-lam_2)*v[index]*(bg_mask_[index].unsqueeze(-1)) 
        
        # mix v,q
        v = v1 + v2
        q_feat_gt = lam_1*q_feat_gt + (1-lam_1)*q_feat_gt[index]
        vid_idx = torch.stack([vid_idx, vid_idx[index]], -1)

        v_a, v_m = self.vid_encoder(v)
        # prediction in training
        # q_feat = torch.cat([lam_1*q_feat.view(B, 5, -1), (1-lam_1)*q_feat.view(B, 5, -1)[index]], 1) # bs, 5x2, 512
        q_feat = torch.cat([q_feat.view(B, 5, -1), q_feat.view(B, 5, -1)[index]], 1) # bs, 5x2, 512
        out = self.fusion_predict(q_feat.view(-1, q_feat.size(-1)),torch.repeat_interleave(v_a, 5*2, dim=0),torch.repeat_interleave(v_m, 5*2, dim=0), mode='xe') # bsx5x2,1 
        ### compute on mixed data ###
        # anchor
        out_anc = self.fusion_predict(q_feat_gt, v_a, v_m, mode='cl')    # bs,1853
        fg_mask, bg_mask =self.frame_att(q_feat_gt, v_a+v_m)            # bs,16

        # pos1: sample bg frame, add to original video
        v_pos1 = self.mem_swap(bg_mask, v, vid_idx) # bs, 16, d
        v_feat_pos_a, v_feat_pos_m = self.vid_encoder(v_pos1)
        out_pos = self.fusion_predict(q_feat_gt, v_feat_pos_a, v_feat_pos_m) #bs,1853

        # neg1: rand_V(swap fg) + Q
        v_neg = self.mem_swap(fg_mask.repeat(self.num_neg,1), v.repeat(self.num_neg,1,1), vid_idx.repeat(self.num_neg,1)) # bsxnum_neg, 16, 4096
        v_feat_neg_a, v_feat_neg_m = self.vid_encoder(v_neg)                                                                                   # bsxnum_neg, 16, d
        out_neg1 = self.fusion_predict(q_feat_gt.repeat(self.num_neg,1), v_feat_neg_a, v_feat_neg_m)                                      # bsxnum_neg, 512

        #neg3: V + randQ(swap gt_qas)
        index_q = torch.randperm(B*self.num_neg)
        q_feat_neg = q_feat_gt.repeat(self.num_neg,1)[index_q,:]
        out_neg3 = self.fusion_predict( q_feat_neg, v_a.repeat(self.num_neg,1,1), v_m.repeat(self.num_neg,1,1)) 

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


    def fusion_predict(self, qns_hidden, outputs_app, outputs_motion, iter_num=3, mode='cl'):
        qns_embed = qns_hidden#.permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)
        m_app = outputs_app[:, -1, :]
        m_mot = outputs_motion[:, -1, :]
        ma, mb = m_app.detach(), m_mot.detach()
        m_app = m_app.unsqueeze(1)
        m_mot = m_mot.unsqueeze(1)
        
        for _ in range(iter_num):
            mm = ma + mb
            m_app = self.epm_app(outputs_app, mm, m_app)
            m_mot = self.epm_mot(outputs_motion, mm, m_mot)
            ma_q = torch.cat((ma, m_app.squeeze(1), qns_embed), dim=1)
            mb_q = torch.cat((mb, m_mot.squeeze(1), qns_embed), dim=1)
            ma = torch.tanh(self.linear_ma(ma_q))
            mb = torch.tanh(self.linear_mb(mb_q))

        mem = torch.cat((ma, mb), dim=1)
        encoder_outputs = self.vq2word(mem)

        ## decoder 
        out_xe=self.decoder(encoder_outputs)
        out_cl=self.project_head(encoder_outputs)

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
    qas_lengths=torch.tensor([7,10,20,7,10],dtype=torch.int64).unsqueeze(0).repeat(2,1)# bs 
    answers=torch.tensor([0,1],dtype=torch.int64)
    qns_keys=None
    vid_idx=torch.tensor([7,7],dtype=torch.int64)
    index = torch.randperm(videos.size(0))
    out, out_anc, out_pos, out_neg = model(videos, qas, qas_lengths, vid_idx,0.7,0.6, index, answers)
    print(out.shape, out_anc.shape, out_pos.shape, out_neg.shape)