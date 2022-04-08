import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import sys
sys.path.append('../')
from networks.gcn import AdjLearner#, GCN
from utils import pause

class Fusion(nn.Module):
    def __init__(self, visual_dim, module_dim, use_bias=False, concat=False,
                 with_score=False, dropout=0.2):
        super(Fusion, self).__init__()
        self.use_bias = use_bias
        self.concat = concat
        self.with_score = with_score

        self.fc_visual = nn.Linear(visual_dim, module_dim, bias=False)
        self.fc_question = nn.Linear(module_dim, module_dim, bias=False)

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(module_dim))
            torch.nn.init.zeros_(self.bias)
        if with_score:
            self.fc_Joint = nn.Linear(module_dim, 1)
        self.fc_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, visual_feature, question_feature):
        if len(visual_feature.size()) != len(question_feature.size()):
            question_feature = question_feature.unsqueeze(dim=1).repeat(1, visual_feature.size(1), 1)
        visual_feat = self.fc_visual(visual_feature)
        question_feat = self.fc_question(question_feature)
        Joint_feat = torch.mul(visual_feat, question_feat)
        if self.use_bias:
            Joint_feat += self.bias
        Joint_feat = self.activation(Joint_feat)
        Joint_feat = self.fc_dropout(Joint_feat)
        if self.with_score:
            score = self.fc_Joint(Joint_feat)
            score = F.softmax(score, dim=-2)
            Joint_feat = torch.mul(visual_feature, score).sum(dim=-2)

        return Joint_feat


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight = nn.Linear(input_dim, output_dim, bias=False)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_feature, adjacency):
        # support = torch.bmm(input_feature, self.weight)
        support = self.weight(input_feature)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class GCN(nn.Module):
    def __init__(self, visual_dim, T, use_bias=True, dropout=0.2):
        super(GCN, self).__init__()
        self.T = T
        self.gcn = nn.Sequential(*[
            GraphConvolution(visual_dim, visual_dim, use_bias=use_bias)
            for i in range(T)
        ])
        self.fc_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, visual_feat, adj):
        for i in range(self.T):
            visual_feat = self.gcn[i](visual_feat, adj)
            if i == 0:
                visual_feat = self.activation(visual_feat)
                visual_feat = self.fc_dropout(visual_feat)
        return visual_feat


class GAT(nn.Module):
    def __init__(self, visual_dim, module_dim, dropout=0.2):
        super(GAT, self).__init__()

        self.fusion_1 = Fusion(visual_dim, module_dim, dropout=dropout)
        self.W3 = nn.Linear(visual_dim, module_dim, bias=False)
        self.fusion_2 = Fusion(visual_dim, module_dim, dropout=dropout)
        self.W6 = nn.Linear(visual_dim + module_dim, visual_dim)
        self.fc_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, visual_now, visual_next, condition, adj):
        Joint_feat = self.fusion_1(visual_now, condition)
        message = torch.bmm(self.W3(visual_next), Joint_feat.transpose(1, 2))
        message = torch.mul(message, adj)
        weight = F.softmax(message, dim=-1)
        Joint_feat = self.fusion_2(visual_now, condition)
        visual_new = torch.bmm(weight, Joint_feat)
        visual_new = self.activation(visual_new)
        visual_new = self.fc_dropout(visual_new)
        visual_new = self.W6(torch.cat([visual_next, visual_new], dim=-1))

        return visual_new


class MSPAN(nn.Module):
    def __init__(self, visual_dim, module_dim, num_frames, num_scale, GCN_adj, GAT_adj, T, K, dropout=0.2):
        super(MSPAN, self).__init__()
        self.num_scale = num_scale
        self.num_frames = num_frames
        self.T = T
        self.K = K

        self.GCN_adj = list(map(lambda x: x.cuda(), GCN_adj))
        self.GAT_adj = list(map(lambda x: x.cuda(), GAT_adj))

        self.GAT_up = nn.Sequential(*[GAT(visual_dim, module_dim, dropout) for i in range(K)])
        self.GAT_down = nn.Sequential(*[GAT(visual_dim, module_dim, dropout) for i in range(K)])
        self.GCN_Module = nn.Sequential(*[GCN(visual_dim, T, True, dropout) for i in range(K)])
        self.attention = Fusion(visual_dim, module_dim, with_score=True)

    def forward(self, visual_feature, question_feature):
        B = question_feature.size(0)
        
        adj_gcn = list(map(lambda x: x.unsqueeze(dim=0).repeat(B, 1, 1), self.GCN_adj))
        adj_up = list(map(lambda x: x.unsqueeze(dim=0).repeat(B, 1, 1), self.GAT_adj))
        adj_down = list(map(lambda x: x.t().unsqueeze(dim=0).repeat(B, 1, 1), self.GAT_adj))

        visual_feat = list(map(lambda x: x.clone(), visual_feature))
        for i in range(self.K):
            for j in range(self.num_scale):
                visual_feat[j] = self.GCN_Module[i](visual_feat[j], adj_gcn[j])
            for j in range(self.num_scale - 1):
                visual_feat[j+1] = self.GAT_down[i](visual_feat[j], visual_feat[j+1], question_feature, adj_down[j])
            for j in range(self.num_scale - 2, -1, -1):
                visual_feat[j] = self.GAT_up[i](visual_feat[j+1], visual_feat[j], question_feature, adj_up[j])
            visual_feat = list(map(lambda x, y: x+y, visual_feat, visual_feature))

        feat = self.attention(visual_feat[0], question_feature)

        return feat