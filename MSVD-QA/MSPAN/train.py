import torch
import numpy as np
import argparse
import os
import pickle
import json
from utils import *

parser = argparse.ArgumentParser(description="MSPAN logger")
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument('-dataset', default='msvd-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
parser.add_argument('-app_feat', default='res152', choices=['resnet', 'res152'], type=str)
parser.add_argument('-mot_feat', default='3dres152', choices=['resnext', '3dres152'], type=str)
parser.add_argument('-ans_num', default=1852, type=int)

# hyper-parameters
parser.add_argument('-num_scale', default=2, type=int)
parser.add_argument('-T', default=2, type=int)
parser.add_argument('-K', default=2, type=int)

parser.add_argument('-num_frames', default=16, type=int)
parser.add_argument('-word_dim', default=768, type=int)
parser.add_argument('-module_dim', default=512, type=int)
parser.add_argument('-app_pool5_dim', default=2048, type=int)
parser.add_argument('-motion_dim', default=2048, type=int)

parser.add_argument('-gpu', default=0, type=int)
parser.add_argument('-epoch', default=45, type=int)
parser.add_argument('-num_workers', default=8, type=int)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-lr', default=0.0001, type=float)
parser.add_argument('-weight_decay', default=0.000005, type=float)
parser.add_argument('-dropout', default=0.3, type=float)

parser.add_argument("-a", type=float, help="alpha", default=1)
parser.add_argument("-a2", type=float, help="alpha", default=1)
parser.add_argument("-tau", type=float, help="temperature for nce loss", default=0.1) 
parser.add_argument("-tau_gumbel", type=float, help="temperature for gumbel_softmax", default=0.9) 
parser.add_argument("-pos", type=int, help="#pos_sample", default=1) 
parser.add_argument("-neg", type=int, help="#neg_sample", default=5) 
parser.add_argument("-b", type=float, action="store", help="kl loss multiplier", default=1) 
args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)

from torch import nn, optim
from torch.utils.data import DataLoader
from DataLoader import VideoQADataset
from networks.network import VideoQANetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from loss import InfoNCE, UberNCE

def train(model, optimizer, train_loader, xe, nce, device):
    model.train()
    total_step = len(train_loader)
    epoch_xe_loss = 0.0
    epoch_nce_loss = 0.0
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    for iter, inputs in enumerate(train_loader):
        videos, qas, qas_lengths, answers, vid_idx = inputs
        video_inputs, qas_inputs, qas_lengths, ans_targets = videos.to(device), qas.to(device), qas_lengths.to(device), answers.to(device)

        # #mix-up 
        lam_1 = np.random.beta(args.a, args.a)
        lam_2 = np.random.beta(args.a2, args.a2)
        index = torch.randperm(video_inputs.size(0))
        targets_a, targets_b = ans_targets, ans_targets[index]
        out, out_anc, out_pos, out_neg = model(video_inputs, qas_inputs, qas_lengths, vid_idx, lam_1, lam_2, index)
        model.zero_grad()

        # xe loss
        xe_loss = lam_1 * xe(out, targets_a) + (1 - lam_1) * xe(out, targets_b) #xe(out, ans_targets)
        # cl loss
        nce_loss = nce(out_anc, out_pos, out_neg)
        
        loss = xe_loss + args.b * nce_loss
        loss.backward()
        optimizer.step()
        epoch_xe_loss += xe_loss.item()
        epoch_nce_loss += args.b*nce_loss.item()
        epoch_loss += loss.item()
        prediction=out.max(-1)[1] # bs,
        prediction_list.append(prediction)
        answer_list.append(answers)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return epoch_loss / total_step, epoch_xe_loss/ total_step, epoch_nce_loss/ total_step, acc_num*100.0 / len(ref_answers)


def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            videos, qas, qas_lengths, answers, vid_idx = inputs
            video_inputs, qas_inputs, qas_lengths = videos.to(device), qas.to(device), qas_lengths.to(device)

            out = model(video_inputs, qas_inputs, qas_lengths, vid_idx )
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(answers)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return acc_num*100.0 / len(ref_answers)


if __name__=="__main__":

    logger, sign =logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set data path
    
    sample_list_path = '/storage_fast/ycli/vqa/qa_dataset'
    feat_path= '/storage_fast/ycli/vqa/qa_feat'
    qst_feat_path = '/storage_fast/ycli/vqa/qa_feat'

    train_data = VideoQADataset(sample_list_path, feat_path, 'train', args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_data = VideoQADataset(sample_list_path, feat_path, 'val', args)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = VideoQADataset(sample_list_path, feat_path, 'test', args)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
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
        'mot': args.mot_feat,
        'pos': args.pos,
        'neg': args.neg,
        'tau_gumbel': args.tau_gumbel
    }

    model = VideoQANetwork(**model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    model.to(device)
    xe = nn.CrossEntropyLoss().to(device)
    cl =  InfoNCE(negative_mode='paired')
    
    # train & val
    print('training...')
    best_eval_score = 0.0
    best_epoch=1
    for epoch in range(1, args.epoch+1):
        train_loss, train_xe_loss, train_nce_loss, train_acc = train(model, optimizer, train_loader, xe, cl, device)
        eval_score = eval(model, val_loader, device)
        logger.debug("==>Epoch:[{}/{}][lr: {}][Train Loss: {:.4f} XE: {:.4f} NCE: {:.4f} Train acc: {:.2f} Val acc: {:.2f}]".
                format(epoch, args.epoch, optimizer.param_groups[0]['lr'], train_loss, train_xe_loss, train_nce_loss, train_acc, eval_score))
        scheduler.step(eval_score)
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            best_epoch = epoch 
            best_model_path='./models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    test_acc=eval(model, test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

    # result_path= './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
    # save_file(results, result_path)