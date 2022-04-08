import torch
import numpy as np
import argparse
import os
import pickle
import json
from utils import *

parser = argparse.ArgumentParser(description="MSPAN logger")
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument('-ans_num', default=5, type=int)

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
parser.add_argument('-epoch', default=35, type=int)
parser.add_argument('-num_workers', default=8, type=int)
parser.add_argument('-bs', default=128, type=int)
parser.add_argument('-lr', default=5e-5, type=float)
parser.add_argument('-wd', default=0, type=float)
parser.add_argument('-drop', default=0.3, type=float)

parser.add_argument("-a", type=float, help="alpha", default=1)
parser.add_argument("-a2", type=float, help="alpha", default=1)
parser.add_argument("-neg", type=int, help="#neg_sample", default=5) 
parser.add_argument("-b", type=float, action="store", help="kl loss multiplier", default=0.0125) 
parser.add_argument("-tau", type=float, help="temperature for nce loss", default=0.1) 
parser.add_argument("-tau_gumbel", type=float, help="temperature for gumbel_softmax", default=0.9) 
args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoader import VideoQADataset
from networks.network import VideoQANetwork
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import eval_mc
from loss import InfoNCE

def train(model, optimizer, train_loader, xe, nce, device):
    model.train()
    total_step = len(train_loader)
    epoch_xe_loss = 0.0
    epoch_nce_loss = 0.0
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    for iter, inputs in enumerate(train_loader):
        # videos, qas, qa_lengths, vid_idx, ans, qns_key = inputs
        input_batch = list(map(lambda x: x.to(device), inputs[:-1]))
        videos, qas, qas_lengths, vid_idx, ans = input_batch

        # #mix-up 
        lam_1 = np.random.beta(args.a, args.a)
        lam_2 = np.random.beta(args.a2, args.a2)
        index = torch.randperm(videos.size(0))
        targets_a, targets_b = ans, ans[index]

        out, out_anc, out_pos, out_neg = model(videos, qas, qas_lengths, vid_idx, lam_1, lam_2, index, ans)
        model.zero_grad()
        # xe loss
        # xe_loss = lam_1 * xe(out[:,:5], targets_a) + (1 - lam_1) * xe(out[:,5:], targets_b)  # xe(out, ans) 
        targets_a = F.one_hot(targets_a, num_classes=5)
        targets_b = F.one_hot(targets_b, num_classes=5)
        target = torch.cat([lam_1*targets_a, (1-lam_1)*targets_b], -1)
        xe_loss = xe(F.log_softmax(out, dim=1), target)
        
        # cl loss
        nce_loss = nce(out_anc, out_pos, out_neg)
        
        loss = xe_loss + args.b * nce_loss
        loss.backward()
        optimizer.step()
        epoch_xe_loss += xe_loss.item()
        epoch_nce_loss += args.b*nce_loss.item()
        epoch_loss += loss.item()
        prediction = out[:,:5].max(-1)[1] # bs,
        prediction_list.append(prediction)
        answer_list.append(inputs[-2])

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
            input_batch = list(map(lambda x: x.to(device), inputs[:3]))
            out = model(*input_batch)
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(inputs[-2])

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return acc_num*100.0 / len(ref_answers)
    
def predict(model,test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    with torch.no_grad():
        for iter, inputs in enumerate(test_loader):
            input_batch = list(map(lambda x: x.to(device), inputs[:3]))
            answers, qns_keys = inputs[-2], inputs[-1]

            out = model(*input_batch)
            prediction=out.max(-1)[1] # bs,
            prediction = prediction.data.cpu().numpy()
            
            for qid, pred, ans in zip(qns_keys, prediction, answers.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    return results


if __name__=="__main__":

    logger, sign =logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set data path
    feat_path = '/storage_fast/ycli/vqa/qa_feat/next-qa'
    sample_list_path = '/storage_fast/ycli/vqa/qa_dataset/next-qa'

    train_data = VideoQADataset(sample_list_path, feat_path, 'train')
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_data = VideoQADataset(sample_list_path, feat_path, 'val')
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    test_data = VideoQADataset(sample_list_path, feat_path, 'test')
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    GCN_adj, GAT_adj = make_adjacency(args)
    model_kwargs = {
        'app_pool5_dim': args.app_pool5_dim,
        'motion_dim': args.motion_dim,
        'num_frames': args.num_frames,
        'word_dim': args.word_dim,
        'module_dim': args.module_dim,
        'num_answers': args.ans_num,
        'dropout': args.drop,
        'GCN_adj': GCN_adj,
        'GAT_adj': GAT_adj,
        "K": args.K,
        "T": args.T,
        'num_scale': args.num_scale,
        'neg': args.neg,
        'tau_gumbel': args.tau_gumbel
    }

    model = VideoQANetwork(**model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = MultiStepLR(optimizer, milestones=[10,15,20,25], gamma=0.5)
    model.to(device)
    xe = nn.KLDivLoss(reduction='batchmean').to(device) # nn.CrossEntropyLoss().to(device)
    cl =  InfoNCE(temperature=args.tau, negative_mode='paired')
    
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

    results=predict(model, test_loader, device)
    eval_mc.accuracy_metric(test_data.sample_list_file, results)
    result_path= './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
    save_file(results, result_path)