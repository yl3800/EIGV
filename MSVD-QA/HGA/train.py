from utils.util import EarlyStopping, save_file, set_gpu_devices, pause
import os
from utils.logger import logger
import time
import logging
import argparse
import os.path as osp
import numpy as np
from utils.loss import InfoNCE, UberNCE

parser = argparse.ArgumentParser(description="GCN train parameter")
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument('-dataset', default='msvd-qa',choices=['msrvtt-qa', 'msvd-qa'], type=str)
parser.add_argument('-app_feat', default='res152', choices=['resnet', 'res152'], type=str)
parser.add_argument('-mot_feat', default='3dres152', choices=['resnext', '3dres152'], type=str)
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=256)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=100)
parser.add_argument("-nfs", action="store_true", help="use local ssd")
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
parser.add_argument("-ans_num", type=int, help="ans vocab num", default=1852)  
parser.add_argument("-es", action="store_true", help="early_stopping")
parser.add_argument("-b", type=float, action="store", help="kl loss multiplier", default=1)
parser.add_argument("-hd", type=int, help="hidden dim of vq encoder", default=512) 
parser.add_argument("-wd", type=int, help="word dim of q encoder", default=512)   
parser.add_argument("-drop", type=float, help="dropout rate", default=0.5) 
parser.add_argument("-neg", type=int, help="#neg_sample", default=5)  
parser.add_argument("-pos", type=int, help="#pos_sample", default=1)  
parser.add_argument("-a", type=float, help="alpha", default=1)
parser.add_argument("-a2", type=float, help="alpha", default=1)
parser.add_argument("-lam_lb", type=float, help="lam lower bound", default=0.75) 
parser.add_argument("-tau", type=float, help="temperature for nce loss", default=0.1) 
# parser.add_argument("-es", type=bool, action="store", help="early_stopping", default=True)
args = parser.parse_args()
set_gpu_devices(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from networks.embed_loss import MultipleChoiceLoss
from networks.hga import HGA
from dataloader.dataset import VideoQADataset 
# from torch.utils.tensorboard import SummaryWriter

seed = 999

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))



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
        video_inputs = videos.to(device)
        qas_inputs = qas.to(device)
        qas_lengths = qas_lengths.to(device)
        ans_targets = answers.to(device)

        lam_1 = np.random.beta(args.a, args.a)
        lam_2 = np.random.beta(args.a2, args.a2)
        index = torch.randperm(video_inputs.size(0))
        # video_inputs = lam*video_inputs + (1-lam)*video_inputs[index,:]
        # qas_inputs = lam*qas_inputs + (1-lam)*qas_inputs[index,:]
        targets_a, targets_b = ans_targets, ans_targets[index]

        out, out_anc, out_pos, out_neg = model(video_inputs, qas_inputs, qas_lengths, vid_idx, lam_1, lam_2, index) # q: [bs,d]  pos: [bs,d] neg: [bs, num_ned, d]
        model.zero_grad()

        # xe loss
        xe_loss = lam_1 * xe(out, targets_a) + (1 - lam_1) * xe(out, targets_b)#xe(out, ans_targets)
        # cl loss
        nce_loss = nce(out_anc, out_pos, out_neg)

        loss = xe_loss + args.b*nce_loss
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
            video_inputs = videos.to(device)
            qas_inputs = qas.to(device)
            qas_lengths = qas_lengths.to(device)
            out = model(video_inputs, qas_inputs, qas_lengths, vid_idx )
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(answers)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return acc_num*100.0 / len(ref_answers)



if __name__ == "__main__":

    # writer = SummaryWriter('./log/tensorboard')
    logger, sign =logger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data set&Loader
    sample_list_path = '/storage_fast/ycli/vqa/qa_dataset'
    feat_path= '/storage_fast/ycli/vqa/qa_feat'
    qst_feat_path = '/storage_fast/ycli/vqa/qa_feat'
    
    # Data set&Loader
    train_data = VideoQADataset(sample_list_path, feat_path, 'train', args)
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

    val_data = VideoQADataset(sample_list_path, feat_path, 'val', args)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    test_data = VideoQADataset(sample_list_path, feat_path, 'test', args)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    # hyper setting
    lr_rate = args.lr
    epoch_num = args.epoch
    model = HGA(args.pos, args.neg, args.ans_num, args.mot_feat)
    optimizer = torch.optim.Adam(params = [{'params':model.parameters()}], lr=lr_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    model.to(device)
    xe = nn.CrossEntropyLoss().to(device)
    # cl = InfoNCE(negative_mode='paired')
    cl =  UberNCE(args.tau)


    if args.es:
        early_stopping = EarlyStopping(patience=7, min_delta=0)

    # train & val
    best_eval_score = 0.0
    best_epoch=1
    for epoch in range(1, epoch_num+1):
        train_loss, train_xe_loss, train_nce_loss, train_acc = train(model, optimizer, train_loader, xe, cl, device)
        eval_score = eval(model, val_loader, device)
        logger.debug("==>Epoch:[{}/{}][lr: {}][Train Loss: {:.4f} XE: {:.4f} NCE: {:.4f} Train acc: {:.2f} Val acc: {:.2f}]".
                format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss, train_xe_loss, train_nce_loss, train_acc, eval_score))
        scheduler.step(eval_score)
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            best_epoch = epoch 
            best_model_path='./models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)

        # check if need to early stop
        if args.es:
            early_stopping(eval_score)
            if early_stopping.early_stop:
                break

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    test_acc=eval(model,test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

