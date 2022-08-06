import re
import os
import sys
import time
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
import os.path as osp

def load_file(file_name):
    annos = None
    if osp.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos


def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != '' and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)


def normalize(adj):
    degree = adj.sum(1)
    d_hat = np.diag(np.power(degree, -0.5).flatten())
    return d_hat.dot(adj).dot(d_hat)


def make_adjacency(args):
    GCN_adj = []
    for i in range(args.num_scale):
        N = args.num_frames - i  # N is node number at scale i, i is kernel size
        sub_adj = np.zeros(shape=(N, N), dtype=np.float32)
        for j in range(N):
            for k in range(N):
                if k-1 > j+i or j-1 > k+i: # node k,j has no overlap
                    continue
                else:
                    sub_adj[j][k] = 1 # node k,j has no overlap
        sub_adj = normalize(sub_adj)
        sub_adj = torch.from_numpy(sub_adj)
        GCN_adj.append(sub_adj)

    GAT_adj = []
    for i in range(args.num_scale - 1): # x graph, compose x-1 cross-graph adj, i is kernel size
        N = args.num_frames - i         # i graph contrain num_frames-i node
        M = args.num_frames - i - 1     # i+1 graph contrain num_frames-i-1 node
        sub_adj = np.zeros(shape=(N, M), dtype=np.float32)
        for j in range(N):
            for k in range(M):
                if j > k+i or j+i < k:
                    continue
                else:
                    sub_adj[j][k] = 1
        sub_adj = torch.from_numpy(sub_adj)
        GAT_adj.append(sub_adj)

    return GCN_adj, GAT_adj


def save_model(args, model, save_path, score_all, score_word=None):
    save_kwargs ={
        "args": args,
        "score_all": score_all,
        "score_word": score_word,
        "model": model.state_dict()
    }
    torch.save(save_kwargs, save_path)


def load_model(load_path):
    kwargs = torch.load(load_path)
    args = kwargs["args"]
    model_dict = kwargs["model"]
    args.train = False
    args.val = False
    args.test = True
    if args.dataset in ['msvd-qa', 'msrvtt-qa']:
        print("The best result of question words are {} !".format(kwargs["score_word"]))
    print("The best result is \033[1;31m {:.4f} \033[0m !".format(kwargs["score_all"]))
    return args, model_dict


def pause():
    programPause=input('press to continue...')


# split string using multi delimiters
def multisplit(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gpu_devices(gpu_id):
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def logger(args):
    args_str = "{}".format(args)
    args_str = multisplit(['(', ')'], args_str)
    args_str = args_str[1].replace(', ', '_')

    logger = logging.getLogger('VQA')  # logging name
    logger.setLevel(logging.DEBUG)  # 接收DEBUG即以上的log info
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    now = time.localtime(time.time())
    now_str = "{0}.{1}_{2}.{3}.{4}".format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    record_dir = "./log/{}_{}".format(now.tm_mon, now.tm_mday)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    record_file = os.path.join(record_dir,"{}_at_{}.log".format(args.v,now_str))

    fh = logging.FileHandler(record_file)  # log info 输入到文件
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)  # log info 输入到屏幕
    sh.setLevel(logging.DEBUG)

    fmt = '[%(asctime)-15s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)  # 设置每条info开头格式
    logger.addHandler(fh)  # 把FileHandler/StreamHandler加入logger
    logger.addHandler(sh)
    logger.debug(args_str)

    return logger, "{}_at_{}".format(args.v,now_str)