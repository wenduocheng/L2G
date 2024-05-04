import argparse
import random

import torch.backends.cudnn as cudnn
from timeit import default_timer
from attrdict import AttrDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import re
from tqdm import tqdm
import sys
from ml_collections import config_dict
# from easydict import EasyDict as edict
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import matplotlib
# import pandas as pd
# from transformers import AutoModel, AutoConfig, RobertaForTokenClassification

from utils import count_params, count_trainable_params, get_params_to_update, \
                            set_grad_state, set_param_grad
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, get_scheduler
from embedder import get_tgt_model, wrapper1D
import json
import time

from functools import partial
from torch.utils.data import random_split
import torchvision.transforms as transforms

from ray import train, tune, init
from ray.tune.schedulers import ASHAScheduler

sys.path.append("~/ORCA/clean/gene-orca")
    
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"




def get_params_to_update(model):

    params_to_update = []
    name_list = ''
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            name_list += "\t" + name
    print("Params to learn:", name_list)
    return params_to_update


# def get_optimizer(config):
#     if config['optimizer'] == 'SGD':
#         return partial(torch.optim.SGD, lr=config['lr'], momentum=0.99, weight_decay=config['weight_decay'])
#     elif config['optimizer'] == 'Adam':
#         return partial(torch.optim.Adam, lr=config['lr'], betas=[0.9, 0.98], weight_decay=config['weight_decay'])
#     elif config['optimizer'] == 'AdamW':
#         return partial(torch.optim.AdamW, lr=config['lr'], betas=[0.9, 0.98], weight_decay=config['weight_decay'])

def train_one_epoch(args, config, model, optimizer, scheduler, loader, loss, temp, label_smoothing_factor=None, decoder=None, transform=None):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()
    
    model_time, data_time = 0,0
    data_time_start = default_timer()

    for i, data in enumerate(loader):

        x, y = data 
            
        x, y = x.to(args.device), y.to(args.device) # accelerate

        data_time += default_timer() - data_time_start
        model_start = default_timer()

        out = model(x)

        # right += (y==out.argmax(-1)).float().sum()  # if using accuracy, count how many out is correct (=y)
        # alldata += len(x)

        model_time += default_timer() - model_start
      
        # print('out:',out.size(),'y:', y.size())
        l = loss(out, y)
        
        l.backward()


        if config['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])

        if (i + 1) % config['accum'] == 0: # to save memory to approximate performance by large batchsize
            optimizer.step()
            optimizer.zero_grad()
            
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break

        data_time_start = default_timer()

    if (not args.lr_sched_iter):
        scheduler.step()
    # print(right/alldata)
    return train_loss / temp, model_time, data_time


def evaluate(args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    right, alldata = 0,0


    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(args.device), y.to(args.device) # accelerate

            out = model(x)

            # right+=(out.argmax(-1)==y).float().sum() #
            # alldata += len(x) #
    
            outs.append(out) 
            ys.append(y) 
            n_data += x.shape[0]
        
            if n_data >= args.eval_batch_size or i == len(loader) - 1:
                outs = torch.cat(outs, 0)
                ys = torch.cat(ys, 0)

                eval_loss += loss(outs, ys).item()
                # print('309',outs.shape)
                # print('309',ys.shape)
                # print('309',outs)
                # print('309',ys)
                # print(metric(outs, ys))
                eval_score += metric(outs, ys).item()
                
                n_eval += 1

                ys, outs, n_data = [], [], 0

        eval_loss /= n_eval
        eval_score /= n_eval


   
    # eval_score = 1-(right/alldata).detach().cpu().numpy() # if using accuracy
    return eval_loss, eval_score

   
def main(config):
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0) 
    torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # eval('setattr(torch.backends.cudnn, "deterministic", True)')
    # eval('setattr(torch.backends.cudnn, "benchmark", False)')
            
    args = config_dict.ConfigDict()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.dataset = dataset_name 
    args.embedder_dataset = 'text'


    args.weight = 'roberta'
    args.eval_batch_size = 1000
    args.maxsamples=256
    args.target_seq_len=config['target_seq_len']
    args.drop_out = config['drop_out']
    # args.accum = 1
    args.validation_freq = 1

    args.batch_size = config['batch_size']
    
    args.joint_optim = True
    args.finetune_method = 'all'
    args.one_hot = config['one_hot']
    args.activation = None
    args.objective = 'MMD'
    # args.lora = config_dict.ConfigDict()
    # args.lora.target_modules = 'query value key dense reduction'
    # args.lora.layer_indices = False
    # args.lora.layers_pattern = False
    # args.lora.bias = 'none'
    # args.lora.rank = 8
    # args.lora.alpha = 16
    
    args.num_workers = 4
    args.valid_split = False
    args.epochs = 30

    args.scheduler = config_dict.ConfigDict()
    args.scheduler.name = 'WarmupLR'
    args.scheduler.params = config_dict.ConfigDict()
    args.scheduler.params.warmup_epochs = 5
    args.scheduler.params.decay_epochs = 30
    args.scheduler.params.sched = [30, 60, 90]
    args.scheduler.params.base = 0.2

    args.no_warmup_scheduler = config_dict.ConfigDict()
    args.no_warmup_scheduler.name = 'StepLR'
    args.no_warmup_scheduler.params = config_dict.ConfigDict()
    args.no_warmup_scheduler.params.warmup_epochs = 10
    args.no_warmup_scheduler.params.decay_epochs = 100
    args.no_warmup_scheduler.params.sched = [40, 60, 80]
    args.no_warmup_scheduler.params.base = 0.2

    args.optimizer = config_dict.ConfigDict()
    args.optimizer.name = config['optimizer']
    args.optimizer.params = config_dict.ConfigDict()
    args.optimizer.params.lr = config['lr']
    args.optimizer.params.betas = [0.9, 0.98] 
    args.optimizer.params.weight_decay = config['weight_decay']
    args.optimizer.params.momentum = 0.99
    
    args.quantize = False
    args.embedder_type = "unet" # dash-->resnet, unet, dash random
    args.embedder_init = "random"

    root = '/home/wenduoc/ORCA/clean/gene-orca/datasets'
    
    print('torch.cuda.is_available():',torch.cuda.is_available())
    print('device:', args.device)


    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    # print(dims, sample_shape, num_classes, loss)

    args.embedder_epochs = config['embedder_epochs']

    args.alpha = config['alpha']

    # wrapper_func = wrapper1D 
    # model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=config['drop_out'], from_scratch=False, args=args)
    model, _ = get_tgt_model(args, root, sample_shape, num_classes, loss, False, False, None)
    model.output_raw = False
    
    

    # train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, config['batch_size'], args.valid_split)
    # decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    # transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    # metric, compare_metrics = get_metric(root, args.dataset)


    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split, quantize=False, rc_aug=True, shift_aug=True, one_hot=args.one_hot)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = None 
    transform = None
    
    train_full = True

    # set whole model to be trainable
    set_grad_state(model, True)  
    # set_param_grad(args, model, args.finetune_method)
    set_param_grad( model, args.finetune_method)

    get_optimizer_scheduler

    # args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=n_train)
    model = model.to(args.device)
    print(model)
    # optimizer = get_optimizer(config)(get_params_to_update(model))
    # lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # print("\n------- Experiment Summary --------")
    # print("id:", args.experiment_id)
    # print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer.params.lr)
    # print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    # print("finetune method:", args.finetune_method)
    # print('train_full:', train_full)
    # print("param count:", count_params(model), count_trainable_params(model))
    
    train_losses, train_score = [], []
    for ep in range(args.epochs):

        train_loss, model_time, data_time = train_one_epoch(args, config, model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
        
        if ep % args.validation_freq == 0 or ep == args.epochs-1: 
            val_loss, val_score = evaluate(args, model, val_loader, loss, metric, n_val, decoder, transform, 
                                           fsd_epoch=ep if args.dataset == 'FSD' else None)
            
            train_losses.append(train_loss)
            train_score.append(val_score)

            # train.report({'val_score': val_score})
            train.report(
                {"val_score": val_score, "val_loss": val_loss}
            )

            print("[train full", ep, "]",                    
                    "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, 
                    "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))



search_space = {
    "weight_decay": tune.choice([1e-5]),
    'batch_size': tune.choice([128]),
    'accum': tune.choice([1]),
    'clip': tune.choice([1]),
    'alpha': tune.choice([0.01,0.1,1]),
    'target_seq_len': tune.choice([64, 128, 512]),
    "lr": tune.choice([5e-4, 5e-5, 5e-6]),
    'drop_out': tune.choice([0,0.1,0.2]),
    'optimizer': tune.choice(['Adam', 'AdamW']),
    'embedder_epochs': tune.choice([40,60,80,100]),
    'one_hot': tune.choice([False])
}

# Uncomment this to enable distributed execution
# init(address="auto")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASHA')
    parser.add_argument('--dataset', type=str, default='H4', help='dataset name')

    args = parser.parse_args()
    print('Dataset:',args.dataset)
    dataset_name = args.dataset

    main_with_gpu = tune.with_resources(main, {"cpu": 12, "gpu": 1})
    tuner = tune.Tuner(
        main_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=ASHAScheduler(metric="val_loss", mode="min"), # val_score
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
            best_result.metrics["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["val_score"]))

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.path: result.metrics_dataframe for result in results}

    print(results)