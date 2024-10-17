import os
import json
import argparse
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from timeit import default_timer

from relax.nas import MixedOptimizer
from dash import MixtureSupernet
from task_configs import get_data, get_config, get_model, get_metric, get_hp_configs, get_optimizer
from task_utils import count_params, print_grad, calculate_stats

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from functools import partial


def main(asha_config, args):
    print("----------")
    print("ASHA config:", asha_config)
    print("----------")

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    if args.reproducibility:
        eval('setattr(torch.backends.cudnn, "deterministic", True)')
        eval('setattr(torch.backends.cudnn, "benchmark", False)')
    else:
        eval('setattr(torch.backends.cudnn, "benchmark", True)')

    args.arch = asha_config['arch']
    args.grad_scale = asha_config['grad_scale']
    args.pool_k = None

    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq, \
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(args.dataset, args)  
    
    # arch_default = asha_config['arch']
    # # lr = asha_config['lr']
    # # arch_lr = asha_config['arch_lr']
    # # weight_decay = asha_config['weight_decay']
    # config_kwargs['grad_scale'] = asha_config['grad_scale']


    
    arch = arch_default

    kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
    dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.dataset, batch_size, arch, args.valid_split)
    model = get_model(arch, sample_shape, num_classes, config_kwargs)
    metric, compare_metrics = get_metric(args.dataset)

    model = MixtureSupernet.create(model.cpu(), in_place=True)

    
    model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable, 
        stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)

    if dims == 1:
        model.remove_module("chomp")
    opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay), arch_opt(model.arch_params(), lr=arch_lr, weight_decay=weight_decay)]

    
    optimizer = MixedOptimizer(opts)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
    lr_sched_iter = False
    
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    n_train_temp = int(quick_search * n_train) + 1 if quick_search < 1 else n_train

    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    # print("search arch:", arch)
    # print("batch size:", batch_size, "\tlr:", lr, "\tarch lr:", arch_lr)
    # print("arch configs:", config_kwargs)
    # print("kernel choices:", kernel_choices_default, "\tdilation choices:", dilation_choices_default)
    # print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)

    # print("\n------- Start Arch Search --------")
    # print("param count:", count_params(model))
    for ep in range(epochs):

        train_loss = train_one_epoch(model, optimizer, scheduler, args.device, train_loader, loss, clip, 1, n_train_temp, decoder, transform, lr_sched_iter, scale_grad=not args.baseline)
            
        val_loss, val_score = evaluate(model, args.device, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

        train.report({'val_score': val_score})


        # if not args.baseline and ((ep + 1) % retrain_freq == 0 or ep == epochs - 1):

        #     param_values, ks, ds = [], [], []
        #     for name, param in model.named_arch_params():
        #         param_values.append(param.data.argmax(0))
        #         if args.verbose:
        #             print(name, param.data)
                
        #         ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        #         ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

        #     param_values = torch.stack(param_values, dim = 0)
        #     print("[searched kernel pattern] ks:", ks, "\tds:", ds)

def train_one_epoch(model, optimizer, scheduler, device, loader, loss, clip, accum, temp, decoder=None, transform=None, lr_sched_iter=False, min_lr=-1.0, scale_grad=False):
        
    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):
        if transform is not None:
            x, y, z = data
            z = z.to(device)
        else:
            x, y = data 
        
        x, y = x.to(device), y.to(device)
            
        out = model(x)

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)
                        
        l = loss(out, y)
        l.backward()

        if scale_grad:
            model.scale_grad()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += l.item()

        if lr_sched_iter and optimizer.param_groups[0]['lr'] > min_lr:
            scheduler.step()
            # print("scheduler step")

        if i >= temp - 1:
            break

    if (not lr_sched_iter) and optimizer.param_groups[0]['lr'] > min_lr:
        scheduler.step()
        # print("scheduler step")

    return train_loss / temp


def evaluate(model, device, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_batch_size=1000
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0
    
    if fsd_epoch is None:
        with torch.no_grad():
            for i, data in enumerate(loader):
                if transform is not None:
                    x, y, z = data
                    z = z.to(device)
                else:
                    x, y = data
                                    
                x, y = x.to(device), y.to(device)
                out = model(x)
                
                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)
                
                outs.append(out) 
                ys.append(y) 
                n_data += x.shape[0]
            
                if n_data >= eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys).item()
                    
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

        eval_loss /= n_eval
        eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                if fsd_epoch < 100:
                    if ix > 2000: break

                x, y = loader[ix]
                x, y = x.to(device), y.to(device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score

def asha_main(args):
    search_program = partial(main, args=args)

    # search_program({"grad_scale": 100, "arch": "wrn-22-1"})

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,3,4,5,6,7"

    search_space = {
        # "lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "arch_lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "weight_decay": tune.choice([1e-3, 1e-4, 1e-5]),
        "grad_scale": tune.choice([100, 500, 1000]),
        "arch": tune.choice(["wrn-22-1", "unet_small"]),
    }

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    main_with_gpu = tune.with_resources(search_program, {"cpu": 3, "gpu": 1})
    
    tuner = tune.Tuner(
        main_with_gpu,
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=ASHAScheduler(metric="val_score", mode="max"),
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_score", "max")

    print("Best trial config: {}".format(best_result.config))

    return best_result.config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='DEEPSEA', help='dataset name')
    parser.add_argument('--arch', type=str, default='', help='backbone architecture')
    parser.add_argument('--baseline', type=int, default=0, help='evaluate backbone without architecture search')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kernel_choices', type=int, default=[None], nargs='+', help='specify the set of kernel sizes (K)' )
    parser.add_argument('--dilation_choices', type=int, default=[None], nargs='+', help='specify the set of dilation rates (D)')
    parser.add_argument('--verbose', type=int, default=0, help='print gradients of arch params for debugging')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--valid_split', type=int, default=0, help='use train-validation-test (3-way) split')
    parser.add_argument('--reproducibility', type=int, default=0, help='exact reproducibility')
    parser.add_argument('--separable', type=int, default=1, help='use separable conv')
    parser.add_argument('--stream', type=int, default=1, help='use streaming for implementing aggconv')


    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asha_main(args)
    

