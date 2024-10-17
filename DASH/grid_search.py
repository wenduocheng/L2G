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
from functools import partial
import copy

from relax.nas import MixedOptimizer
from dash import MixtureSupernet
from task_configs import get_data, get_config, get_model, get_metric, get_hp_configs, get_optimizer
from task_utils import count_params, print_grad, calculate_stats

from main import train_one_epoch, evaluate
from itertools import product

from ray import train, tune

def main():

    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='DEEPSEA', help='dataset name')
    parser.add_argument('--root_dir', type=str, default='./data/', help='root directory for the dataset')
    parser.add_argument('--save_dir', type=str, default='', help='save directory for the dataset')
    parser.add_argument('--arch', type=str, default='', help='backbone architecture')
    parser.add_argument('--baseline', type=int, default=0, help='evaluate backbone without architecture search')
    parser.add_argument('--experiment_id', type=str, default='0', help='directory name to save the experiment results')
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
    exp_id = 'baseline' if args.baseline else args.experiment_id
    args.save_dir = os.path.join(args.save_dir, 'results_acc/'  + args.dataset + '/' + 'grid_search' +'/' + exp_id + "/" + str(args.seed))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("------- Experiment Summary --------")
    print(args.__dict__)
    _, compare_metrics = get_metric(args.dataset)

    search_hp_configs = get_search_hp_configs()
    grid_scores, retrain_hp_configs, searched_ks, searched_ds = [], [], [], []

    for arch, grad_scale in search_hp_configs:
        ks, ds, val_score, retrain_hp = run_expt(args, arch, grad_scale)
        grid_scores.append(val_score)
        retrain_hp_configs.append(retrain_hp)
        searched_ks.append(ks)
        searched_ds.append(ds)
    
    idx = np.argwhere(grid_scores == compare_metrics(grid_scores))[0][0]
    selected_arch, selected_grad_scale = search_hp_configs[idx]
    lr, drop_rate, weight_decay, momentum = retrain_hp_configs[idx]
    ks, ds = searched_ks[idx], searched_ds[idx]
    
    print("\n------- Completed Grid Search -------")
    print("[selected search hp] arch = ", selected_arch, " grad_scale = ", selected_grad_scale)
    print("[searched kernel pattern] ks:", ks, "\tds:", ds)
    print("[selected retrain hp] lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
    print("[selected score] val_score = ", "%.4f" % grid_scores[idx])
    print(f"\n------- Results saved at {args.save_dir} -------")
    np.save(os.path.join(args.save_dir, 'search_hps.npy'), search_hp_configs[idx])
    np.save(os.path.join(args.save_dir, 'network_hps.npy'), retrain_hp_configs[idx])
    np.save(os.path.join(args.save_dir, 'searched_kernel_pattern.npy'), (searched_ks[idx], searched_ds[idx]))

def main_ray():

    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='DEEPSEA', help='dataset name')
    parser.add_argument('--root_dir', type=str, default='./data/', help='root directory for the dataset')
    parser.add_argument('--save_dir', type=str, default='', help='save directory for the dataset')
    parser.add_argument('--arch', type=str, default='', help='backbone architecture')
    parser.add_argument('--baseline', type=int, default=0, help='evaluate backbone without architecture search')
    parser.add_argument('--experiment_id', type=str, default='0', help='directory name to save the experiment results')
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
    exp_id = 'baseline' if args.baseline else args.experiment_id
    args.save_dir = os.path.join(args.save_dir, 'results_acc/'  + args.dataset + '/' + 'grid_search' +'/' + exp_id + "/" + str(args.seed))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("------- Experiment Summary --------")
    print(args.__dict__)
    _, compare_metrics = get_metric(args.dataset)

    # archs = ['unet2d_small', 'wrn-22-1']
    archs = ['unet', 'wrn']
    grad_scales = [100, 500, 1000]

    best_result = grid_search_main(args, compare_metrics, archs, grad_scales)

    selected_arch, selected_grad_scale = best_result.config['arch'], best_result.config['grad_scale']
    lr, drop_rate, weight_decay, momentum = best_result.metrics['lr'], best_result.metrics['drop_rate'], best_result.metrics['weight_decay'], best_result.metrics['momentum']
    ks, ds = best_result.metrics['ks'], best_result.metrics['ds']

    # Delete all other init files except the selected one
    for arch, grad_scale in list(product(archs, grad_scales)):
        model_path = os.path.join(args.save_dir, f'init_{arch}_{grad_scale}.pt')
        if arch == selected_arch and grad_scale == selected_grad_scale:
            os.rename(model_path, os.path.join(args.save_dir, 'init.pt'))
        else:
            if os.path.exists(model_path):
                os.remove(model_path)
            else:
                print(f"File {model_path} does not exist")

    print("\n------- Completed Grid Search -------")
    print("[selected search hp] arch = ", selected_arch, " grad_scale = ", selected_grad_scale)
    print("[searched kernel pattern] ks:", ks, "\tds:", ds)
    print("[selected retrain hp] lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
    print("[selected score] val_score = ", "%.4f" % best_result.metrics['val_score'])
    print(f"\n------- Results saved at {args.save_dir} -------")
    np.save(os.path.join(args.save_dir, 'search_hps.npy'), (selected_arch, selected_grad_scale))
    np.save(os.path.join(args.save_dir, 'network_hps.npy'), (lr, drop_rate, weight_decay, momentum))
    np.save(os.path.join(args.save_dir, 'searched_kernel_pattern.npy'), (ks, ds))

    
def get_search_hp_configs():
    grad_scale = [100, 500, 1000]
    archs = ['unet2d_small', 'wrn-22-1']
    configs = list(product(archs, grad_scale))
    return configs

def run_expt(args, arch, grad_scale):
    args.arch = arch
    args.grad_scale = grad_scale
    args.pool_k = None

    print(f"\n------- Running configuration: arch: {args.arch}, grad_scale: {args.grad_scale} -------")

    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, weight_sched_hpo, accum, clip, retrain_clip, validation_freq, retrain_freq, \
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(args.dataset, args)  
    
    if config_kwargs['arch_retrain_default'] is not None:
        arch_retrain = config_kwargs['arch_retrain_default']
    else:
        arch_retrain = arch

    if arch_retrain == "unet2d" or arch_retrain == "unet2d_small":
        optimizer_type = "AdamW"
    else:
        optimizer_type = "SGD"

    kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
    dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.root_dir, args.dataset, batch_size, arch, args.valid_split)
    model = get_model(arch, sample_shape, num_classes, config_kwargs)
    metric, compare_metrics = get_metric(args.dataset)

    train_score, train_time, retrain_score, retrain_time, param_values_list, prev_param_values = [], [], [], [], [], None

    model = MixtureSupernet.create(model.cpu(), in_place=True)
    model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable, 
        stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)
    if dims == 1:
        model.remove_module("chomp")
    opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay), arch_opt(model.arch_params(), lr=arch_lr, weight_decay=weight_decay)]

    optimizer = MixedOptimizer(opts)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
    lr_sched_iter = arch == 'convnext'
    
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

    for ep in range(epochs):
        time_start = default_timer()

        train_loss = train_one_epoch(model, optimizer, scheduler, args.device, train_loader, loss, clip, 1, n_train_temp, decoder, transform, lr_sched_iter, scale_grad=not args.baseline)

        if args.verbose and not args.baseline and (ep + 1) % args.print_freq == 0:
            print_grad(model, kernel_choices, dilation_choices)

        if (ep + 1) % args.print_freq == 0 or ep == epochs - 1:
            print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (default_timer() - time_start), "\ttrain loss:", "%.4f" % train_loss)

    
    param_values, ks, ds = [], [], []
    for name, param in model.named_arch_params():
        param_values.append(param.data.argmax(0))
        if args.verbose:
            print(name, param.data)
        ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

    param_values = torch.stack(param_values, dim = 0)
    print("[searched kernel pattern] ks:", ks, "\tds:", ds)
    # Arch Search Finished

    print("\n------- Start Retrain Hyperparameter Search --------")
    search_scores = []
    search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.root_dir, args.dataset, accum * batch_size, arch_retrain, False)
    retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds)
    retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)

    retrain_model = retrain_model.to(args.device)
    torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, 'init.pt'))

    hp_configs, search_epochs, subsampling_ratio = get_hp_configs(args.dataset, n_train, arch_retrain)
            
    search_n_temp = int(subsampling_ratio * search_n_train) + 1

    prev_lr = hp_configs[0][0]
    best_score_prev = None

    for lr, drop_rate, weight_decay, momentum in hp_configs:
        if lr != prev_lr:
            best_score = compare_metrics(search_scores)
            if best_score_prev is not None:
                if best_score == best_score_prev:
                    break
            best_score_prev = best_score
            prev_lr = lr

        retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
        retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
        retrain_model = retrain_model.to(args.device)

        retrain_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'init.pt')))
        retrain_model.set_arch_requires_grad(False)

        retrain_optimizer = get_optimizer(momentum=momentum, weight_decay=weight_decay, type=optimizer_type)(retrain_model.parameters(), lr=lr)
        retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda=weight_sched_train)
        retrain_time_start = default_timer()

        for retrain_ep in range(search_epochs):
            retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, search_train_loader, loss, retrain_clip, 1, search_n_temp, decoder, transform, lr_sched_iter)
        
        retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
        retrain_time_end = default_timer()
        search_scores.append(retrain_val_score)
        train_time.append(retrain_time_end - retrain_time_start)
        print("[hp search] bs = ", batch_size, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum, " time elapsed:", "%.4f" % (retrain_time_end - retrain_time_start), "\ttrain loss:", "%.4f" % retrain_loss, "\tval loss:", "%.4f" % retrain_val_loss, "\tval score:", "%.4f" % retrain_val_score)
        del retrain_model
    
    idx = np.argwhere(search_scores == compare_metrics(search_scores))[0][0]
    lr, drop_rate, weight_decay, momentum = hp_configs[idx]
    print("[selected hp] val_score = ", "%.4f" % search_scores[idx], "lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
    del search_train_loader, search_val_loader
    return ks, ds, compare_metrics(search_scores), (lr, drop_rate, weight_decay, momentum)

def run_expt_ray(grid_configs, args):
    args.arch = grid_configs['arch']
    args.grad_scale = grid_configs['grad_scale']
    args.pool_k = None

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

    dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, weight_sched_hpo, accum, clip, retrain_clip, validation_freq, retrain_freq, \
    einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(args.dataset, args)  
    
    if config_kwargs['arch_retrain_default'] is not None:
        arch_retrain = config_kwargs['arch_retrain_default']
    else:
        arch_retrain = args.arch

    if arch_retrain == "unet2d" or arch_retrain == "unet2d_small":
        optimizer_type = "AdamW"
    else:
        optimizer_type = "SGD"

    kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
    dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.root_dir, args.dataset, batch_size, args.arch, args.valid_split)
    model = get_model(args.arch, sample_shape, num_classes, config_kwargs)
    metric, compare_metrics = get_metric(args.dataset)

    train_score, train_time, retrain_score, retrain_time, param_values_list, prev_param_values = [], [], [], [], [], None

    model = MixtureSupernet.create(model.cpu(), in_place=True)
    model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable, 
        stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)
    if dims == 1:
        model.remove_module("chomp")
    opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay), arch_opt(model.arch_params(), lr=arch_lr, weight_decay=weight_decay)]

    optimizer = MixedOptimizer(opts)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
    lr_sched_iter = args.arch == 'convnext'
    
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

    for ep in range(epochs):

        train_loss = train_one_epoch(model, optimizer, scheduler, args.device, train_loader, loss, clip, 1, n_train_temp, decoder, transform, lr_sched_iter, scale_grad=not args.baseline)

    
    param_values, ks, ds = [], [], []
    for name, param in model.named_arch_params():
        param_values.append(param.data.argmax(0))
        ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

    param_values = torch.stack(param_values, dim = 0)
    # print("[searched kernel pattern] ks:", ks, "\tds:", ds)
    # Arch Search Finished

    search_scores = []
    best_score = -1000
    search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.root_dir, args.dataset, accum * batch_size, arch_retrain, False)
    retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds)
    retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)

    retrain_model = retrain_model.to(args.device)
    model_base = retrain_model
    torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, f'init_{args.arch}_{args.grad_scale}.pt'))

    hp_configs, search_epochs, subsampling_ratio = get_hp_configs(args.dataset, n_train, arch_retrain)
            
    search_n_temp = int(subsampling_ratio * search_n_train) + 1

    prev_lr = hp_configs[0][0]
    best_score_prev = None

    for lr, drop_rate, weight_decay, momentum in hp_configs:
        search_score = []
        if lr != prev_lr:
            best_score = compare_metrics(search_scores)
            if best_score_prev is not None:
                if best_score == best_score_prev:
                    break
            best_score_prev = best_score
            prev_lr = lr

        

        # retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
        # retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
        # retrain_model = retrain_model.to(args.device)

        # retrain_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'init.pt')))
        retrain_model = copy.deepcopy(model_base).to(args.device)
        retrain_model.set_arch_requires_grad(False)

        retrain_optimizer = get_optimizer(momentum=momentum, weight_decay=weight_decay, type=optimizer_type)(retrain_model.parameters(), lr=lr)
        retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda=weight_sched_hpo)

        for retrain_ep in range(search_epochs):
            retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, search_train_loader, loss, retrain_clip, 1, search_n_temp, decoder, transform, lr_sched_iter)
            retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            search_score.append(retrain_val_score)
            if retrain_val_score > best_score:
                best_score = retrain_val_score
                # best_model = copy.deepcopy(retrain_model)
                # torch.save(best_model.state_dict(), os.path.join(args.save_dir, f'init_{args.arch}_{args.grad_scale}.pt'))
        
        # retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
        # search_scores.append(retrain_val_score)
        search_scores.append(compare_metrics(search_score))
        del retrain_model
    assert best_score == compare_metrics(search_scores), f"Best score: {best_score}, compare_metrics: {compare_metrics(search_scores)}"
    idx = np.argwhere(search_scores == compare_metrics(search_scores))[0][0]
    lr, drop_rate, weight_decay, momentum = hp_configs[idx]
    # print("[selected hp] val_score = ", "%.4f" % search_scores[idx], "lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
    del search_train_loader, search_val_loader
    train.report({'val_score': compare_metrics(search_scores), 'ks': ks, 'ds': ds, 'lr': lr, 'drop_rate': drop_rate, 'weight_decay': weight_decay, 'momentum': momentum})

def grid_search_main(args, compare_metrics, archs, grad_scales):
    search_program = partial(run_expt_ray, args=args)

    search_space = {
        # "lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "arch_lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "weight_decay": tune.choice([1e-3, 1e-4, 1e-5]),
        "grad_scale": tune.grid_search(grad_scales),
        "arch": tune.grid_search(archs),
    }

    # Uncomment this to enable distributed execution
    # `ray.init(address="auto")`

    main_with_gpu = tune.with_resources(search_program, {"cpu": 5, "gpu": 1})

    mode = "max" if compare_metrics == np.max else "min"
    
    tuner = tune.Tuner(
        main_with_gpu,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=search_space,
    )

    results = tuner.fit()

    best_result = results.get_best_result("val_score", mode)

    print("Best trial config: {}".format(best_result.config))

    return best_result

if __name__ == '__main__':
    main_ray()