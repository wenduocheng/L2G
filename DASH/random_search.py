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

def main_ray():

    parser = argparse.ArgumentParser(description='DASH')
    parser.add_argument('--dataset', type=str, default='DEEPSEA', help='dataset name')
    parser.add_argument('--root_dir', type=str, default='./data/', help='root directory for the dataset')
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
    args.save_dir = 'results_acc/'  + args.dataset + '/' + 'random_search' +'/' + exp_id + "/" + str(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("------- Experiment Summary --------")
    print(args.__dict__)
    _, compare_metrics = get_metric(args.dataset)

    best_result = grid_search_main(args, compare_metrics)

    selected_arch, selected_grad_scale = best_result.config['arch'], best_result.config['grad_scale']
    lr, drop_rate, weight_decay, momentum = best_result.metrics['lr'], best_result.metrics['drop_rate'], best_result.metrics['weight_decay'], best_result.metrics['momentum']
    ks, ds = best_result.metrics['ks'], best_result.metrics['ds']
    
    print("\n------- Completed Grid Search -------")
    print("[selected search hp] arch = ", selected_arch, " grad_scale = ", selected_grad_scale)
    print("[searched kernel pattern] ks:", ks, "\tds:", ds)
    print("[selected retrain hp] lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
    print("[selected score] val_score = ", "%.4f" % best_result.metrics['val_score'])
    print(f"\n------- Results saved at {args.save_dir} -------")
    np.save(os.path.join(args.save_dir, 'search_hps.npy'), (selected_arch, selected_grad_scale))
    np.save(os.path.join(args.save_dir, 'retrain_hps.npy'), (lr, drop_rate, weight_decay, momentum))
    np.save(os.path.join(args.save_dir, 'searched_kernel_pattern.npy'), (ks, ds))

    


def run_expt_ray(grid_configs, args):
    arch = grid_configs['arch']
    grad_scale = grid_configs['grad_scale']
    args.arch = arch
    args.grad_scale = grad_scale
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
        arch_retrain = arch

    if arch_retrain == "unet" or arch_retrain == "unet2d_small":
        optimizer_type = "AdamW"
    else:
        optimizer_type = "SGD"

    kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
    dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

    metric, compare_metrics = get_metric(args.dataset)
    lr_sched_iter = arch == 'convnext'

    all_score = []

    all_kernel_list = []
    all_dilation_list = []
    for all_index in range(100):
        kernel_list = []
        dilation_list = []
        for index in range(18):
            kernel_list.append(random.choice(kernel_choices))
            dilation_list.append(random.choice(dilation_choices))
        all_kernel_list.append(kernel_list)
        all_dilation_list.append(dilation_list)

    for kernel_index in range(len(all_kernel_list)):
        ks = all_kernel_list[kernel_index]
        ds = all_dilation_list[kernel_index]

        search_scores = []
        search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.root_dir, args.dataset, accum * batch_size, arch_retrain, 0)
        
        decoder = search_data_kwargs['decoder'] if search_data_kwargs is not None and 'decoder' in search_data_kwargs else None
        transform = search_data_kwargs['transform'] if search_data_kwargs is not None and 'transform' in search_data_kwargs else None  
        
        retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds)
        retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)

        retrain_model = retrain_model.to(args.device)
        model_base = retrain_model
        # torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, 'init.pt'))
        if args.device == 'cuda':
            try:
                loss.cuda()
            except:
                pass

        hp_configs, search_epochs, subsampling_ratio = get_hp_configs(args.dataset, search_n_train, arch_retrain)
                
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

            # retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
            # retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
            # retrain_model = retrain_model.to(args.device)

            # retrain_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'init.pt')))
            retrain_model = copy.deepcopy(model_base).to(args.device)
            retrain_model.set_arch_requires_grad(False)

            retrain_optimizer = get_optimizer(momentum=momentum, weight_decay=weight_decay, type=optimizer_type)(retrain_model.parameters(), lr=lr)
            retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda=weight_sched_train)
            # retrain_time_start = default_timer()

            for retrain_ep in range(search_epochs):
                retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, search_train_loader, loss, retrain_clip, 1, search_n_temp, decoder, transform, lr_sched_iter)
            
            retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            # retrain_time_end = default_timer()
            search_scores.append(retrain_val_score)
            # train_time.append(retrain_time_end - retrain_time_start)
            # print("[hp search] bs = ", batch_size, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum, " time elapsed:", "%.4f" % (retrain_time_end - retrain_time_start), "\ttrain loss:", "%.4f" % retrain_loss, "\tval loss:", "%.4f" % retrain_val_loss, "\tval score:", "%.4f" % retrain_val_score)
            del retrain_model
        
        idx = np.argwhere(search_scores == compare_metrics(search_scores))[0][0]
        lr, drop_rate, weight_decay, momentum = hp_configs[idx]
        # print("[selected hp] val_score = ", "%.4f" % search_scores[idx], "lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
        del search_train_loader, search_val_loader
        all_score.append(compare_metrics(search_scores))
        if compare_metrics(search_scores) == compare_metrics(all_score):
            best_config = {'val_score': compare_metrics(search_scores), 'ks': ks, 'ds': ds, 'lr': lr, 'drop_rate': drop_rate, 'weight_decay': weight_decay, 'momentum': momentum}
        

    train.report(best_config)
    

def grid_search_main(args, compare_metrics):
    search_program = partial(run_expt_ray, args=args)

    search_space = {
        # "lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "arch_lr": tune.choice([1e-3, 1e-4, 1e-5]),
        # "weight_decay": tune.choice([1e-3, 1e-4, 1e-5]),
        "grad_scale": tune.grid_search([100, 500, 1000]),
        "arch": tune.grid_search(["wrn-22-1", "unet2d_small"]),
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