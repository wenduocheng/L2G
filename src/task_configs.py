import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, partial
import os
# import data loaders, task-specific losses and metrics
import sys 
sys.path.append('./')
# from src.data_loaders import load_imagenet, load_text, load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd, load_domainnet, load_pde, load_openml, load_drug
from src.data_loaders import load_nucleotide_transformer,load_genomic_benchmarks, load_deepsea_full, load_deepstarr, load_deepstarr
from src.data_loaders import load_hg38, load_text_large, load_text_llama, load_text_llama2,load_text_xs_pythia_1b, load_text_xs_flan_t5_small, load_text_xs_flan_t5_base, load_text_xs_flan_t5_large #
from src.utils import FocalLoss, LpLoss, conv_init, get_params_to_update, set_param_grad, set_grad_state
from src.utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value, inv_auroc, r2_score, inverse_score, auc_metric, nmse, rmse_loss, nrmse_loss 
from src.utils import binary_f1, mcc, pcc, pcc_deepstarr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False, quantize=False,rc_aug=False,shift_aug=False, one_hot=True):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]: 
        train_loader, val_loader, test_loader = load_genomic_benchmarks(root, batch_size, one_hot = one_hot, valid_split=valid_split, dataset_name = dataset, quantize=quantize,rc_aug=rc_aug, shift_aug=shift_aug)
    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        train_loader, val_loader, test_loader = load_nucleotide_transformer(root, batch_size, one_hot = one_hot, valid_split=valid_split, dataset_name = dataset, quantize=quantize,rc_aug=rc_aug, shift_aug=shift_aug)
    elif dataset == "deepstarr":
        train_loader, val_loader, test_loader = load_deepstarr(root, batch_size,one_hot = one_hot, valid_split=valid_split, quantize=quantize, rc_aug=rc_aug, shift_aug=shift_aug)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size,one_hot = one_hot, valid_split=valid_split,quantize=False,rc_aug=rc_aug, shift_aug=shift_aug)
    elif dataset == "DEEPSEA_FULL":
        train_loader, val_loader, test_loader = load_deepsea_full(root, batch_size, one_hot = one_hot,valid_split=valid_split,quantize=quantize,rc_aug=rc_aug, shift_aug=shift_aug)
    elif dataset == "hg38":
        train_loader, val_loader, test_loader = load_hg38(root, batch_size, maxsize=maxsize)
    elif dataset == "text_roberta_large":
        train_loader, val_loader, test_loader = load_text_large(root, batch_size, maxsize=maxsize)
    elif dataset == "text_llama":
        train_loader, val_loader, test_loader = load_text_llama(root, batch_size, maxsize=maxsize)
    elif dataset == "text_llama2":
        train_loader, val_loader, test_loader = load_text_llama2(root, batch_size, maxsize=maxsize)  
    elif dataset == "text_flan_t5_small":
        train_loader, val_loader, test_loader = load_text_xs_flan_t5_small(root, batch_size, maxsize=maxsize)  
    elif dataset == "text_flan_t5_base":
        train_loader, val_loader, test_loader = load_text_xs_flan_t5_base(root, batch_size, maxsize=maxsize)
    elif dataset == "text_flan_t5_large":
        train_loader, val_loader, test_loader = load_text_xs_flan_t5_large(root, batch_size, maxsize=maxsize)
    elif dataset == "text_pythia_1b":
        train_loader, val_loader, test_loader = load_text_xs_pythia_1b(root, batch_size, maxsize=maxsize)  

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_config(root, args):
    dataset = args.dataset
    args.infer_label = False
    args.activation = None
    args.target_seq_len = 512 if not hasattr(args, 'target_seq_len') else args.target_seq_len
    print("target_seq_len", args.target_seq_len)
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, num_classes = None, None
        loss = None

    elif dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]:  
        loss = nn.CrossEntropyLoss()
        args.infer_label = False
        if dataset == "dummy_mouse_enhancers_ensembl":
            dims, sample_shape, num_classes = 1, (1, 5, 4707), 2
        elif dataset == "demo_coding_vs_intergenomic_seqs":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
        elif dataset == "demo_human_or_worm":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
        elif dataset == "human_enhancers_cohn":
            dims, sample_shape, num_classes = 1, (1, 5, 500), 2
        elif dataset == "human_enhancers_ensembl":
            dims, sample_shape, num_classes = 1, (1, 5, 573), 2 
        elif dataset == "human_ensembl_regulatory":
            dims, sample_shape, num_classes = 1, (1, 5, 802), 3
        elif dataset == "human_nontata_promoters":
            dims, sample_shape, num_classes = 1, (1, 5, 251), 2
        elif dataset == "human_ocr_ensembl":
            dims, sample_shape, num_classes = 1, (1, 5, 593), 2
        if not args.one_hot:
            sample_shape = list(sample_shape)
            sample_shape[1] = 1
            sample_shape = tuple(sample_shape)
    elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        dims, sample_shape, num_classes = 1, (1, 5, 200), 2, #4#2 #1, 784), 10
        args.infer_label = False
        loss = nn.CrossEntropyLoss()
        if dataset == "enhancer":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 2
        elif dataset == "enhancers_types":
            dims, sample_shape, num_classes = 1, (1, 5, 200), 3
        elif dataset in ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']:
            dims, sample_shape, num_classes = 1, (1, 5, 500), 2
        elif dataset in ['promoter_all', 'promoter_no_tata', 'promoter_tata']:
            dims, sample_shape, num_classes = 1, (1, 5, 300), 2
        elif dataset in ['splice_sites_acceptors', 'splice_sites_donors']:
            dims, sample_shape, num_classes = 1, (1, 5, 600), 2
        elif dataset == 'splice_sites_all':
            dims, sample_shape, num_classes = 1, (1, 5, 400), 3
        
        if not args.one_hot:
            sample_shape = list(sample_shape)
            sample_shape[1] = 1
            sample_shape = tuple(sample_shape)
   

    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))
        # loss = nn.CrossEntropyLoss()
        args.infer_label = True

        if not args.one_hot:
            sample_shape = list(sample_shape)
            sample_shape[1] = 1
            sample_shape = tuple(sample_shape)
            loss = nn.BCEWithLogitsLoss(pos_weight=1 * torch.ones((36, )))

    elif dataset == "DEEPSEA_FULL":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 919
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((919, )))
        args.infer_label = True
        if not args.one_hot:
            sample_shape = list(sample_shape)
            sample_shape[1] = 1
            sample_shape = tuple(sample_shape)
            loss = nn.BCEWithLogitsLoss(pos_weight=1 * torch.ones((919, )))
    
    elif dataset == "deepstarr_dev" or dataset == "deepstarr_hk":
        dims, sample_shape, num_classes = 1, (1, 4, 249), 1
        loss = nn.MSELoss()
        args.infer_label = False
    
    elif dataset == "deepstarr":
        dims, sample_shape, num_classes = 1, (1, 4, 249), 2 # (1, 4, 249)
        loss = nn.MSELoss()
        args.infer_label = False

        if not args.one_hot:
            sample_shape = list(sample_shape)
            sample_shape[1] = 1
            sample_shape = tuple(sample_shape)


    return dims, sample_shape, num_classes, loss, args



def get_metric(root, dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return accuracy, np.max
    if dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]:  
        return accuracy, np.max
    if dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']: 
        return mcc, np.max 
    if dataset in ['promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors', 'splice_sites_all']: 
        # return mcc, np.max
        return binary_f1, np.max
    if dataset in ['splice_sites_all']: 
        # return mcc, np.max
        return accuracy, np.max
    if dataset == "DEEPSEA":
        return auroc, np.max
    if dataset == "DEEPSEA_FULL":
        return auroc, np.max
    if dataset == "deepstarr":
        return pcc_deepstarr, np.max
        # return pcc, np.max




def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])


def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter


def get_optimizer_scheduler(args, model, module=None, n_train=1):
    if module is None:
        set_grad_state(model, True, args.finetune_method)
        set_param_grad(model, args.finetune_method)
        optimizer = get_optimizer(args.optimizer.name, args.optimizer.params)(get_params_to_update(model, ""))
        print('final optimizer',optimizer.defaults)
        lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return args, model, optimizer, scheduler

    elif module == 'embedder-without-linear':  # exclude linear
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] *= 10
        params_to_update = get_params_to_update(model, "", module='embedder-without-linear') #
        embedder_optimizer = get_optimizer(args.optimizer.name, embedder_optimizer_params)(params_to_update) # 
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'embedder-pretraining': # include linear
        # embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
       
        # lr = 0.01
        
        # momentum = 0.99
        # weight_decay = 0.0005
        # embedder_optimizer = partial(torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)(get_params_to_update(model, ""))
        
        # print('embedder optimizer',embedder_optimizer.defaults)

        # lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        # embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        # return args, model, embedder_optimizer, embedder_scheduler
    
        if hasattr(args,'embedder_optimizer'):
            if args.run_dash:
                for embedder_type in ["unet", "wrn", "deepsea"]:
                    dash_result_path = f"./dash_results/results_acc/{args.dataset}/search_init/{embedder_type}/{args.seed}/dash_final_results.npy"
                    print(f"Checking path for embedder architecture: {embedder_type}")
                    
                    if os.path.exists(dash_result_path):
                        print(f"Found DASH results for architecture: {embedder_type}")
                        # Proceed with further processing if necessary
                        break  # Exit loop as the file was found
                    else:
                        print(f"DASH results not found for architecture: {embedder_type}. Trying the next one.")

                # dash_result_path = f"./dash_results/results_acc/{args.dataset}/search_init/wrn/{args.seed}/dash_final_results.npy"
                dash_results = np.load(dash_result_path,allow_pickle=True).item()
                args.embedder_optimizer.params.lr = dash_results['lr']
                args.embedder_optimizer.params.weight_decay = dash_results['weight decay']
                args.embedder_optimizer.params.momentum = dash_results['momentum']
                print('lr: ',dash_results['lr'], 'momentum: ',dash_results['momentum'], 'weight_decay: ',dash_results['weight decay'])
                print('Load embedder optimizer hps from DASH results!')

            embedder_optimizer_params = copy.deepcopy(args.embedder_optimizer.params)
        
            params_to_update = get_params_to_update(model, "") #

            print(embedder_optimizer_params)
            embedder_optimizer = get_optimizer(args.embedder_optimizer.name, embedder_optimizer_params)(params_to_update) # 
            lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
            embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)
        else: 
            lr = 0.01
        
            momentum = 0.99
            weight_decay = 0.0005
            embedder_optimizer = partial(torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)(get_params_to_update(model, ""))
            
            print('embedder optimizer',embedder_optimizer.defaults)

            lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
            embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler
    
    elif module == 'embedder-with-linear': # include linear
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] *= 10
        print('embedder optimizer',embedder_optimizer_params)
        embedder_optimizer = get_optimizer(args.optimizer.name, embedder_optimizer_params)(get_params_to_update(model, ""))
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'predictor':

        try:
            predictor = model.predictor
            set_grad_state(model, False)
            for n, m in model.embedder.named_parameters():
                m.requires_grad = True
            for n, m in model.predictor.named_parameters():
                m.requires_grad = True

            predictor_optimizer_params = copy.deepcopy(args.optimizer.params)
            if predictor_optimizer_params['lr'] <= 0.001:
                predictor_optimizer_params['lr'] *= 10
            predictor_optimizer = get_optimizer(args.optimizer.name, predictor_optimizer_params)(get_params_to_update(model, ""))
            lr_lambda, args.lr_sched_iter = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.predictor_epochs, 1)
            predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda=lr_lambda)

            return args, model, predictor_optimizer, predictor_scheduler
        except:
            print("No predictor module.")

