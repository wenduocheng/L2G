import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timeit import default_timer
from attrdict import AttrDict

from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler
from utils import count_params, count_trainable_params, calculate_stats
from embedder import get_tgt_model

# torch.cuda.set_device(2)
# from accelerate import Accelerator # 
# accelerator = Accelerator() #

def main(use_determined, args, info=None, context=None):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = accelerator.device # accelerate
    print("device:", args.device)
    root = '/datasets' if use_determined else './src/datasets'

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

    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    
    if load_embedder(use_determined, args):
        args.embedder_epochs = 0

    model, embedder_stats = get_tgt_model(args, root, sample_shape, num_classes, loss, False, use_determined, context)
    if args.quantize == True:
        model=model.bfloat16() ##
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split, quantize=args.quantize if hasattr(args, 'quantize') else False, rc_aug=args.rc_aug if hasattr(args, 'rc_aug') else False, shift_aug=args.shift_aug if hasattr(args, 'shift_aug') else False, one_hot=args.one_hot if hasattr(args, 'one_hot') else True)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    
    model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, model, None, None, n_train, freq=args.validation_freq, test=True)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved

    offset = 0 if ep_start == 0 else 1
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs #
    
    
    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer.params.lr)
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print("finetune method:", args.finetune_method)
    print("param count:", count_params(model), count_trainable_params(model))
    print(model)
    
    model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, model, optimizer, scheduler, n_train, freq=args.validation_freq)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []
    
    # model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler) # accelerate

    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")
    print("param count (before fine tuning):", count_params(model), count_trainable_params(model))

    # weights and biases
    if hasattr(args, 'use_wandb') and args.use_wandb:
            configs = {}
            for k, v in args.items():
                if k not in ['validation_freq','optimizer','scheduler','no_warmup_scheduler','use_wandb','device']:
                    configs[k] = v
            configs['optimizer_name']=args.optimizer.name 
            configs['initial_lr']=args.optimizer.params.lr
            configs['optimizer_betas']=args.optimizer.params.betas
            configs['optimizer_weight_decay']=args.optimizer.params.weight_decay
            configs['optimizer_momentum']=args.optimizer.params.momentum
            configs['scheduler_name']=args.scheduler.name
            configs['scheduler_warmup_epochs']=args.scheduler.params.warmup_epochs
            configs['scheduler_decay_epochs']=args.scheduler.params.decay_epochs
            configs['scheduler_sched']=args.scheduler.params.sched
            configs['scheduler_base']=args.scheduler.params.base 
            configs['lora_target_modules']=args.lora_target_modules if hasattr(args, 'lora_target_modules') else None

            print('configs:',configs)
            import wandb
            wandb.login(key=args.wandb_key)
            print('wandb login')
            run = wandb.init(
                # Set the project where this run will be logged
                name = str(args.dataset)+'+'+str(args.experiment_id), 
                entity="wenduoc",
                project=str(args.dataset),
                config= configs,
                settings=wandb.Settings(_service_wait=3000)
                )
            print('wandb init')

    for ep in range(ep_start, args.epochs + args.predictor_epochs):
        if not train_full and ep >= args.predictor_epochs:
            args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=n_train)
            train_full = True


        time_start = default_timer()

        train_loss = train_one_epoch(context, args, model, optimizer, scheduler, train_loader, loss, n_train, args.label_smoothing_factor, decoder, transform)
        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            # val_loss, val_score = evaluate(context, args, model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)
            val_loss, val_score = evaluate(args, model, val_loader, loss, metric)
            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

            if use_determined:
                id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                    
            if compare_metrics(train_score) == val_score:
                if not use_determined:
                    id_current = save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                id_best = id_current
            
            if hasattr(args, 'use_wandb') and args.use_wandb:
                wandb.log({
                    'train_loss': train_loss,  
                    'valid_loss': val_loss,
                    'lr'        : optimizer.param_groups[0]['lr'],
                    "val score:": val_score, 
                    "best val score:": compare_metrics(train_score)
                })


        if ep == args.epochs + args.predictor_epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = model
            test_time_start = default_timer()
            # test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_loss, test_score = evaluate(args, test_model, test_loader, loss, metric)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
            test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True)
            test_time_start = default_timer()
            # test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_loss, test_score = evaluate(args, test_model, test_loader, loss, metric)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            
            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
        if use_determined and context.preempt.should_preempt():
            print("paused")
            return

    if hasattr(args, 'use_wandb') and args.use_wandb:
        run.finish()

def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, label_smoothing_factor=None, decoder=None, transform=None):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()
    right, alldata = 0,0
    for i, data in enumerate(loader):

        x, y = data
        #print(x,x.max(),x.min(),x.mean())
        #print(y,y.max(),y.min(),y.float().mean())
            
        x, y = x.to(args.device), y.to(args.device) # accelerate
        out = model(x)
        #print((out.argmax(-1)==y).float().mean())
        # right += (y==out.argmax(-1)).float().sum()  # if using accuracy, count how many out is correct (=y)
        # alldata += len(x)

        if label_smoothing_factor != 0:
            y = y * (1 - label_smoothing_factor) + label_smoothing_factor * (1 - y)
      
        # print('out:',out.size(),'y:', y.size())
        l = loss(out, y)
        
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break
        #break

    if (not args.lr_sched_iter):
        scheduler.step()
    # print(right/alldata)
    return train_loss / temp


# def evaluate(context, args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
#     model.eval()
    
#     eval_loss, eval_score = 0, 0
#     right, alldata = 0,0


#     ys, outs, n_eval, n_data = [], [], 0, 0

#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             x, y = data
                                
#             x, y = x.to(args.device), y.to(args.device) # accelerate

#             out = model(x)

#             # right+=(out.argmax(-1)==y).float().sum() #
#             # alldata += len(x) #
    
#             outs.append(out) 
#             ys.append(y) 
#             n_data += x.shape[0]
        
#             if n_data >= args.eval_batch_size or i == len(loader) - 1:
#                 outs = torch.cat(outs, 0)
#                 ys = torch.cat(ys, 0)

#                 eval_loss += loss(outs, ys).item()
#                 # print('309',outs.shape)
#                 # print('309',ys.shape)
#                 # print('309',outs)
#                 # print('309',ys)
#                 # print(metric(outs, ys))
#                 eval_score += metric(outs, ys).item()
                
#                 n_eval += 1

#                 ys, outs, n_data = [], [], 0

#         eval_loss /= n_eval
#         eval_score /= n_eval


   
#     # eval_score = 1-(right/alldata).detach().cpu().numpy() # if using accuracy
#     return eval_loss, eval_score

def evaluate(args, model, loader, loss, metric):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
            #print("eval",x,x.min(),x.max(),x.mean())
            #print("eval",y,y.min(),y.max(),y.float().mean())
                                
            x, y = x.to(args.device), y.to(args.device) 

            out = model(x)

            outs.append(out) 
            ys.append(y)
            #print("eval",(out.argmax(-1)==y).float().mean())
            n_data += x.shape[0]
        
        
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)
        # print("eval total",(outs.argmax(-1)==ys).float().mean(),mc(ys.detach().cpu().numpy(),outs.argmax(-1).detach().cpu().numpy()))
        eval_loss += loss(outs, ys).item()

        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score

########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None


def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        np.random.set_state(rng_state_dict['numpy_rng_state'])
        random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')

    args = parser.parse_args()
    if args.config is not None:     
        import yaml

        with open(args.config, 'r') as stream:
            args = AttrDict(yaml.safe_load(stream)['hyperparameters'])
            main(False, args)

    else:
        import determined as det
        from determined.experimental import client
        from determined.pytorch import DataLoader

        info = det.get_cluster_info()
        args = AttrDict(info.trial.hparams)
        
        with det.core.init() as context:
            main(True, args, info, context)
