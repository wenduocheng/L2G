import os
import argparse
import random
import math 
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timeit import default_timer
from sklearn import metrics
from networks.vq import Encoder_v2, Encoder_v3
# from task_configs import get_data
# from utils import calculate_auroc, calculate_aupr, auroc, inverse_score
# from utils import auroc_aupr,inverse_two_scores #
from networks.wrn1d import ResNet1D, ResNet1D_v2, ResNet1D_v3 # 
import copy
import scipy
from scipy import stats
from helper_scripts.genomic_benchmarks_utils import GenomicBenchmarkDataset, CharacterTokenizer, combine_datasets, NucleotideTransformerDataset 
from torch.utils.data import DataLoader


# torch.cuda.set_device(3)
print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", DEVICE)



#-----------------------Configurations
configs = {'weight':'resnet', # resnet, unet, nas-deepsea ,
           'dataset':'deepstarr', # DEEPSEA_FULL
           'one_hot':True,
           'lr':0.01,
           'optimizer': 'SGD',
           'weight_decay': 0.0005,
           'momentum':0.9,
           'batch_size':128,
           'epochs':80,
            'channels': [16,32,64],
            'drop_out':0.05,
            'rc_aug':True,
            'shift_aug':True}
# configs = {'weight':'CNN', # resnet, unet, nas-deepsea ,
#            'dataset':'H3K14ac', # DEEPSEA_FULL
#            'one_hot':True,
#            'lr':0.01,
#            'optimizer': 'Adam',
#            'weight_decay': 0.0005,
#            'momentum':0.99,
#            'batch_size':128,
#            'epochs':30,
#             'channels': [16,32,64],
#             'drop_out':0.2,
#             'rc_aug':False,
#             'shift_aug':False}
print(configs)
# weight: nas-deepsea, one_hot True, lr 0,01

root = "/home/wenduoc/ORCA/L2G/src/datasets"

#--------------------------Metric
def mcc(output, target):
    target = target.cpu().detach().numpy()
    output = np.argmax(output.cpu().detach().numpy(), axis=1)

    return np.float64(metrics.matthews_corrcoef(output, target))
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res
def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    score = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, score
def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr
def auroc(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()

    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc
def auroc_aupr(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()

    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc, avg_aupr
def pcc(output, target):
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    target = np.squeeze(target)
    output = np.squeeze(output)

    pearson_corr, _ = stats.pearsonr(target, output)
    
    return np.float64(pearson_corr)
def pcc_deepstarr(output, target):
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    correlations = []

    for i in range(output.shape[1]): # 2
        # Extract the i-th column from both target and output
        target_col = target[:, i]
        output_col = output[:, i]
        
        # Calculate Pearson correlation coefficient for the current column
        corr, _ = stats.pearsonr(target_col, output_col)
        correlations.append(corr)

    # Calculate the average Pearson correlation across all columns
    avg_corr = np.mean(correlations)

    return np.float64(avg_corr)
class inverse_score(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, output, target):
        return 1 - self.score_func(output, target)
class inverse_two_scores(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, output, target):
        return 1 - self.score_func(output, target)[0], 1 - self.score_func(output, target)[1]

#-----------------------Data loaders
def load_deepsea(root, batch_size, one_hot = True, valid_split=-1,rc_aug=False, shift_aug=False):
    filename = root + '/deepsea_filtered.npz'

    if not os.path.isfile(filename):
        with open(filename, 'wb') as f:
            f.write(requests.get("https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz").content)

    data = np.load(filename)

    y_train = torch.from_numpy(np.concatenate((data['y_train'], data['y_val']), axis=0)).float() 
    y_test = torch.from_numpy(data['y_test']).float()   # shape = (149400, 36)
    if one_hot:
        x_train = torch.from_numpy(np.concatenate((data['x_train'], data['x_val']), axis=0)).transpose(-1, -2).float()  
        x_test = torch.from_numpy(data['x_test']).transpose(-1, -2).float()  # shape = (149400, 1000, 4)
    else:
        x_train = torch.from_numpy(np.argmax(np.concatenate((data['x_train'], data['x_val']), axis=0), axis=2)).unsqueeze(-2).float()
        x_test = torch.from_numpy(np.argmax(data['x_test'], axis=2)).unsqueeze(-2).float()

        if rc_aug:
            print('reverse complement')
            x_train2 = copy.deepcopy(x_train) 
            y_train2 = copy.deepcopy(y_train)

            x_train = torch.concatenate(x_train,x_train2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return train_loader, None, test_loader


def load_deepstarr(root, batch_size, one_hot = True, valid_split=-1, quantize=False, rc_aug=True, shift_aug=False):
    # filename = root + '/deepstarr' + '/Sequences_activity_all.txt'
    filename = root + '/deepstarr' + '/Sequences_activity_subset.txt'

    data = pd.read_table(filename)
    nucleotide_dict = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]} # sometimes there are Ns

    # define a function to one-hot encode a single DNA sequence
    def one_hot_encode(seq):
        return np.array([nucleotide_dict[nuc] for nuc in seq])

    # function to load sequences and enhancer activity
    def prepare_input(data_set):
        # one-hot encode DNA sequences, apply function
        seq_matrix = np.array(data_set['Sequence'].apply(one_hot_encode).tolist())
        print(seq_matrix.shape) # dimensions are (number of sequences, length of sequences, nucleotides)

        # Get output array with dev and hk activities
        Y_dev = data_set.Dev_log2_enrichment
        Y_hk = data_set.Hk_log2_enrichment
        Y = [Y_dev, Y_hk]

        return seq_matrix, Y
    
    # Process data for train/val/test sets
    X_train, Y_train = prepare_input(data[data['set'] == "Train"])
    X_valid, Y_valid = prepare_input(data[data['set'] == "Val"])
    X_test, Y_test = prepare_input(data[data['set'] == "Test"])

    if one_hot:
        x_train = torch.from_numpy(X_train).transpose(-1, -2).float() 
        x_val = torch.from_numpy(X_valid).transpose(-1, -2).float()
        x_test = torch.from_numpy(X_test).transpose(-1, -2).float()
    else:
        x_train = torch.from_numpy(np.argmax(X_train, axis=2)).unsqueeze(-2).float() 
        x_val = torch.from_numpy(np.argmax(X_valid, axis=2)).unsqueeze(-2).float()
        x_test = torch.from_numpy(np.argmax(X_test, axis=2)).unsqueeze(-2).float()
    
    y_train = torch.stack( ( torch.from_numpy(np.array(Y_train[0])), torch.from_numpy(np.array(Y_train[1]))  ), dim=1).float() 
    y_val   = torch.stack( ( torch.from_numpy(np.array(Y_valid[0])), torch.from_numpy(np.array(Y_valid[1]))  ), dim=1).float() 
    y_test = torch.stack( ( torch.from_numpy(np.array(Y_test[0])), torch.from_numpy(np.array(Y_test[1]))  ), dim=1).float()  

    if quantize:
        x_train = x_train.to(torch.bfloat16)  
        y_train = y_train.to(torch.bfloat16) 
        x_val = x_val.to(torch.bfloat16)  
        y_val = y_val.to(torch.bfloat16)  
        x_test = x_test.to(torch.bfloat16)  
        y_test = y_test.to(torch.bfloat16) 

    del X_train, Y_train, X_valid, Y_valid, X_test, Y_test

    if shift_aug:
        if not one_hot:
            def shift_seqs(seq, shift=0):
                seq = copy.deepcopy(seq)
                if shift > 0:
                    seq[:, :, :-shift] = seq.clone()[:, :, shift:]
                    seq[:, :, -shift:] = 4  # Fill with special token
                elif shift < 0:  # shift up 
                    seq[:, :, shift:] = seq.clone()[:, :, :-shift]
                    seq[:, :, :shift] = 4  # Fill with special token
                return seq
            x_train2 = shift_seqs(x_train,shift=3)
            x_train3 = shift_seqs(x_train,shift=-3)
            x_train = torch.cat((x_train, x_train2), dim=0)
            x_train = torch.cat((x_train, x_train3), dim=0)
            y_train2 = copy.deepcopy(y_train)
            y_train3 = copy.deepcopy(y_train)
            y_train = torch.cat((y_train, y_train2), dim=0)
            y_train = torch.cat((y_train, y_train3), dim=0)
            del x_train2, x_train3, y_train2, y_train3
        else:
            def shift_seqs(seq, shift=0):
                seq = copy.deepcopy(seq)
                if shift > 0:
                    seq[:, :, :-shift] = seq.clone()[:, :, shift:]
                    seq[:, :, -shift:] = 0
                elif shift < 0:  # shift up 
                    seq[:, :, shift:] = seq.clone()[:, :, :-shift]
                    seq[:, :, :shift] = 0
                return seq
            x_train2 = shift_seqs(x_train,shift=3)
            x_train3 = shift_seqs(x_train,shift=-3)
            x_train = torch.cat((x_train, x_train2), dim=0)
            x_train = torch.cat((x_train, x_train3), dim=0)
            y_train2 = copy.deepcopy(y_train)
            y_train3 = copy.deepcopy(y_train)
            y_train = torch.cat((y_train, y_train2), dim=0)
            y_train = torch.cat((y_train, y_train3), dim=0)
            del x_train2, x_train3, y_train2, y_train3



    print('x_train',x_train.shape)
    print('y_train',y_train.shape)
    print('x_val',x_val.shape)
    print('y_val',y_val.shape)
    print('x_test',x_test.shape)
    print('y_test',y_test.shape)

    if valid_split > 0:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, None, test_loader

def load_deepsea_full(root, batch_size, one_hot = True, valid_split=-1,quantize=False,rc_aug=False, shift_aug=False):
    # import mat73
    # train_data = mat73.loadmat(root + '/deepsea_full' + '/deepsea_full_train.mat')

    trainxdata = np.load(root + '/deepsea_full' + '/deepsea_full_trainx.npz')['data']
    trainydata = np.load(root + '/deepsea_full' + '/deepsea_full_trainy.npz')['data']
    valid_data = scipy.io.loadmat(root + '/deepsea_full' + '/deepsea_full_valid.mat')
    test_data = scipy.io.loadmat(root + '/deepsea_full' + '/deepsea_full_test.mat')

    # with h5py.File(path+"train.mat", 'r') as file:
    #     x_train = file['trainxdata']
    #     y_train = file['traindata']
    #     x_train = np.transpose(x_train, (2, 1, 0))    
    #     y_train = np.transpose(y_train, (1, 0)) 
        
    if one_hot:
        x_train = torch.from_numpy(trainxdata).float() 
        x_val = torch.from_numpy(valid_data['validxdata']).float() 
        x_test = torch.from_numpy(test_data['testxdata']).float() 
    else:
        x_train = torch.from_numpy(np.argmax(trainxdata, axis=1)).unsqueeze(-2).float() 
        x_val = torch.from_numpy(np.argmax(valid_data['validxdata'], axis=1)).unsqueeze(-2).float()
        x_test = torch.from_numpy(np.argmax(test_data['testxdata'], axis=1)).unsqueeze(-2).float() 
    
    y_train = torch.from_numpy(trainydata).float() 
    y_val = torch.from_numpy(valid_data['validdata']).float()  
    y_test = torch.from_numpy(test_data['testdata']).float()  
    del trainxdata, trainydata, valid_data, test_data

    if quantize:
        x_train = x_train.to(torch.bfloat16)  
        # y_train = y_train.to(torch.bfloat16) 
        x_val = x_val.to(torch.bfloat16)  
        # y_val = y_val.to(torch.bfloat16)  
        x_test = x_test.to(torch.bfloat16)  
        # y_test = y_test.to(torch.bfloat16) 

    print('x_train',x_train.shape)
    print('y_train',y_train.shape)
    print('x_val',x_val.shape)
    print('y_val',y_val.shape)
    print('x_test',x_test.shape)
    print('y_test',y_test.shape)

    if valid_split > 0:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        del x_train, y_train, x_val, y_val, x_test, y_test
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
        del x_train, y_train, x_val, y_val, x_test, y_test
        return train_loader, None, test_loader




def load_nucleotide_transformer(root, batch_size, one_hot = True, valid_split=-1, dataset_name = 'enhancers', quantize=False, rc_aug = False, shift_aug=False):
    # Define a dictionary mapping dataset names to max_length
    max_length_dict = {
        "enhancers": 200,
        "enhancers_types": 200,  # nclass=3
        "H3": 500,
        "H3K4me1": 500,
        "H3K4me2": 500,
        "H3K4me3": 500,
        "H3K9ac": 500,
        "H3K14ac": 500,
        "H3K36me3": 500,
        "H3K79me3": 500,
        "H4": 500,
        "H4ac": 500,
        "promoter_all": 300,
        "promoter_no_tata": 300,
        "promoter_tata": 300,
        "splice_sites_acceptors": 600,
        "splice_sites_donors": 600,
        "splice_sites_all": 400
    }
    # Use the dictionary to get max_length
    max_length = max_length_dict.get(dataset_name)
    use_padding = True
    add_eos = False  # add end of sentence token
    tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )
    ds_train = NucleotideTransformerDataset(
            max_length = max_length,
            dest_path = root + '/nucleotide_transformer_downstream_tasks',
            use_padding = use_padding,
            split = 'train',
         
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            rc_aug=False,
            add_eos=add_eos,
            one_hot=one_hot,
            quantize=quantize
        )
    ds_test = NucleotideTransformerDataset(
        max_length = max_length,
        dest_path = root+ '/nucleotide_transformer_downstream_tasks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=False,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
        
    if shift_aug:
        ds_train3 = NucleotideTransformerDataset(
            max_length = max_length,
            dest_path=root + '/nucleotide_transformer_downstream_tasks',
            use_padding = use_padding,
            split = 'train',
            
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            rc_aug=True,
            add_eos=add_eos,
            one_hot=one_hot,
            quantize=quantize
        )
        # ds_train4 = NucleotideTransformerDataset(
        #     max_length = max_length,
        #     dest_path=root + '/nucleotide_transformer_downstream_tasks',
        #     use_padding = use_padding,
        #     split = 'train',
            
        #     tokenizer=tokenizer,
        #     dataset_name=dataset_name,
        #     rc_aug=True,
        #     add_eos=add_eos,
        #     one_hot=one_hot,
        #     quantize=quantize
        # )
        ds_train3.shift=3
        # ds_train4.shift=1
        ds_train = combine_datasets(ds_train, ds_train3)
        # ds_train=combine_datasets(ds_train,ds_train4)
        #ds_test=interleave_datasets(ds_test,ds_test2)

    if rc_aug:
        ds_train2 = NucleotideTransformerDataset(
            max_length = max_length,
            dest_path=root + '/nucleotide_transformer_downstream_tasks',
            use_padding = use_padding,
            split = 'train',
            
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            rc_aug=True,
            add_eos=add_eos,
            one_hot=one_hot,
            quantize=quantize
        )
        ds_train = combine_datasets(ds_train, ds_train2)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, None, test_loader

def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False, quantize=False,rc_aug=False,shift_aug=False, one_hot=True):
    data_kwargs = None

    if dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size,one_hot = one_hot, valid_split=valid_split,rc_aug=rc_aug, shift_aug=shift_aug)
    if dataset == "DEEPSEA_FULL":
        train_loader, val_loader, test_loader = load_deepsea_full(root, batch_size,one_hot = one_hot, valid_split=valid_split,rc_aug=rc_aug, shift_aug=shift_aug)
    if dataset == "deepstarr":
        train_loader, val_loader, test_loader = load_deepstarr(root, batch_size,one_hot = one_hot, valid_split=valid_split,rc_aug=rc_aug, shift_aug=shift_aug)
    if dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
        train_loader, val_loader, test_loader =load_nucleotide_transformer(root, 64, one_hot, -1, dataset, False, rc_aug, shift_aug)
    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs

print('load data')
# if dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]: 
#     metric = accuracy
#     test_loader, test_loader2 = load_genomic_benchmarks2(root, 64, False, -1, dataset, False, False, False)
if configs['dataset'] in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
    metric = inverse_score(mcc)
    num_classes=2
    if configs['dataset'] == "enhancer":
        dims, sample_shape, num_classes = 1, (1, 5, 200), 2
    elif configs['dataset'] == "enhancers_types":
        dims, sample_shape, num_classes = 1, (1, 5, 200), 3
    elif configs['dataset'] in ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac']:
        dims, sample_shape, num_classes = 1, (1, 5, 500), 2
    elif configs['dataset'] in ['promoter_all', 'promoter_no_tata', 'promoter_tata']:
        dims, sample_shape, num_classes = 1, (1, 5, 300), 2
    elif configs['dataset'] in ['splice_sites_acceptors', 'splice_sites_donors']:
        dims, sample_shape, num_classes = 1, (1, 5, 600), 2
    elif configs['dataset'] == 'splice_sites_all':
        dims, sample_shape, num_classes = 1, (1, 5, 400), 3
    # train_loader, _, test_loader = load_nucleotide_transformer(root, 64, configs['one_hot'], -1, configs['dataset'], False, configs['rc_aug'], configs['shift_aug'])
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, configs['dataset'],configs['batch_size'], valid_split=False, maxsize=None, get_shape=False, quantize=False,rc_aug=False,shift_aug=False, one_hot=configs['one_hot'])
    loss = torch.nn.CrossEntropyLoss().to(DEVICE)
elif configs['dataset'] == 'deepstarr': 
    metric = pcc_deepstarr
    # _, _, test_loader, _, _, n_test, data_kwargs = get_data(root, dataset, args.batch_size, args.valid_split, quantize=args.quantize, rc_aug=args.rc_aug, shift_aug=args.shift_aug, one_hot=args.one_hot)
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, configs['dataset'],configs['batch_size'], valid_split=False, maxsize=None, get_shape=False, quantize=False,rc_aug=False,shift_aug=False, one_hot=configs['one_hot'])
    dims, sample_shape, num_classes = 1, (1, 4, 249), 2
    loss = nn.MSELoss().to(DEVICE)
elif configs['dataset'] == 'DEEPSEA':
    metric = inverse_score(auroc)
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA', batch_size=configs['batch_size'],valid_split=False, one_hot=configs['one_hot'])
    sample_shape, num_classes = (1, 4, 1000), 36
    loss = torch.nn.BCEWithLogitsLoss(pos_weight = 4 * torch.ones((36, )).to(DEVICE))
elif configs['dataset'] == 'DEEPSEA_FULL':
    metric = inverse_score(auroc)
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, 'DEEPSEA_FULL', batch_size=configs['batch_size'],valid_split=False, one_hot=configs['one_hot'])
    sample_shape, num_classes = (1, 4, 1000), 919
    loss = torch.nn.BCEWithLogitsLoss(pos_weight = 4 * torch.ones((919, )).to(DEVICE))

for batch in train_loader: 
        x, y = batch
   
        print('x:',x.size())
        print('y:',y.size())
        break

#-----------------------Model


# NAS-BENCH-360 DeepSea
class NAS_DeepSEA(nn.Module):
    def __init__(self, ):
        super(NAS_DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, padding = 4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding = 4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding = 4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(63*960, 925) 
        self.Linear2 = nn.Linear(925, 36)
        

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = self.flatten(x)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
  
        return x

class Baseline(nn.Module):
    def __init__(self, ):
        super(Baseline, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=768, kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(768*500, 36)
        
    def forward(self, input):
        x = self.Conv1(input)
        x = self.flatten(x)
        x = self.Linear(x)
        return x

class DASH_DEEPSEA(nn.Module):
    def __init__(self, ks=None,ds=None):
        super(DASH_DEEPSEA, self).__init__()
        k, d = 8 if ks is None else ks[0], 1 if ds is None else ds[0]
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[1], 1 if ds is None else ds[1]
        self.conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=k, dilation=d, padding=k//2 * d)
        k, d = 8 if ks is None else ks[2], 1 if ds is None else ds[2]
        self.conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=k, dilation=d, padding=k//2 * d)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(59520, 925)
        self.linear2 = nn.Linear(925, 36)

    def forward(self, input):
        # print("input", input.shape)
        s = input.shape[-1]
        x = self.conv1(input)[..., :s]
        x = F.relu(x)
        # print("1", x.shape)
        x = self.maxpool(x)
        # print("2", x.shape)
        x = self.drop1(x)
        # print("3", x.shape)
        s = x.shape[-1]
        x = self.conv2(x)[..., :s]
        # print("4", x.shape)
        x = F.relu(x)
        x = self.maxpool(x)
        # print("5", x.shape)
        x = self.drop1(x)
        # print("6", x.shape)
        s = x.shape[-1]
        x = self.conv3(x)[..., :s]
        # print("7", x.shape)
        x = F.relu(x)
        x = self.drop2(x)
        # print("8", x.shape)
        x = x.view(x.size(0), -1)
        # print("9", x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


 
class CNN(nn.Module):
    def __init__(self, number_of_classes=2, vocab_size=5, embedding_dim=100, input_len=500):
        super(CNN, self).__init__()

        # if number_of_classes == 2:
        #     self.is_multiclass = False
        #     number_of_output_neurons = 1
        #     loss = F.binary_cross_entropy_with_logits
        #     output_activation = nn.Sigmoid()
        # else:
        #     self.is_multiclass = True
        #     number_of_output_neurons = number_of_classes
        #     loss = torch.nn.CrossEntropyLoss()
        #     output_activation = lambda x: x
        
        number_of_output_neurons = number_of_classes
        loss = torch.nn.CrossEntropyLoss()
        output_activation = lambda x: x

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_model = nn.Sequential(
            # nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True),
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=8, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            nn.Linear(self.count_flatten_size(input_len), 512),
            nn.Linear(512, number_of_output_neurons)
        )
        self.output_activation = output_activation
        self.loss = loss

    def count_flatten_size(self, input_len):
        # zeros = torch.zeros([5, input_len], dtype=torch.long)
        # x = self.embeddings(zeros)
        # x = x.transpose(1, 2)
        # x = self.cnn_model(x)
        zeros = torch.zeros([1, 5, input_len], dtype=torch.float)
        x = self.cnn_model(zeros)
        print('flatten size',x.size())
        return x.size()[-1]

    def forward(self, x):
        # print('713',x.shape)
        # x = self.embeddings(x.long())
        # x = x.transpose(1, 2)
        # print('716',x.shape)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        x = self.output_activation(x)
        return x



if configs['weight']=='nas-deepsea':
    model = NAS_DeepSEA(number_of_classes=2, vocab_size=5, embedding_dim=100, input_len=500).to(DEVICE)
if configs['weight']=='CNN': # genomic benchmark cnn
    model = CNN(number_of_classes=num_classes, vocab_size=5, embedding_dim=100, input_len=sample_shape[-1]).to(DEVICE)
elif configs['weight']=='unet':
    # if not configs['one_hot']:
    #     model = Encoder_v3(768, channels = configs['channels'], dropout=configs['drop_out'], f_channel=input_shape[-1],num_class=num_classes,ks=None,ds=None,downsample=False,seqlen=sample_shape[-1])
    # else: 
    model = Encoder_v2(sample_shape[1], channels = configs['channels'], dropout=configs['drop_out'], f_channel=sample_shape[-1],num_class=num_classes,ks=None,ds=None,downsample=False,seqlen=sample_shape[-1])
    model = model.to(DEVICE)
elif configs['weight']=='resnet':
    in_channel=sample_shape[1]
    # mid_channels=min(4 ** (num_classes // 10 + 1), 64)
    mid_channels=128
    dropout=configs['drop_out']

    ks = [3, 3, 5, 3, 3, 5, 3, 9, 11]
    ds= [1, 1, 1, 1, 1, 1, 1, 1, 1]
    if configs['dataset']=='DEEPSEA' or configs['dataset']=='DEEPSEA_FULL':
        ks=[15, 19, 19, 7, 7, 7, 19, 19, 19]
        ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]
        in_channel=4
    
    elif configs['dataset']=='H3K4me3':
        ks=[3, 5, 3, 3, 3, 5, 5, 7, 5]
        ds=[1, 1, 1, 1, 5, 1, 1, 1, 1]
    # elif configs['dataset']=='H3K4me2':
    #     ks=[11,9,3,3,5,5,5,5,9]
    #     ds=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    elif configs['dataset']=='H3K4me1':
        ks=[3, 11, 3, 5, 9, 5, 5, 11, 11]
        ds=[1, 7, 1, 1, 1, 1, 1, 1, 1]
    elif configs['dataset']=='deepstarr':
        ks=[7, 15, 3, 11, 19, 7, 11, 3, 3]
        ds=[7, 1, 7, 1, 1, 7, 1, 7, 1]
    
    # ks = [3, 3, 5, 3, 3, 5, 3, 9, 11]
    # ds= [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # ks=[19, 19, 19, 19, 19, 19, 11, 19, 19]
    # ds=[1, 1, 15, 1, 15, 7, 15, 15, 1]
    print('ks:',ks, 'ds:',ds)
    activation=None
    remain_shape=False
    # model = ResNet1D_v2(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, input_shape=sample_shape, embed_dim=768).to(DEVICE)
    model = ResNet1D_v3(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape).to(DEVICE)
print(model)




# --------------- Train and Evaluate Functions


def evaluate(model, loader, loss, metric):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(DEVICE), y.to(DEVICE) 

            out = model(x)

            outs.append(out) 
            ys.append(y) 
            n_data += x.shape[0]
        
        
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()
        # print(outs)
        # print(ys)
        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score

def train_one_epoch(model, optimizer, scheduler, loader, loss, temp):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        x, y = data 
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        # print('699', out.shape, y.shape)
        l = loss(out, y)
        l.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
       
        optimizer.step()
        optimizer.zero_grad()
        
        # scheduler.step()

        train_loss += l.item()

        if i >= temp - 1:
            break

    # scheduler.step()

    return train_loss / temp



# ---------------------Optimizer, scheduler
if configs['optimizer']=='SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=configs["lr"], momentum=configs['momentum'], weight_decay=configs['weight_decay']) # momentum 0.9
elif configs['optimizer']=='Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), weight_decay=configs['weight_decay'])

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=0)
base, accum = 0.2, 1
# sched = [30, 60, 90, 120, 160]
sched = [30, 60, 90, 120, 160]
def weight_sched_train(epoch):    
    optim_factor = 0
    for i in range(len(sched)):
        if epoch > sched[len(sched) - 1 - i]:
            optim_factor = len(sched) - i
            break
    return math.pow(base, optim_factor)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = weight_sched_train)




# training and validating
print("\n------- Start Training --------")



train_time, train_score, train_losses = [], [], []

for ep in range(configs['epochs']):
    # train
    time_start = default_timer()
    train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss, n_train)
    train_time_ep = default_timer() -  time_start 
    # val    
    val_loss, val_score = evaluate(model, val_loader, loss, metric)
    
    train_losses.append(train_loss)
    train_score.append(val_score)
    train_time.append(train_time_ep)

    scheduler.step()
    
    print("[train", "full", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % np.max(train_score))
    
    # if np.max(train_score) == val_score:
    #     # torch.save({'model_state_dict':model.state_dict(),
    #     #           'optimizer_state_dict':optimizer.state_dict(),
    #     #           'scheduler_state_dict':scheduler.state_dict(),
    #     #           'val_score': val_score, 
    #     #           'epoch': ep}, os.path.join('/home/wenduoc/ORCA/clean/gene-orca/pretrained_embedders', configs['dataset'] + '_pretrained_model.pth'))
    #     torch.save(model.state_dict(), os.path.join('./pretrained_embedders', configs['dataset'] + '_pretrained_model.pth'))
    # if ep in [19,39,59,79,99]:
    #     torch.save(model.state_dict(), os.path.join('./pretrained_embedders', configs['dataset']+'_' + str(ep+1)+ '_pretrained_model.pth'))

# np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_losses.npy'), train_losses)
# np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_score.npy'), train_score)    
# np.save(os.path.join('/home/wenduoc/automation/automation/deepsea/', 'train_time.npy'), train_time) 

# test
print("\n------- Start Test --------")
test_scores = []

test_model = model
test_time_start = default_timer()
test_loss, test_score = evaluate(test_model, test_loader, loss,metric)
test_time_end = default_timer()
test_scores.append(test_score)

print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss,"\ttest score:", "%.4f" % test_score)




