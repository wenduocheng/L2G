import random
import math 
import numpy as np
import torch
from timeit import default_timer
from task_configs import get_data, get_config, get_metric
from attrdict import AttrDict
from sklearn import metrics
from utils import count_params, count_trainable_params
from embedder import get_tgt_model
from scipy import stats
from src.utils import binary_f1, mcc, pcc, pcc_deepstarr
import pandas as pd
import copy

from src.helper_scripts.genomic_benchmarks_utils import GenomicBenchmarkDataset, CharacterTokenizer, combine_datasets, NucleotideTransformerDataset 
from torch.utils.data import DataLoader 

torch.cuda.set_device(4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", DEVICE)

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0) 
torch.cuda.manual_seed_all(0)

model_name = 'ORCA' # DeepSEA, DeepSEA_FULL, DeepSEA_Original, DASH_FULL_0, DASH_FULL_1, ORCA
dataset = 'splice_sites_all' # DeepSEA_FULL, DeepSEA_NAS
exp_id = '5' # '16' # 26
if model_name == 'ORCA':
    trained_pth = './results/' + dataset + '/all_' + exp_id + '/0/state_dict.pt'

    trained = torch.load(trained_pth, map_location=DEVICE)
    args = np.load('./results/' + dataset + '/all_' + exp_id + '/0/hparams.npy', allow_pickle=True).item()


    print(args)
    args= AttrDict(args)
    # args.channels=[16, 16, 16, 32, 64, 64]
    args.run_dash = True # False
    args.backbone_select = True

if model_name == "ORCA":
    root = './src/datasets' 
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    args.embedder_epochs = 0
    args.pretrain_epochs = 0
    model, _ = get_tgt_model(args, root, sample_shape, num_classes, loss, False, False, None)
    model.load_state_dict(trained['network_state_dict'])
    print("param count:", count_params(model), count_trainable_params(model))


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
def pcc(output, target):
    target = target.cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    target = np.squeeze(target)
    output = np.squeeze(output)

    pearson_corr, _ = stats.pearsonr(target, output)
    
    return np.float64(pearson_corr)


def evaluate(args, model, loader, loss, metric):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(args.device), y.to(args.device) 

            out = model(x)

            outs.append(out) 
            ys.append(y) 
            n_data += x.shape[0]
        
        
        outs = torch.cat(outs, 0).to(args.device)
        ys = torch.cat(ys, 0).to(args.device)

        eval_loss += loss(outs, ys).item()

        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score

# for forward and reverse test-time augmentation
def evaluate2(args, model, loader, loader2, loss, metric):
    model.eval()

    eval_loss, eval_score = 0, 0
    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        # Assume loader and loader2 are aligned and have the same length
        for (data_forward, data_reverse) in zip(loader, loader2):
            x_forward, y_forward = data_forward
            x_forward, y_forward = x_forward.to(args.device), y_forward.to(args.device)
            out_forward = model(x_forward)

            x_reverse, _ = data_reverse  
            x_reverse = x_reverse.to(args.device)
            out_reverse = model(x_reverse)

            # Average the predictions of the forward and reverse complement sequence pairs
            out_avg = (out_forward + out_reverse) / 2

            outs.append(out_avg)
            ys.append(y_forward)
            n_data += x_forward.shape[0]

        outs = torch.cat(outs, 0).to(args.device)
        ys = torch.cat(ys, 0).to(args.device)

        eval_loss += loss(outs, ys).item()
        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score

# for forward and reverse and shift test-time augmentation
def evaluate3(args, model, loader1, loader2, loader3, loader4, loss, metric):
    model.eval()

    eval_loss, eval_score = 0, 0
    ys, outs, n_data = [], [], 0

    with torch.no_grad():
        # Assuming all loaders are aligned and have the same length
        for ((x_forward, y), (x_reverse, _), (x_shifted, _), (x_shifted_reverse, _)) in zip(loader1, loader2, loader3, loader4):
            # Move data to the appropriate device
            x_forward = x_forward.to(args.device)
            x_reverse = x_reverse.to(args.device)
            x_shifted = x_shifted.to(args.device)
            x_shifted_reverse = x_shifted_reverse.to(args.device)
            y = y.to(args.device)

            # Compute model outputs for each variant
            out_forward = model(x_forward)
            out_reverse = model(x_reverse)
            out_shifted = model(x_shifted)
            out_shifted_reverse = model(x_shifted_reverse)

            # Average the predictions from all four types of inputs
            out_avg = (out_forward + out_reverse + out_shifted + out_shifted_reverse) / 4

            outs.append(out_avg)
            ys.append(y)
            n_data += x_forward.shape[0]

        # Concatenate all outputs and targets for computing the loss and metric
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()
        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score


# for deepstarr, two tasks
def evaluate_deepstarr(args, model, loader, loss, metric):
    model.eval()
    
    eval_loss, eval_score1, eval_score2 = 0, 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(args.device), y.to(args.device) 

            out = model(x)

            outs.append(out) 
            ys.append(y) 
            n_data += x.shape[0]
        
        # Concatenate all batches
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()

        # Assume outs and ys are of shape (n, 2)
        # Compute PCC for each label
        eval_score1 += metric(outs[:, 0], ys[:, 0]).item()  # PCC for the first label
        eval_score2 += metric(outs[:, 1], ys[:, 1]).item()  # PCC for the second label

    return eval_loss, eval_score1, eval_score2

# for deepstarr, two tasks
def evaluate_deepstarr2(args, model, loader1, loader2, loss, metric):
    model.eval()
    
    eval_loss, eval_score1, eval_score2 = 0, 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for (data_forward, data_reverse) in zip(loader1, loader2):
            x_forward, y_forward = data_forward
            x_forward, y_forward = x_forward.to(args.device), y_forward.to(args.device)
            out_forward = model(x_forward)

            x_reverse, _ = data_reverse  
            x_reverse = x_reverse.to(args.device)
            out_reverse = model(x_reverse)

            # Average the predictions of the forward and reverse complement sequence pairs
            out_avg = (out_forward + out_reverse) / 2

            outs.append(out_avg)
            ys.append(y_forward)
            n_data += x_forward.shape[0]
        
        # Concatenate all batches
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()

        # Assume outs and ys are of shape (n, 2)
        # Compute PCC for each label
        eval_score1 += metric(outs[:, 0], ys[:, 0]).item()  # PCC for the first label
        eval_score2 += metric(outs[:, 1], ys[:, 1]).item()  # PCC for the second label

    return eval_loss, eval_score1, eval_score2

# Data & loss function
# _, _, test_loader, _, _, n_test, data_kwargs = get_data(root, dataset, args.batch_size, args.valid_split, quantize=args.quantize, rc_aug=args.rc_aug, shift_aug=args.shift_aug, one_hot=args.one_hot)


# two test loaders: one forward, one reverse
def load_nucleotide_transformer2(root, batch_size, one_hot = True, valid_split=-1, dataset_name = 'enhancers', quantize=False, rc_aug = True, shift_aug=True):
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
    ds_test2 = NucleotideTransformerDataset(
        max_length = max_length,
        dest_path = root+ '/nucleotide_transformer_downstream_tasks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=True,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
        
    
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader2 = DataLoader(ds_test2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return test_loader, test_loader2

def load_nucleotide_transformer3(root, batch_size, one_hot = True, valid_split=-1, dataset_name = 'enhancers', quantize=False, rc_aug = True, shift_aug=True):
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
    ds_test2 = NucleotideTransformerDataset(
        max_length = max_length,
        dest_path = root+ '/nucleotide_transformer_downstream_tasks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=True,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
        
    
    ds_test3 = NucleotideTransformerDataset(
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
    ds_test3.shift=3
    
    ds_test4 = NucleotideTransformerDataset(
        max_length = max_length,
        dest_path = root+ '/nucleotide_transformer_downstream_tasks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=True,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
    ds_test4.shift=3

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader2 = DataLoader(ds_test2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader3 = DataLoader(ds_test3, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader4 = DataLoader(ds_test4, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return test_loader, test_loader2, test_loader3, test_loader4

def load_genomic_benchmarks2(root, batch_size, one_hot = True, valid_split=-1, dataset_name = 'human_enhancers_cohn', quantize=False, rc_aug = True, shift_aug=True):
    if dataset_name == "dummy_mouse_enhancers_ensembl":
        max_length = 4707
    if dataset_name == "demo_coding_vs_intergenomic_seqs":
        max_length = 200
    if dataset_name == "demo_human_or_worm":
        max_length = 200
    if dataset_name == "human_enhancers_cohn":
        max_length = 500
    if dataset_name == "human_enhancers_ensembl":
        max_length = 573  
    if dataset_name == "human_ensembl_regulatory":
        max_length = 802  
    if dataset_name == "human_nontata_promoters":
        max_length = 251  
    if dataset_name == "human_ocr_ensembl":
        max_length = 593  
        
    use_padding = True
    add_eos = False  # add end of sentence token

    tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )
    

    ds_test = GenomicBenchmarkDataset(
        max_length = max_length,
        dest_path = root + '/genomic_benchmarks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=False,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
    
    ds_test2 = GenomicBenchmarkDataset(
        max_length = max_length,
        dest_path = root + '/genomic_benchmarks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=True,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
   

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader2 = DataLoader(ds_test2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return test_loader, test_loader2

def load_genomic_benchmarks3(root, batch_size, one_hot = True, valid_split=-1, dataset_name = 'human_enhancers_cohn', quantize=False, rc_aug = True, shift_aug=True):
    if dataset_name == "dummy_mouse_enhancers_ensembl":
        max_length = 4707
    if dataset_name == "demo_coding_vs_intergenomic_seqs":
        max_length = 200
    if dataset_name == "demo_human_or_worm":
        max_length = 200
    if dataset_name == "human_enhancers_cohn":
        max_length = 500
    if dataset_name == "human_enhancers_ensembl":
        max_length = 573  
    if dataset_name == "human_ensembl_regulatory":
        max_length = 802  
    if dataset_name == "human_nontata_promoters":
        max_length = 251  
    if dataset_name == "human_ocr_ensembl":
        max_length = 593  
        
    use_padding = True
    add_eos = False  # add end of sentence token

    tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )
    

    ds_test = GenomicBenchmarkDataset(
        max_length = max_length,
        dest_path = root + '/genomic_benchmarks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=False,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
    
    ds_test2 = GenomicBenchmarkDataset(
        max_length = max_length,
        dest_path = root + '/genomic_benchmarks',
        use_padding = use_padding,
        split = 'test',
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        rc_aug=True,
        add_eos=add_eos,
        one_hot=one_hot,
        quantize=quantize
        )
    
    ds_test3 = copy.deepcopy(ds_test)
    ds_test3.shift=3
    
    ds_test4 = copy.deepcopy(ds_test2)
    ds_test4.shift=3

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader2 = DataLoader(ds_test2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader3 = DataLoader(ds_test3, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader4 = DataLoader(ds_test4, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return test_loader, test_loader2, test_loader3, test_loader4

def load_deepstarr2(root, batch_size, one_hot = True, valid_split=-1,rc_aug=True, shift_aug=False):
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

    X_test, Y_test = prepare_input(data[data['set'] == "Test"])

    if one_hot:
        x_test = torch.from_numpy(X_test).transpose(-1, -2).float()
    else:
        x_test = torch.from_numpy(np.argmax(X_test, axis=2)).unsqueeze(-2).float()
    
    y_test = torch.stack( ( torch.from_numpy(np.array(Y_test[0])), torch.from_numpy(np.array(Y_test[1]))  ), dim=1).float()  

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_forward = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test[:20593,:,:], y_test[:20593,:]), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_reverse= torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test[20593:,:,:], y_test[20593:,:]), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader, test_loader_forward, test_loader_reverse

def reverse_complement(one_hot_encoded):
    """ Reverse complement for a batch of one-hot encoded DNA sequences.
    Args:
        one_hot_encoded (torch.Tensor): Tensor of shape (N, L, 4) where N is batch size,
                                        L is sequence length, and 4 represents one-hot encoding.

    Returns:
        torch.Tensor: Tensor of reverse complement sequences.
    """
    # Reverse the sequences along the sequence length dimension
    reversed_seqs = one_hot_encoded.flip(dims=[2])
    
    # Swap A and T, C and G
    # A [1, 0, 0, 0] <-> T [0, 0, 0, 1]
    # C [0, 1, 0, 0] <-> G [0, 0, 1, 0]
    # Original:  [A, C, G, T] = [0, 1, 2, 3]
    # Swap A <-> T and C <-> G: [T, G, C, A] = [3, 2, 1, 0]
    idx = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    reversed_complement = reversed_seqs.index_select(dim=1, index=idx)
    return reversed_complement
def reverse_complement_integer(encoded_dna):
    """ Compute the reverse complement of integer-encoded DNA sequences.
    Args:
        encoded_dna (torch.Tensor): Tensor of shape (N, L) where N is batch size,
                                    L is sequence length, and values are 0 (A), 1 (C), 2 (G), 3 (T).

    Returns:
        torch.Tensor: Tensor of the reverse complements.
    """
    # Reverse the sequence
    reversed_seqs = encoded_dna.flip(dims=[1]).long() 
    # Mapping: 0 <-> 3 and 1 <-> 2
    # Create a mapping tensor
    complement_map = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    # Apply the mapping
    reversed_complement = complement_map[reversed_seqs]
    return reversed_complement
def coin_flip():
    return random() > 0.5
def shift_up(seq,shift=3):
  res = copy.deepcopy(seq)
  res[:, :, shift:] = res[:, :, :-shift]
  res[:, :, :shift] = 4  # Fill with special token
  return res

def shift_down(seq,shift=3):
  res = copy.deepcopy(seq)
  res[:, :, :-shift] = res[:, :, shift:]
  res[:, :, -shift:] = 4  # Fill with special token
  return res
def load_deepsea2(root, batch_size, one_hot = True, valid_split=-1,rc_aug=False, shift_aug=False):
    filename = root + '/deepsea_filtered.npz'

    # if not os.path.isfile(filename):
    #     with open(filename, 'wb') as f:
    #         f.write(requests.get("https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz").content)

    data = np.load(filename)

    # y_train = torch.from_numpy(np.concatenate((data['y_train'], data['y_val']), axis=0)).float() 
    y_test = torch.from_numpy(data['y_test']).float()   # shape = (149400, 36)
    if one_hot:
        # x_train = torch.from_numpy(np.concatenate((data['x_train'], data['x_val']), axis=0)).transpose(-1, -2).float()  
        x_test = torch.from_numpy(data['x_test']).transpose(-1, -2).float()  # shape = (149400, 1000, 4)
    else:
        # x_train = torch.from_numpy(np.argmax(np.concatenate((data['x_train'], data['x_val']), axis=0), axis=2)).unsqueeze(-2).float()
        x_test = torch.from_numpy(np.argmax(data['x_test'], axis=2)).unsqueeze(-2).float()

  
    if not one_hot:
        x_test_rc = reverse_complement_integer(x_test.squeeze(-2)).unsqueeze(1)
        x_test_rc = x_test_rc.unsqueeze(-2)  # Reshape back if needed
    else:
        x_test_rc = reverse_complement(x_test)
            # y_train_rc = y_train.clone()  # Assuming the labels remain the same for the reverse complement
            # x_train = torch.cat([x_train, x_train_rc], dim=0)
            # y_train = torch.cat([y_train, y_train_rc], dim=0)
        # if shift_aug:
        #     print('sequence shift: ',3)
        #     x_train3 = shift_up(x_train,shift=3)
        #     y_train3 = copy.deepcopy(y_train) 
        #     x_train4 = shift_down(x_train,shift=3)
        #     y_train4 = copy.deepcopy(y_train)

        #     x_train = torch.concat((x_train,x_train3))
        #     y_train = torch.concat((y_train,y_train3))
        #     x_train = torch.concat((x_train,x_train4))
        #     y_train = torch.concat((y_train,y_train4))

    
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size = batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_rc, y_test), batch_size = batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return test_loader,test_loader2

print('load data')
if dataset in ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]: 
    metric = accuracy
    # test_loader, test_loader2 = load_genomic_benchmarks2(root, 64, False, -1, dataset, False, False, False)
    test_loader, test_loader2,test_loader3, test_loader4 = load_genomic_benchmarks3(root, 64, args.one_hot, -1, dataset, False, False, False)
elif dataset in ['enhancers', 'enhancers_types', 'H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac', 'promoter_all', 'promoter_no_tata', 'promoter_tata', 'splice_sites_acceptors', 'splice_sites_donors','splice_sites_all']: 
    metric = mcc
    test_loader, test_loader2 = load_nucleotide_transformer2(root, 64, False, -1, dataset, False, False, False)
    test_loader, test_loader2,test_loader3, test_loader4 = load_nucleotide_transformer3(root, 64, args.one_hot, -1, dataset, False, False, False)
elif dataset == 'deepstarr': 
    metric = pcc
    # _, _, test_loader, _, _, n_test, data_kwargs = get_data(root, dataset, args.batch_size, args.valid_split, quantize=args.quantize, rc_aug=args.rc_aug, shift_aug=args.shift_aug, one_hot=args.one_hot)
    test_loader, test_loader_forward, test_loader_reverse = load_deepstarr2(root, 64, args.one_hot, valid_split=-1,rc_aug=True, shift_aug=False)
elif dataset == 'DEEPSEA': 
    metric = auroc
    # _, _, test_loader, _, _, n_test, data_kwargs = get_data(root, dataset, args.batch_size, args.valid_split, quantize=args.quantize, rc_aug=args.rc_aug, shift_aug=args.shift_aug, one_hot=args.one_hot)
    test_loader, test_loader2 = load_deepsea2(root, 64, args.one_hot, valid_split=-1,rc_aug=True, shift_aug=False)





# Evaluate
print("\n------- Start Test --------")
if dataset == 'deepstarr': 
    test_time_start = default_timer()
    test_loss, test_score1, test_score2 = evaluate_deepstarr(args, model, test_loader, loss, metric)
    test_time_end = default_timer()
    print("[test]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\tdev pcc:", "%.4f" % test_score1, "\thk pcc:", "%.4f" % test_score2)

    test_time_start = default_timer()
    test_loss, test_score1, test_score2 = evaluate_deepstarr2(args, model,  test_loader_forward, test_loader_reverse, loss, metric)
    test_time_end = default_timer()
    print("[test-time augmentation]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\tdev pcc:", "%.4f" % test_score1, "\thk pcc:", "%.4f" % test_score2)
else:
    # mcc
    metric = mcc
    test_time_start = default_timer()
    test_loss, test_score = evaluate(args, model, test_loader, loss, metric)
    test_time_end = default_timer()
    print("[test]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

    test_time_start = default_timer()
    test_loss, test_score = evaluate2(args, model, test_loader, test_loader2, loss, metric)
    test_time_end = default_timer()
    print("[test-time augmentation]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)


    # test_time_start = default_timer()
    # test_loss, test_score = evaluate3(args, model, test_loader, test_loader2, test_loader3, test_loader4,loss, metric)
    # test_time_end = default_timer()
    # print("[test-time augmentation w shift]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

    # f1
    metric = binary_f1 if dataset != "splice_sites_all" else accuracy
    test_time_start = default_timer()
    test_loss, test_score = evaluate(args, model, test_loader, loss, metric)
    test_time_end = default_timer()
    print("[test]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

    test_time_start = default_timer()
    test_loss, test_score = evaluate2(args, model, test_loader, test_loader2, loss, metric)
    test_time_end = default_timer()
    print("[test-time augmentation]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)




    