import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification, AutoTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers.models.roberta.modeling_roberta import RobertaLayer
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import copy
from datasets import load_dataset

import sys 
sys.path.append('./')

from src.task_configs import get_data, get_optimizer_scheduler, get_config, get_metric
from src.utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss, get_params_to_update 
from src.networks.wrn1d import ResNet1D, ResNet1D_v2, ResNet1D_v3
from src.networks.vq import Encoder, Encoder_v2
from src.networks.unet1d import UNet1D

 

 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset))
    return d


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False, args=None, root=None):
        super().__init__()

        self.dense = False
        self.output_raw = True   # during the embedder learning stage, output the raw embeddings; will be changed to false 
        self.weight = weight
        self.output_shape = output_shape
        self.use_lora=True if args.finetune_method=='lora' else False
        self.joint_optim = True if hasattr(args,'joint_optim') and args.joint_optim else False

        if isinstance(output_shape, tuple):
            self.dense = True

        if weight =='swin':
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k") if not from_scratch else SwinForImageClassification()
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity() 
        elif weight == 'hyenadna-small-32k-seqlen':
            from helper_scripts.huggingface import HyenaDNAPreTrainedModel 
            embed_dim = 256
            pretrained_model_name = weight 
            max_lengths = {
                'hyenadna-tiny-1k-seqlen': 1024,
                'hyenadna-small-32k-seqlen': 32768,
                'hyenadna-medium-160k-seqlen': 160000,
                'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
                'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
            }
            # max_length = max_lengths[pretrained_model_name]
            # use_padding = True
            # rc_aug = False  # reverse complement augmentation
            # add_eos = False  # add end of sentence token
            use_head = True # True
            n_classes = output_shape  # not used for embeddings only

            backbone_cfg = None
            if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
                # use the pretrained Huggingface wrapper instead
                print("Pretrained hyenaDNA!")
                self.model = HyenaDNAPreTrainedModel.from_pretrained(
                    './checkpoints',
                    pretrained_model_name,
                    download=False, #
                    config=backbone_cfg,
                    device=device,
                    use_head=use_head,
                    n_classes=n_classes,
                )
            if from_scratch:
                print("Train from Scratch!")
                from hyenaDNA import HyenaDNAModel 
                self.model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)
        elif weight == 'llama' or weight == 'llama2':
            embed_dim = 4096
            from transformers import LlamaConfig, LlamaModel#
            # self.model = LlamaModel.from_pretrained("/home/wenduoc/ORCA/src_backup/llama/huggingface/7B",return_dict=True, torch_dtype=torch.float16)
            model_name = "/home/wenduoc/ORCA/src_backup/llama/huggingface/7B"
            # tokenizer = LlamaTokenizer.from_pretrained(model_name )
            configuration = LlamaConfig.from_pretrained(model_name)
            configuration.num_hidden_layers = 1
            self.model = LlamaModel.from_pretrained(pretrained_model_name_or_path=model_name, config=configuration)
            print('Pretrained LLaMA!')
            if args.finetune_method == 'lora':
                from peft import get_peft_model, LoraConfig #
                self.use_lora=True
                peft_config = LoraConfig(task_type="FEATURE_EXTRACTION", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05)
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
        else: # roberta
            modelname = 'roberta-base' if weight[:7] == 'roberta' else 'bert-base-uncased'
            embed_dim = 768
            if weight == 'roberta-large':
                modelname = 'roberta-large'
                embed_dim = 1024
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                print('dropout:',drop_out)
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            if hasattr(args,'nlayers') and args.nlayers == 'one':
                configuration.num_hidden_layers=1
            if from_scratch:
                print("Train from Scratch!")
                self.model = AutoModel.from_config(configuration)
            else:
                print("Pretrained RoBERTa!")
                self.model = AutoModel.from_pretrained(modelname, config = configuration)
                # print(self.model.encoder.layer[0].attention.output.dense.weight)

                # print("Load DeepSEA weights")
                # trained = torch.load('/home/wenduoc/ORCA/src/results/DEEPSEA/all_415/0/state_dict.pt', map_location=device) 
                # # trained = torch.load('/home/wenduoc/ORCA/src/results/DEEPSEA_FULL/all_224/0/state_dict.pt', map_location=device) # 12 layers
                # # trained = torch.load('/home/wenduoc/ORCA/src_ablations/results/DEEPSEA_FULL/all_223/0/state_dict.pt', map_location=device) # 1 layer
                # print(trained['network_state_dict'].keys())
                # # keep only the roberta weights
                # filtered_keys = {k: trained['network_state_dict'][k] for k in trained['network_state_dict'].keys() if k.startswith('model.encoder')}
                # print(filtered_keys.keys())
                # # Modify the keys in filtered_keys to remove the "model." prefix
                # corrected_keys = {k.replace("model.", ""): v for k, v in filtered_keys.items()}
                # print(corrected_keys.keys())
                # # Load the corrected state dict
                # self.model.load_state_dict(corrected_keys, strict=False)
                # print(self.model.encoder.layer[0].attention.output.dense.weight)
            if args.finetune_method == 'lora':
                self.use_lora=True
                peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
            
           
        
        if use_embedder:
            if weight == 'hyenadna-small-32k-seqlen': 
                source = self.model.backbone.embeddings
            elif weight == 'llama' or weight == 'llama2':
                source = self.model.embed_tokens
            elif weight == 'roberta' or weight == 'roberta-large': # roberta base
                source = self.model.embeddings
            self.embedder = Embeddings1D(input_shape, config=self.model.config if weight != 'hyenadna-small-32k-seqlen' else None, embed_dim=embed_dim, target_seq_len=target_seq_len, dense=self.dense, args=args, output_shape=output_shape,root=root)
            embedder_init(source=source, target=self.embedder, train_embedder=train_epoch > 0, args=args)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = self.model.embeddings # nn.Identity()

        if weight == 'hyenadna-small-32k-seqlen':
            self.model.backbone.embeddings = self.embedder 
            # self.predictor = self.model.head
            # self.flatten = torch.nn.Flatten()
            # self.predictor = nn.Linear(in_features=256, out_features=output_shape)
        elif weight == 'llama' or weight == 'llama2':
            # self.model.embed_tokens = self.embedder
            self.model.embed_tokens = embedder_placeholder()
            self.pooler = adaptive_pooler()
            self.predictor = nn.Linear(in_features=4096, out_features=output_shape)
        elif weight == 'roberta' or weight == 'roberta-large':  # roberta
            self.model.embeddings = embedder_placeholder()
        
            if self.dense:
                self.model.pooler = nn.Identity()
                self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)
                
            else:
              
                self.model.pooler = adaptive_pooler()
                if weight == 'roberta-large':
                    self.predictor = nn.Linear(in_features=1024, out_features=output_shape)
                else: # roberta base
                
                    if use_embedder:
                        if args.embedder_type == 'resnet' or args.embedder_type == 'unet' or args.embedder_type == 'vq':
                            # if (args.dataset == 'DEEPSEA' or args.dataset == 'DEEPSEA_FULL'):
                            #     self.predictor = nn.Linear(in_features=384000, out_features=output_shape)
                            # else:
                    
                                self.predictor = nn.Linear(in_features=768, out_features=output_shape)
                        else:
                            if (args.dataset == 'DEEPSEA' or args.dataset == 'DEEPSEA_FULL') and args.embedder_type == 'resnet':
                                # self.predictor = nn.Linear(in_features=372480, out_features=output_shape)
                                self.predictor = nn.Linear(in_features=768*input_shape[-1]//self.embedder.stack_num, out_features=output_shape)
                            else:
                                self.predictor = nn.Linear(in_features=768*input_shape[-1]//self.embedder.stack_num, out_features=output_shape) 
                        # self.predictor = nn.Linear(in_features=768, out_features=output_shape)
                    else:
                        self.predictor = nn.Identity()
                conv_init(self.predictor)

                # print("Load pretrained linear layer!") 
                # trained_deepsea = torch.load("/home/wenduoc/ORCA/src_ablations/pretrained_embedders/deepsea_nas_two_linear/best_deepsea_nas_two_linear.pth")
                # self.predictor.weight.data.copy_(trained_deepsea['model_state_dict']['Linear2.weight'])
                # self.predictor.bias.data.copy_(trained_deepsea['model_state_dict']['Linear2.bias'])
         
        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
        
        if weight != 'hyenadna-small-32k-seqlen': #
            set_grad_state(self.model, False)
            set_grad_state(self.predictor, True) # False
        
        # self.predictor0 = nn.Linear(125,1)#50,1), unet; 235 dash
        # conv_init(self.predictor0)
        # set_grad_state(self.predictor0, True)
        # self.ln = nn.LayerNorm(embed_dim)
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, True)#False)


        # self.gamma = torch.nn.parameter.Parameter(torch.tensor(0.5))
        # print('intial gamma:',self.gamma)
      

    def forward(self, x):
        # if self.weight == 'roberta-large' or self.weight == 'roberta': # roberta
            if self.output_raw:
                return self.embedder(x) 

            # if self.joint_optim:
            #     x, _ = self.embedder(x)
            #     # print('266', x.size()) # ([64, 100, 1024])
            # else:
            #     x = self.embedder(x)
            x, fno = self.embedder(x)
            # print('259',x.shape,fno.shape)
            if self.dense:
                x = self.model(inputs_embeds=x)['last_hidden_state']
                x = self.predictor(x)
            else:
                # if self.use_lora:
                #     x = self.model.base_model(inputs_embeds=x)['pooler_output']
                #     x = self.predictor(x)
                # else:
                    # out = self.model(inputs_embeds=x)   # if using two layer predictor
                    # x = out['last_hidden_state']
                    # x = self.predictor0(x.permute(0,2,1)).squeeze(-1) 
                    # x = self.ln(x)

                    x = self.model(inputs_embeds=x)['pooler_output']
                    # x = self.model(inputs_embeds=x)['last_hidden_state'] #pooler_output']  # pooler_output: shape (batch_size, hidden_size); last_hidden_state: shape (batch_size, sequence_length, hidden_size)
                    # x = x.reshape(x.shape[0],-1)
                    # print('250',x.shape)
                    x = self.predictor(x)
            return x
            
            # gamma = torch.sigmoid(self.gamma) # make gamma between 0 and 1
            # # print('270',gamma)
            # out = gamma * fno + (1-gamma)*x
            # return out        
        
        
        # elif self.weight == 'hyenadna-small-32k-seqlen':
        #     if self.output_raw:
        #         x = self.model(x, return_embedding=True)
        #         return x
        #     else:
        #         x = self.model(x, return_embedding=False)
        #         return x
        #     # return self.predictor(x)
        # elif self.weight == 'llama':
        #     if self.output_raw:
        #         return self.embedder(x)
        #     x = self.embedder(x)
        #     # print('embedder:',x.size())
            
        #     if self.use_lora:
        #         x = self.model.base_model(inputs_embeds=x).last_hidden_state
        #     x = self.model(inputs_embeds=x).last_hidden_state
          
        #     x = self.pooler(x)
     
        #     x = self.predictor(x)

        #     return x



class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False, args=None, output_shape=None, root=None):
        super().__init__()
        self.dense = dense
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        print('stack num',self.stack_num)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)
        self.joint_optim = True if hasattr(args,'joint_optim') and args.joint_optim else False
   
        self.embedder_type = args.embedder_type if args is not None else None
        self.one_hot = args.one_hot if args is not None else False

        
        if not args.run_dash:
            if self.embedder_type == 'unet':
                in_channel=input_shape[-2]
                num_classes=output_shape
                try:
                    ks=args.ks 
                    ds=args.ds
                except: # use default kernel sizes and dilation sizes
                    ks=[3] * 18
                    ds=[1] * 18
                self.dash = UNet1D(n_channels=in_channel, num_classes=num_classes, ks=ks, ds=ds)

                self.projection = nn.Conv1d(64, embed_dim, kernel_size=self.stack_num, stride=self.stack_num) 
                downsample = False
                conv_init(self.projection)
            elif self.embedder_type == 'vq':
                self.projection = nn.Conv1d(128, embed_dim, kernel_size=self.stack_num, stride=self.stack_num) 
                downsample = False
                conv_init(self.projection)

                channels= args.channels if hasattr(args,'channels') else [16,32,64]
                try:
                    ks=args.ks 
                    ds=args.ds
                except: # use default kernel sizes and dilation sizes
                    ks=[3] * 18
                    ds=[1] * 18
                self.fno = Encoder_v2(input_shape[1],channels=channels,dropout=args.drop_out,f_channel=input_shape[-1],num_class=output_shape,ks=None,ds=None,downsample=downsample,seqlen=input_shape[-1]) 
   
                self.fno.apply(conv_init)
            elif self.embedder_type == 'resnet': # use default ResNet architecture
                in_channel=input_shape[-2]
                num_classes=output_shape
                # mid_channels=min(4 ** (num_classes // 10 + 1), 64)
                mid_channels = 128
                dropout=0
                try:
                    ks=args.ks 
                    ds=args.ds
                except: # use default kernel sizes and dilation sizes
                    # ks=[15, 19, 19, 7, 7, 7, 19, 19, 19]
                    # ds=[1, 15, 15, 1, 1, 1, 15, 15, 15]
                    ks=[3, 3, 3, 3, 3, 3, 3, 3, 3]
                    ds=[1, 1, 1, 1, 1, 1, 1, 1, 1]
                activation=None
                remain_shape=False
                self.dash = ResNet1D_v3(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = [15, 19, 19, 7, 7, 7, 19, 19, 19], ds = [1, 15, 15, 1, 1, 1, 15, 15, 15], activation=activation, remain_shape=remain_shape, input_shape=input_shape, embed_dim=embed_dim)
        else: # run_dash = True
            print('Backbone selection')
            # backbone zoo: resnet, unet
            backbone_select = args.backbone_select if hasattr(args,'backbone_select') else True
            if backbone_select: 
                # data
                train_loader, val_loader, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split, quantize=args.quantize if hasattr(args, 'quantize') else False, rc_aug=args.rc_aug if hasattr(args, 'rc_aug') else False, shift_aug=args.shift_aug if hasattr(args, 'shift_aug') else False, one_hot=args.one_hot if hasattr(args, 'one_hot') else True)
                # loss
                _, _, _, loss, _ = get_config(root, args)
                loss = loss.to(args.device)
                # metric
                metric, _ = get_metric(root, args.dataset)
                
                # resnet
                in_channel=input_shape[-2]
                num_classes=output_shape
                # mid_channels=min(4 ** (num_classes // 10 + 1), 64)
                mid_channels=128
                dropout=0
                activation=None
                remain_shape=False
                ks = [15, 19, 19, 7, 7, 7, 19, 19, 19]
                ds = [1, 15, 15, 1, 1, 1, 15, 15, 15]
                model = ResNet1D_v3(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, input_shape=input_shape, embed_dim=embed_dim).to(args.device)
                if args.quantize == True:
                    model=model.bfloat16() 
                # optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
                # scheduler
                base, accum = 0.2, 1
                sched = [30, 60, 90, 120, 160]
                def weight_sched_train(epoch):    
                    optim_factor = 0
                    for i in range(len(sched)):
                        if epoch > sched[len(sched) - 1 - i]:
                            optim_factor = len(sched) - i
                            break
                    return math.pow(base, optim_factor)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = weight_sched_train)

                train_loss_1 = train_one_epoch3(args, model, optimizer, train_loader, loss, n_train)
                val_loss_1, val_score_1 = evaluate3(args, model, val_loader, loss, metric)
                print("[backbone ", "resnet]", "\ttrain loss:", "%.4f" % train_loss_1, "\tval loss:", "%.4f" % val_loss_1, "\tval score:", "%.4f" % val_score_1)
                del model, optimizer, scheduler

                # unet
                # channels= args.channels if hasattr(args,'channels') else [16,32,64]
                # downsample = False
                # model = Encoder_v2(input_shape[1],channels=channels,dropout=args.drop_out,f_channel=input_shape[-1],num_class=output_shape,ks=None,ds=None,downsample=downsample,seqlen=input_shape[-1]).to(args.device)
                ks=None
                ds=None
                model = UNet1D(n_channels=in_channel, num_classes=num_classes, ks=ks, ds=ds).to(args.device)
                if args.quantize == True:
                    model=model.bfloat16() 
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = weight_sched_train)
                train_loss_2 = train_one_epoch3(args, model, optimizer, train_loader, loss, n_train)
                val_loss_2, val_score_2 = evaluate3(args, model, val_loader, loss, metric)
                print("[backbone ", "unet]", "\ttrain loss:", "%.4f" % train_loss_2, "\tval loss:", "%.4f" % val_loss_2, "\tval score:", "%.4f" % val_score_2)
                del train_loader, val_loader
                del model, optimizer, scheduler
                torch.cuda.empty_cache() 
                if val_score_1 > val_score_2:
                    self.embedder_type = 'resnet' 
                else:
                    self.embedder_type = 'unet'
                print('Backbone selection: ', self.embedder_type)
                
            self.embedder_type = 'resnet'
            # optimization
            if self.embedder_type == 'resnet':
                in_channel=input_shape[-2]
    
                num_classes=output_shape
                # mid_channels=min(4 ** (num_classes // 10 + 1), 64)
                mid_channels=128
                dropout=0
                if args.run_dash:
                    dash_result_path = f"./dash_results/results_acc/{args.dataset}/search_init/wrn/{args.seed}/dash_final_results.npy"
                    if not os.path.exists(dash_result_path):
                        print('Start to run DASH!')
                        subprocess.run(f"python -W ignore ./DASH/search_init.py --dataset {args.dataset} --arch wrn --experiment_id wrn --seed {args.seed} --valid_split 0 --save_dir '/home/wenduoc/ORCA/L2G/dash_results/' ", shell=True, check=True)
                        print('DASH Finish!')
                    else:
                        print('Found existing DASH results!')
                    # Load the results file
                    
                    dash_results = np.load(dash_result_path,allow_pickle=True).item()
                    print(type(dash_results))
                    print(dash_results)
                    ks = dash_results['ks']
                    ds = dash_results['ds']
                    args.ks =ks
                    args.ds =ds
                    dropout = dash_results['drop out']
                    args.embedder_optimizer.params.lr = dash_results['lr']
                    args.embedder_optimizer.params.weight_decay = dash_results['weight decay']
                    args.embedder_optimizer.params.momentum = dash_results['momentum']
                    print('DASH result:', dash_results['test best score'])
                    
                else:
                    ks = args.ks if hasattr(args,'ks') else [3, 3, 3, 3, 3, 3, 3, 3, 3]
                    ds = args.ds if hasattr(args,'ds') else [1, 1, 1, 1, 1, 1, 1, 1, 1]
                activation=None
                remain_shape=False
                self.dash = ResNet1D_v3(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, input_shape=input_shape, embed_dim=embed_dim)
                # self.dash = ResNet1D_v2(in_channels = in_channel, mid_channels=mid_channels, num_pred_classes=num_classes, dropout_rate=dropout, ks = ks, ds = ds, activation=activation, remain_shape=remain_shape, input_shape=input_shape, embed_dim=embed_dim,target_seq_len=target_seq_len)
            elif self.embedder_type == 'vq':
                self.projection = nn.Conv1d(128, embed_dim, kernel_size=self.stack_num, stride=self.stack_num) 
                downsample = False
                conv_init(self.projection)
         
                channels= args.channels if hasattr(args,'channels') else [16,32,64]
                self.fno = Encoder_v2(input_shape[1],channels=channels,dropout=args.drop_out,f_channel=input_shape[-1],num_class=output_shape,ks=None,ds=None,downsample=downsample,seqlen=input_shape[-1]) 
            

                self.fno.apply(conv_init)
            elif self.embedder_type == 'unet':
                dash_result_path = f"./dash_results/results_acc/{args.dataset}/search_init/unet/{args.seed}/dash_final_results.npy"
                if not os.path.exists(dash_result_path):
                    print('Start to run DASH!')
                    subprocess.run(f"python -W ignore ./DASH/search_init.py --dataset {args.dataset} --arch unet --experiment_id wrn --seed {args.seed} --valid_split 0 --save_dir '/home/wenduoc/ORCA/L2G/dash_results/' ", shell=True, check=True)
                    print('DASH Finish!')
                else:
                    print('Found existing DASH results!')
                # Load the results file
                
                dash_results = np.load(dash_result_path,allow_pickle=True).item()
                print(type(dash_results))
                print(dash_results)
                ks = dash_results['ks']
                ds = dash_results['ds']
                args.ks =ks
                args.ds =ds
                dropout = dash_results['drop out']
                args.embedder_optimizer.params.lr = dash_results['lr']
                args.embedder_optimizer.params.weight_decay = dash_results['weight decay']
                args.embedder_optimizer.params.momentum = dash_results['momentum']
                print('DASH result:', dash_results['test best score'])

                self.dash = UNet1D(n_channels=in_channel, num_classes=num_classes, ks=ks, ds=ds)

    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768 or self.embed_dim == 1024: # 
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i

    def forward(self, x=None, inputs_embeds=None, position_ids=None, *args, **kwargs): # 
        if x is None:
            x = inputs_embeds
        # b, c, l = x.shape # batch size, channel, length   e.g., human_enhancers_cohn (64,5,500) if one hot encoded

 
        if self.embedder_type == 'resnet': # dash resnet
            xfno,x = self.dash(x, return_embeddings=True)
        
        if self.embedder_type == 'unet':
            xfno, x = self.dash(x, return_embeddings=True) # x: (64,128,500)
            x = self.projection(x)
        
        elif self.embedder_type == 'vq':
            xfno, x = self.fno(x, return_embeddings=True) # x: (64,128,500)
            x = self.projection(x) 

        x = x.transpose(1, 2)
        x = self.norm(x)
        
        # roberta
        # position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        # self.ps = self.position_embeddings(position_ids)
        # x = x + self.ps
        
        return x, xfno





####################################################

def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None, opid=0):
    
    
    
    # if args.embedder_dataset == 'deepsea':
    #     # src_train_loader, _, _ = load_deepsea("/home/wenduoc/ORCA/src_backup/datasets", 16, one_hot = True, valid_split=-1)
    #     src_train_loader, _ = infer_labels(src_train_loader)
    #     src_feats, src_ys = src_train_loader.dataset.tensors[0][:2000,].reshape(2000, 4000), src_train_loader.dataset.tensors[1][:2000,].squeeze()
    #     src_feats = src_feats[:, :768]
    #     src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    # elif args.embedder_dataset == 'hg38':
    #     for batch in src_train_loader: #
    #         x = batch
    #         print(x.size())
    #         break
    #     src_feats = src_train_loader.dataset
    #     src_feats = src_feats.reshape(2000, 5120)[:, :256]
    #     src_ys = src_feats #
    # elif args.embedder_dataset == 'text_roberta_large':
    #     src_feats = src_train_loader.dataset.tensors[0] 
    #     src_ys = src_feats
    # else: # text
    #     src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    #     if args.weight == 'roberta-large' and args.embedder_dataset == 'text':
    #         padding = (0, 256)
    #         src_feats = F.pad(src_feats, padding, "constant", 0)
    #     src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    
    # generate source data (text)

    # def get_src_feats(weight='roberta'):
    #     trainset = load_dataset("conll2003",split='validation')
    #     trainset = trainset.select_columns(['tokens'])
    #     if args.weight == 'roberta-large':
    #         tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    #     elif args.weight == 'roberta':
    #         tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #     def preprocess_function(examples):
    #         examples["strs"] = ["".join(toks) for toks in examples["tokens"]]
    #         examples["input_ids"] = tokenizer(examples["strs"])['input_ids']
    #         del examples['tokens']
    #         del examples['strs']
    #         return examples
    #     trainset = trainset.map(preprocess_function, batched=True)
    #     data_collator = DataCollatorWithPadding(tokenizer)
    #     src_train_loader = DataLoader(trainset, batch_size=32,collate_fn=data_collator)
    #     src_model = wrapper1D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out, args=args)
    #     src_model = src_model.to(args.device).eval()
    #     src_model.output_raw = "src"
    #     src_feats = []
    #     src_ys = []
    #     for i, data in enumerate(src_train_loader):
    #         # print(data)
    #         # print(data.keys())
    #         #x_ = data['tokens']
    #         x_ = data['input_ids']
    #         # print(x_)
    #         x_ = x_.to(args.device)
    #         # y_ = x_
    #         out = src_model(x_)
    #         #print(out.shape)
    #         if len(out.shape) > 2:
    #             out = out.mean(1)
    #             # src_ys.append(y_.detach().cpu())
    #         src_feats.append(out.detach().cpu())
    #         #src_feats = torch.cat(src_feats, 0)
    #         if len(src_feats)>5000:
    #             break
    #     # src_ys = torch.cat(src_ys, 0).long()
    #     src_feats = torch.cat(src_feats, 0)
    #     # src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_feats)        
    #     del src_model, src_train_loader, trainset 
    #     torch.cuda.empty_cache() 
    #     src_ys = None
    #     return src_feats, src_ys

    def get_src_feats(weight='roberta'):

        conll2003 = load_dataset("conll2003")

        if weight == 'roberta-large':
            tokenizer = AutoTokenizer.from_pretrained("roberta-large",add_prefix_space=True,padding=True,truncation=True)
        elif weight == 'roberta': # base
            tokenizer = AutoTokenizer.from_pretrained("roberta-base",add_prefix_space=True,padding=True,truncation=True)

        def align_labels_with_tokens(labels, word_ids):
            # Initialize a list to store the adjusted labels
            new_labels = []
        
            # Initialize a variable to keep track of the current word's ID
            current_word = None
        
            # Iterate through each word ID in the word_ids list
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word/entity
                    current_word = word_id
        
                    # Assign -100 to labels for special tokens, else use the word's label
                    label = -100 if word_id is None else labels[word_id]
        
                    # Append the adjusted label to the new_labels list
                    new_labels.append(label)
                elif word_id is None:
                    # Handle special tokens by assigning them a label of -100
                    new_labels.append(-100)
                else:
                    # Token belongs to the same word/entity as the previous token
                    label = labels[word_id]
        
                    # If the label is in the form B-XXX, change it to I-XXX
                    if label % 2 == 1:
                        label += 1
        
                    # Append the adjusted label to the new_labels list
                    new_labels.append(label)
        
            # Return the list of adjusted labels
            return new_labels
        

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], truncation=True, is_split_into_words=True
            )
            all_labels = examples["ner_tags"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))
        
            tokenized_inputs["labels"] = new_labels
            return tokenized_inputs
        

        tokenized_datasets = conll2003.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=conll2003["train"].column_names,
        )

        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        src_train_loader = torch.utils.data.DataLoader(
            tokenized_datasets["validation"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=8,
        )


        src_model = wrapper1D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out, args=args)
        
        src_model = src_model.to(args.device).eval()
        src_model.output_raw = True 

        
        src_feats = []
        src_xs = []
        src_ys = []


        features = []
        labels = []
        for batch in src_train_loader:
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            with torch.no_grad():
                # outputs = trainer.model(**batch)
                hidden_states = src_model(batch['input_ids'])

                for i in range(hidden_states.size(0)):  # Iterate over batch size
                    length = (batch['attention_mask'][i] == 1).sum().item()
                    features.append(hidden_states[i, :length].cpu().numpy())
                    labels.append(batch['labels'][i, :length].cpu().numpy())

        max_len = max(f.shape[0] for f in features)
        padded_features = np.array([np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant') for f in features])
        padded_labels = np.array([np.pad(l, (0, max_len - l.shape[0]), mode='constant', constant_values=-100) for l in labels])

        print(padded_features.shape, padded_labels.shape)
        reshaped_features = padded_features.reshape(-1, padded_features.shape[-1])
        reshaped_labels = padded_labels.flatten()

        # Filter out -100 labels
        valid_indices = reshaped_labels != -100
        filtered_features = reshaped_features[valid_indices]
        filtered_labels = reshaped_labels[valid_indices]
        
        
        num_samples_per_label = 350
        selected_features = []
        selected_labels = []
        
        # Randomly select 3000 data points
        # Randomly select points for each label
        np.random.seed(42)  # For reproducibility
        for label in np.unique(filtered_labels):
            label_indices = np.where(filtered_labels == label)[0]
            selected_indices = np.random.choice(label_indices, num_samples_per_label, replace=False)
            selected_features.append(filtered_features[selected_indices])
            selected_labels.append(filtered_labels[selected_indices])
        selected_features = torch.from_numpy(np.concatenate(selected_features, axis=0))
        selected_labels = torch.from_numpy(np.concatenate(selected_labels, axis=0)).long()

        print(selected_features.shape, selected_labels.shape)

        # src_ys = torch.cat(src_ys, 0)
    
        # src_feats = torch.cat(src_feats, 0)
        
        del src_model, src_train_loader, conll2003
        torch.cuda.empty_cache() 

        return selected_features, selected_labels



    src_feats, src_ys = get_src_feats(args.weight)
    print("src feat shape", src_feats.shape)
  

    # tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    tgt_train_loader, _, test_loader, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split, quantize=args.quantize if hasattr(args, 'quantize') else False, rc_aug=args.rc_aug if hasattr(args, 'rc_aug') else False, shift_aug=args.shift_aug if hasattr(args, 'shift_aug') else False, one_hot=args.one_hot if hasattr(args, 'one_hot') else True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    joint_optim = True if hasattr(args,'joint_optim') and args.joint_optim else False
    

    for batch in tgt_train_loader: 
        x, y = batch
        print('x:',x.size())
        print('y:',y.size())
        break
    
  
    wrapper_func = wrapper1D 
   
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out, from_scratch=False, args=args, root= root)
    if hasattr(args, 'data_parallel') and args.data_parallel:
        tgt_model = nn.DataParallel(tgt_model) 
    if hasattr(args, 'quantize') and args.quantize:
        tgt_model.to(torch.bfloat16)
    tgt_model = tgt_model.to(args.device).train()
    print(tgt_model)

    # args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder-with-linear') # only update embedder
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder-pretraining') 
    
    
    tgt_model_optimizer.zero_grad()


    # if args.objective == 'otdd-exact':
    #     src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    #     score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    # # elif args.objective == 'otdd-gaussian':
    # #     score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    # # elif args.objective == 'l2':
    # #     score_func = partial(l2, src_train_dataset=src_train_dataset)
    # else:
    #     score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    
    if joint_optim and args.objective == 'MMD':
        score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
        total_losses, total_MMD_losses, total_second_losses, times, embedder_stats = [], [], [], [], []
        alpha = args.alpha if hasattr(args,'alpha') else 1
        beta = args.beta if hasattr(args,'beta') else 1
        # downstream task loss
        _, _, _, second_loss, args = get_config(root, args)
        second_loss = second_loss.to(args.device)

        for ep in range(args.embedder_epochs):   
            if ep<args.pretrain_epochs:
                alpha = 0
            else:
                alpha = args.alpha
            total_loss,total_MMD_loss,total_second_loss = 0,0,0    
            time_start = default_timer()
            
            feats, feats2, ys = [],[],[]
            datanum = 0
            for j, data in enumerate(tgt_train_loader): 
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                x = x.to(args.device) 
                # print('533',y.size())
                out, xfno = tgt_model(x)
                feats.append(out)
                feats2.append(xfno)
                ys.append(y.to(args.device))

                datanum += x.shape[0]
                
                if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: # accumulate samples until reach maxsamples
                    feats = torch.cat(feats, 0).mean(1) 
                    feats2 = torch.cat(feats2, 0)
                    ys = torch.cat(ys, 0)
                    loss1 = score_func(feats)
                    loss2 = second_loss(feats2, ys)
                    loss = alpha * loss1 + beta * loss2
                    loss.backward()

                    tgt_model_optimizer.step()
                    tgt_model_optimizer.zero_grad()

                    total_loss += loss.item()
                    total_MMD_loss += loss1.item()
                    total_second_loss += loss2.item()

                    feats, feats2, ys = [],[],[]
                    datanum = 0

            time_end = default_timer()  
            times.append(time_end - time_start) 

            total_losses.append(total_loss) #
            total_MMD_losses.append(total_MMD_loss) #
            total_second_losses.append(total_second_loss) #
            embedder_stats.append([total_losses[-1], total_MMD_losses[-1], total_second_losses[-1],times[-1]])
            print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\ttotal loss:", "%.4f" % total_losses[-1], "\tMMD loss:", "%.4f" % total_MMD_losses[-1], "\tCE loss:", "%.4f" % total_second_losses[-1]) 

            tgt_model_scheduler.step()
        
        # metric, compare_metrics = get_metric(root, args.dataset)
        # test_time_start = default_timer()
        # test_loss, test_score = evaluate_embedder(args, tgt_model, test_loader, second_loss, metric)
        # test_time_end = default_timer()
        # print("[test embedder with predictor]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss,"\ttest score:", "%.4f" % test_score)
    
    elif joint_optim and args.objective == 'otdd-exact':
        src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=2000)
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        if args.infer_label:
            tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
        else:
            num_classes_new = num_classes

        if args.objective=='otdd-exact' or args.objective=='otdd':
            print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 
            tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)

        total_losses, total_MMD_losses, total_second_losses, times, embedder_stats = [], [], [], [], []
        alpha = args.alpha if hasattr(args,'alpha') else 1
        beta = args.beta if hasattr(args,'beta') else 1
        # Classification loss
        _, _, _, second_loss, args = get_config(root, args)
        second_loss = second_loss.to(args.device)

        for ep in range(args.embedder_epochs):   
            total_loss,total_MMD_loss,total_second_loss = 0,0,0    
            time_start = default_timer()
            
            for i in np.random.permutation(num_classes_new):
                feats, feats2, ys = [],[],[]
                datanum = 0
                for j, data in enumerate(tgt_train_loaders[i]): # for otdd
                    
                    if transform is not None:
                        x, y, z = data
                    else:
                        if args.infer_label:
                            x, y, y_original = data 
                        else:
                            x, y = data 
                            y_original = y
                    x = x.to(args.device) 
                    y_original = y_original.to(args.device)
                    # print('533',y.size())
                    out, xfno = tgt_model(x)
                    feats.append(out)
                    feats2.append(xfno)
                    ys.append(y_original.to(args.device))

                    feats.append(out)

                    datanum += x.shape[0]
                    
                    if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: # accumulate samples until reach maxsamples
                        feats = torch.cat(feats, 0).mean(1)
                        feats2 = torch.cat(feats2, 0)
                        ys = torch.cat(ys, 0)
      
                        loss1 = tgt_class_weights[i] * score_func(feats)
                        loss2 = second_loss(feats2, ys) / len(tgt_train_loaders[i])
                        loss = alpha * loss1 + beta * loss2
                        loss.backward()

                        tgt_model_optimizer.step()
                        tgt_model_optimizer.zero_grad()

                        total_loss += loss.item()
                        total_MMD_loss += loss1.item()
                        total_second_loss += loss2.item()

                        feats, feats2, ys = [],[],[]
                        datanum = 0

            time_end = default_timer()  
            times.append(time_end - time_start) 

            total_losses.append(total_loss) #
            total_MMD_losses.append(total_MMD_loss) #
            total_second_losses.append(total_second_loss) #
            embedder_stats.append([total_losses[-1], total_MMD_losses[-1], total_second_losses[-1],times[-1]])
            print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\ttotal loss:", "%.4f" % total_losses[-1], "\totdd loss:", "%.4f" % total_MMD_losses[-1], "\tCE loss:", "%.4f" % total_second_losses[-1]) 

            tgt_model_scheduler.step()
    else: # not joint optimization
        total_losses, times, embedder_stats = [], [], []
        for ep in range(args.embedder_epochs):   
            total_loss = 0    
            time_start = default_timer()

            for i in np.random.permutation(num_classes_new):
                feats, ys = [],[]
                datanum = 0
                # for j, data in enumerate(tgt_train_loaders[i]): # for otdd
                for j, data in enumerate(tgt_train_loader): # for other losses such as MMD
                    if transform is not None:
                        x, y, z = data
                    else:
                        x, y = data 
                    x = x.to(args.device) 
                    ys.append(y.to(args.device))
                    out = tgt_model(x)
                    feats.append(out)
                    datanum += x.shape[0]  
                    if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: # accumulate samples until reach maxsamples
                        feats = torch.cat(feats, 0).mean(1) # can be improved?
                        ys = torch.cat(ys, 0)
                        
                        loss = score_func(feats)
                        loss.backward()

                        tgt_model_optimizer.step()
                        tgt_model_optimizer.zero_grad()
                        total_loss += loss.item()

                        feats, ys = [],[]
                        datanum = 0

                # feats = torch.cat(feats, 0).mean(1)
                # if feats.shape[0] > 1:
                #     loss = tgt_class_weights[i] * score_func(feats) #
                #     loss.backward()
                #     total_loss += loss.item()

            time_end = default_timer()  
            times.append(time_end - time_start) 

            total_losses.append(total_loss)
            embedder_stats.append([total_losses[-1], times[-1]])
            print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tMMD loss:", "%.4f" % total_losses[-1])

            tgt_model_optimizer.step()
            tgt_model_scheduler.step()
            tgt_model_optimizer.zero_grad()

    try:
        del tgt_train_loader #
    except:
        pass
    torch.cuda.empty_cache()
    
    if hasattr(args, 'data_parallel') and args.data_parallel:
        tgt_model.module.output_raw = False
    else:
        tgt_model.output_raw = False  

    return tgt_model, embedder_stats


# def infer_labels(loader, k = 10):
#     from sklearn.cluster import k_means, MiniBatchKMeans
    
#     if hasattr(loader.dataset, 'tensors'):
#         X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
#         try:
#             Z = loader.dataset.tensors[2].cpu()
#         except:
#             Z = None
#     else:
#         X, Y, Z = get_tensors(loader.dataset)

#     Y = Y.reshape(len(Y), -1)

#     if len(Y) <= 10000:
#         labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
#         Y = labeling_fun(Y).unsqueeze(1)
#     else:
#         kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
#         Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

#     if Z is None:
#         return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
#     return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k

def infer_labels(loader, k = 10): #  k=10
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):

        # X=tgt_model.embedder(X)
        # kmeans cluster of embeddings    instead of raw data
        # otdd   10 class  
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        Y_original = loader.dataset.tensors[1] # 
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Y_original), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k  # 
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


# def load_by_class(loader, num_classes):
#     train_set = loader.dataset
#     subsets = {}
#     # print(len(train_set.__getitem__(0)))
#     if len(train_set.__getitem__(0)) == 3:
#         try:
#             subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
#         except:
#             subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
#     else:
#         try:
#             subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
#         except:
#             subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
#     loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
#     class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
#     print("class weights")
#     for target, subset in subsets.items():
#         print(target, len(subset), len(train_set), len(subset)/len(train_set))

#     return loaders, class_weights

def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, y_original) in enumerate(train_set) if y == target]) for target in range(num_classes)} #
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)} 
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights

# def load_by_class_genomic_benchmarks(loader, num_classes):
#     train_set = loader.dataset
#     subsets = {}

#     subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)} #
    
#     DATASET = "human_enhancers_ensembl"
#     USE_PADDING = True
#     tokenizer = get_tokenizer(LetterTokenizer())
#     vocabulary = build_vocab(train_set, tokenizer, use_padding=USE_PADDING)
#     max_seq_len, nn_input_len = check_seq_lengths(dataset=train_set, use_padding=USE_PADDING)
#     collate = coll_factory(vocabulary, tokenizer, pad_to_length = nn_input_len, one_hot=True)

#     loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, collate_fn=collate) for target, subset in subsets.items()}
#     class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
#     print("class weights")
#     for target, subset in subsets.items():
#         print(target, len(subset), len(train_set), len(subset)/len(train_set))

#     return loaders, class_weights

def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs

def evaluate_embedder(args, model, loader, loss, metric):
    model.eval()
    
    eval_loss, eval_score = 0, 0

    ys, outs, n_eval, n_data = [], [], 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(args.device), y.to(args.device) 

            _,out = model(x)

            outs.append(out) 
            ys.append(y) 
            n_data += x.shape[0]
        
        
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()

        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score


def evaluate3(args,model, loader, loss, metric):
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
        
        
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)

        eval_loss += loss(outs, ys).item()

        eval_score += metric(outs, ys).item()

    return eval_loss, eval_score

def train_one_epoch3(args,model, optimizer, loader, loss, temp):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        x, y = data 
        
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)
        # print(x.get_device(),out.get_device(),y.get_device())
        l = loss(out, y)
        l.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
       
        optimizer.step()
        optimizer.zero_grad()
        

        train_loss += l.item()

        if i >= temp - 1:
            break

    return train_loss / temp