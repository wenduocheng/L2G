from random import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from pyfaidx import Fasta
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

# augmentation
string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# for data augmentation
class combine_datasets(torch.utils.data.Dataset):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)+len(self.d2)

    def __getitem__(self, i: int):
        if i < len(self.d1):
            return self.d1.__getitem__(i)
        else:
            return self.d2.__getitem__(i-len(self.d1))

class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {  #
            # "[CLS]": 0, 
            # "[SEP]": 1,
            # "[BOS]": 2,
            # "[MASK]": 3,
            # "[PAD]": 4,
            # "[RESERVED]": 5,
            # "[UNK]": 6,
            "[CLS]": 4, 
            "[SEP]": 4,
            "[BOS]": 4,
            "[MASK]": 4,
            "[PAD]": 4,
            "[RESERVED]": 4,
            "[UNK]": 4,
            **{ch: i  for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


class GenomicBenchmarkDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.

    Genomic Benchmarks Dataset, from:
    https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name='human_enhancers_cohn',
        d_output=2, # default binary classification
        dest_path=None, 
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        one_hot=False, #
        quantize=False, #
        truncate=None, #
        # addlab=0 #
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        # self.add_eos = add_eos
        self.d_output = d_output  # 
        self.rc_aug = rc_aug
        self.one_hot  = one_hot #
        self.quantize = quantize
        self.truncate = truncate
        self.shift = 0

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug: # and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        
        seq = seq["input_ids"]  # get input_ids

        # # need to handle eos here
        # if self.add_eos:
        #     # append list seems to be faster than append tensor
        #     seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        if self.one_hot: 
            seq = torch.LongTensor(seq)
            seq = torch.nn.functional.one_hot(seq, 5)
            seq = seq.permute(1,0) #
            seq = seq.type(torch.FloatTensor)
            if self.quantize:
                seq = seq.to(torch.bfloat16) # bfloat16
            if self.truncate:
                seq=seq[:self.truncate,...]
        else:
            # seq = torch.LongTensor(seq)
            seq = torch.FloatTensor(seq).unsqueeze(0)
            if self.truncate:
                seq = seq[:,:self.truncate]

        if self.shift>0:
            if coin_flip(): # shift down 
                seq[:,:-self.shift] = seq.clone()[:,self.shift:]
                if self.one_hot:
                    seq[:,-self.shift:] = 0
                    seq[4,-self.shift:] = 1 
                else:
                    seq[:,-self.shift:] = 4 # 4 is special token
            else: # shift up 
                seq[:,self.shift:] = seq.clone()[:,:-self.shift]
                if self.one_hot:
                    seq[:,:self.shift]=0
                    seq[4,:self.shift]=1
                else:
                    seq[:,:self.shift]=4

        # need to wrap in list
        target = torch.LongTensor([y]).squeeze()  #

        return seq, target



class NucleotideTransformerDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        # d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        # return_mask=False,
        one_hot=False, #
        quantize=False, #
        truncate=None, #
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        # self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        # self.return_mask = return_mask
        self.one_hot  = one_hot #
        self.quantize = quantize
        self.truncate = truncate
        self.shift = 0

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name 
        assert base_path.exists(), 'path to fasta file must exist'

        for file in (base_path.iterdir()):
            if (str(file).endswith('.fasta') or str(file).endswith('.fna') ) and split in str(file):  #
                self.seqs = Fasta(str(file), read_long_names=True)    

        self.label_mapper = {}
        for i, key in enumerate(self.seqs.keys()):
            self.label_mapper[i] = (key, int(key.rstrip()[-1]))


    def __len__(self):
        return len(self.seqs.keys())

    def __getitem__(self, idx):
        seq_id = self.label_mapper[idx][0]
        x = self.seqs[seq_id][:].seq # only one sequence
        y = self.label_mapper[idx][1] # 0 or 1 for binary classification

        # apply rc_aug here if using
        if self.rc_aug: # and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        # seq_ids = seq["input_ids"]  # get input_ids

        # seq_ids = torch.LongTensor(seq_ids)

        # # convert to tensor
        # seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # # need to wrap in list
        # target = torch.LongTensor([y])  # offset by 1, includes eos

        # return seq_ids, target


        seq = seq["input_ids"]
        if self.one_hot: 
            seq = torch.LongTensor(seq)
            seq = torch.nn.functional.one_hot(seq, 5)
            seq = seq.permute(1,0) #
            seq = seq.type(torch.FloatTensor)
            if self.quantize:
                seq = seq.to(torch.bfloat16) # bfloat16
            if self.truncate:
                seq=seq[:self.truncate,...]
        else:
            # seq = torch.LongTensor(seq)
            seq = torch.FloatTensor(seq).unsqueeze(0)
            if self.truncate:
                seq = seq[:,:self.truncate]

        if self.shift>0:
            if coin_flip(): # shift down 
                seq[:,:-self.shift] = seq.clone()[:,self.shift:]
                if self.one_hot:
                    seq[:,-self.shift:] = 0
                    seq[4,-self.shift:] = 1 
                else:
                    seq[:,-self.shift:] = 4 # 4 is special token
            else: # shift up 
                seq[:,self.shift:] = seq.clone()[:,:-self.shift]
                if self.one_hot:
                    seq[:,:self.shift]=0
                    seq[4,:self.shift]=1
                else:
                    seq[:,:self.shift]=4

        # need to wrap in list
        target = torch.LongTensor([y]).squeeze()  #

        return seq, target

# name maxlen classes samples metric

# enhancer 200   2  14968 MCC
# enhancer_types 200   3  14968 MCC
# H3 500   2  13468 MCC
# H3K4me1  500   2  28509 MCC
# H3K4me2  500   2  27614 MCC
# H3K4me3  500   2  33119 MCC
# H3K9ac   500   2  25003 MCC
# H3K14ac  500   2  29743 MCC
# H3K36me3 500   2  31392 MCC
# H3K79me3 500   2  25953 MCC
# H4 500   2  13140 MCC
# H4ac  500   2  30685 MCC
# promoter_all   300   2  53276 F1
# promoter_non_tata 300   2  47759 F1
# promoter_tata  300   2  5517  F1
# splice_sites_acceptor   600   2  19961 F1
# splice_sites_donor   600   2  19775 F1