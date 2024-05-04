# pip install tokenizers==0.13.3
# pip install datasets
# pip install git+https://github.com/huggingface/transformers

from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

import numpy as np

# Reference: https://github.com/4AI/LS-LLaMA/blob/main/modeling_llama.py

torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_name = 'flan-t5-large' # llama, llama2, pythia-1b, roberta-large, flan-t5-small, flan-t5-base, flan-t5-large

# load the dataset conll2003
ds = load_dataset("conll2003")
label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys()) # ds["train"].features[f"ner_tags"].feature.names

if model_name == 'pythia-1b':
    from transformers import GPTNeoXForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step143000",
        cache_dir="./pythia-1b-deduped/step3000",
    )
    tokenizer.pad_token = tokenizer.eos_token 
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b-deduped",
        revision="step143000",
        cache_dir="./pythia-1b-deduped/step143000",
    )

elif model_name  == 'roberta-large':
    tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
    print('Tokenizer is fast? ',tokenizer.is_fast)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        # output_hidden_states=True
    )

elif model_name  == 'llama' or model_name == 'llama2':
    from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, LlamaTokenizerFast

    if model_name == 'llama':
        model_path = "/home/wenduoc/ORCA/src_backup/llama/huggingface/7B"  
    elif model_name == 'llama2':
        model_path = "/work/magroup/4DN/llama/llama-2/huggingface/7B" 

    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # 
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # initialize the model
    configuration = LlamaConfig.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(pretrained_model_name_or_path=model_path, config=configuration, torch_dtype=torch.float16)
    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

elif model_name =='flan-t5-small' or model_name == 'flan-t5-base' or model_name == 'flan-t5-large':
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")


model = model.to(device)
print(model)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
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

tokenized_ds = ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds["train"].column_names,
)

train_dataloader = torch.utils.data.DataLoader(
    tokenized_ds["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
)

for batch in train_dataloader:
  print(batch.keys())
  break


# Using embeddings as source features
embeddings = []
for i, batch in enumerate(train_dataloader):
    batch = batch.to(device)
    if model_name == 'llama' or model_name == 'llama2':
        embedding_outputs = model.base_model.embed_tokens(batch['input_ids'])
    elif model_name == 'pythia-1b':
        embedding_outputs = model.gpt_neox.embed_in(batch['input_ids'])
        # print('141')
        # print(embedding_outputs)
    elif model_name == 'roberta-large':
        embedding_outputs = model.roberta.embeddings(batch['input_ids'])
    elif model_name =='flan-t5-small' or model_name == 'flan-t5-base' or model_name == 'flan-t5-large':
        embedding_outputs = model.encoder.embed_tokens(batch['input_ids'])
    embeddings.append(embedding_outputs)
    # ys.append()
    
    if i > 500:
        break
print('len of embeddings', len(embeddings))

for i in range(len(embeddings)):
    embeddings[i] = embeddings[i].mean(1)
print(embeddings[1].size())

# Concatenate all embeddings
text_xs = torch.cat(embeddings, dim=0)
print('text_xs shape',text_xs.shape)

# save
np.save(f'./datasets/text_xs_{model_name}_embeddings.npy', text_xs.cpu().detach().float().numpy())