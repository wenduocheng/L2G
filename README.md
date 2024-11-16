# L2G: Cross-Modal Fine-Tuning for Genomics
**L2G (Language-to-Genome)** is a cross-modal fine-tuning method tailored for genomics. By leveraging neural architecure search and modality alignment, it can repurpose pretrained language models, such as RoBERTa, for genomic applications. The L2G model is trained
in three stages. In stage 1, L2G performs a Neural Architecture Search to optimize the embedder architecture for a given task. In stage 2, the CNN embedder is pre-trained to minimize the modality gap between DNA embeddings and language embeddings. In stage 3, the entire model is fine-tuned on task-specific data in a supervised manner by minimizing the task-specific loss between the final predictions and the true labels.

<img src="https://github.com/user-attachments/assets/a28e4acf-dd07-490b-a1e2-8f06fd2b19df" alt="Overview" width="600"/>

---

## Installation
We recommend using a Conda environment to manage dependencies. Please refer to the [Anaconda webpage](https://anaconda.org/) for Conda installation instructions.

1. **Create and activate the environment:**
```bash
conda create --name L2G python=3.8.13
conda activate L2G
```

2. **Install dependencies:**
   Run the following script to install all necessary packages:
```bash
sh ./src/setup.sh
```


## Example Usage

To use L2G with a specific configuration, run:
```bash
python ./src/main.py --config ./src/configs/task.yaml
```

Using Weights & Biases is optional but recommended for experiment tracking. Please refer to the [Weights & Biases Quickstart Guide](https://docs.wandb.ai/quickstart) for instructions.

If using Weights & Biases, create `src/wandb_key.py` with the following content:
```python
key = 'your_wandb_api_key_here'
```

## Experiments

### NT Benchmark Experiment

1. **Data**
```bash
cd ./src/datasets
mkdir nucleotide_transformer_downstream_tasks
cd nucleotide_transformer_downstream_tasks
```
 Download the data from the [Nucleotide Transformer Downstream Tasks dataset](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks).

```bash
cd ../../..
```
 
2. **Run L2G** 

Example usage:
```bash
python ./src/main.py --config ./src/configs/nt_H3.yaml
```


### Genomic Benchmark Experiment

1. **Data**
```bash
cd ./src/datasets
mkdir genomic_benchmarks
cd genomic_benchmarks
```
Download the data from the [Genomic Benchmarks repository](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks?tab=readme-ov-file).

```bash
cd ../../..
```
 
2. **Run L2G**  

Example usage:
```bash
python ./src/main.py --config ./src/configs/human_enhancers_cohn.yaml
```

## Adding New Datasets

To integrate new datasets into the L2G framework, follow these steps:

1. **Add Data Loaders**: Implement new data loaders in `./src/data_loaders.py`.
2. **Define Data Processing**: Update the `get_data` function in `./src/task_configs.py`.
3. **Add Loss Functions and Metrics**: If your dataset requires specific loss functions or evaluation metrics, define them in `./src/utils.py` and add them to the `get_metric` function in `./src/task_configs.py`.
4. **Update Configuration**: Update the `get_config` function in `./src/task_configs.py` to incorporate the new dataset configuration.
5. **Add YAML Configuration**: Create a new configuration YAML file for the dataset and place it in `./src/configs`.
6. **Run the Configuration**: Once everything is set up, run the following command to test the new dataset:
```bash
python ./src/main.py --config ./src/configs/task.yaml
```

# Citation
If you find this work useful, please consider citing our paper:

> To be added
