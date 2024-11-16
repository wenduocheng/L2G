#!/bin/bash


echo "Setting up the L2G environment..."

echo "Creating the Conda environment..."
conda create --name L2G python=3.8.13 -y
conda activate L2G


echo "Installing PyTorch and related libraries..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 torchtext==0.16.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html


echo "Installing additional Python libraries..."
pip install attrdict
pip install yarl gdown datasets
pip install transformers==4.28.0
pip install timm==0.9.7
pip install pyfaidx mat73 einops Bio wandb
pip install genomic-benchmarks
pip install jupyter
pip install scipy scikit-learn tqdm ml-collections h5py requests timm


cd relax
pip install -e .
cd ..


pip install captum
pip install deeplift
pip install modisco
pip install pytabix kipoiseq
pip install -U "ray[data,train,tune,serve]"

echo "L2G environment setup is complete!"
