import h5py
import mat73
import scipy.io 
import torch
import numpy as np

def convert_mat_to_hdf5(mat_file_path, hdf5_file_path):
    # Load the .mat file
    data = mat73.loadmat(mat_file_path)
    print('read mat file')
    # Create a new HDF5 file
    with h5py.File(hdf5_file_path, 'w') as hdf:
        # Iterate through the keys (e.g., 'trainxdata', 'trainydata') in the .mat file data
        for key in data.keys():
            # Store each dataset in the HDF5 file with compression
            hdf.create_dataset(name=key, data=data[key], compression='gzip')
    
    print(f"Data successfully stored in {hdf5_file_path}")

root = '/home/wenduoc/ORCA/clean/gene-orca/datasets'

mat_file_path = root + '/deepsea_full' + '/deepsea_full_train.mat'
hdf5_file_path = root + '/deepsea_full' + '/deepsea_full_train.hdf5'
convert_mat_to_hdf5(mat_file_path, hdf5_file_path)