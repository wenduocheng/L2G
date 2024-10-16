import numpy as np
import torch
# test_score = np.load('/home/wenduoc/DASH/src/results_acc/human_enhancers_cohn/default/0/0/test_score.npy')

network_arch_params = np.load('/home/wenduoc/DASH/src/results_acc/human_enhancers_cohn/unet/unet/1/network_arch_params.npy')
network_hps = np.load('/home/wenduoc/DASH/src/results_acc/human_enhancers_cohn/unet/unet/1/network_hps.npy')
arch = torch.load('/home/wenduoc/DASH/src/results_acc/human_enhancers_cohn/unet/unet/1/arch.th')
network_weights = torch.load('/home/wenduoc/DASH/src/results_acc/human_enhancers_cohn/unet/unet/1/network_weights.pt')

# print(test_score)
print(network_arch_params)
print(network_hps)

print(arch)
print(network_weights.keys())

from networks.vq import Encoder

embed_dim=1024
ks=[5, 3, 7, 11, 3, 9, 3, 5, 5, 7, 5, 11, 3, 7, 9, 5, 7, 5, 5, 3, 9, 9, 3, 7, 5, 9, 9, 5, 3, 7, 5, 5, 3, 3, 3, 5, 7, 5, 11, 11, 5, 5, 11, 7, 9, 3, 3, 5, 11, 9, 3, 3, 3, 11, 11, 7, 9, 3, 3, 9, 5, 3, 7, 3, 5, 7, 3, 3, 7, 3] 
ds=[3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3]

model = Encoder(in_channels=embed_dim, f_channel=500, num_class=2, ks = ks, ds = ds)
model.load_state_dict(network_weights)