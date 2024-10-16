import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from fast_pytorch_kmeans import KMeans
from torch import einsum
import torch.distributed as dist
from einops import rearrange


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="linear")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0, ks, ds):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # ks = 9 if in_channels == 16 else 3
        # pads = ks // 2

    
        if ks is None:
            if in_channels == 16:
                kernel_sizes = [9, 3] 
            else:
                kernel_sizes = [3, 3] 
        else:
            kernel_sizes = ks
        dilations = [1, 1] if ds is None else ds
        

        # pads = kernel_sizes[0] // 2

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_sizes[0],
                                     stride=1,
                                     dilation = dilations[0],
                                     padding=(kernel_sizes[0] - 1) * dilations[0] // 2,
                                    #  padding=pads
                                     )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_sizes[1],
                                     stride=1,
                                     dilation = dilations[1],
                                     padding=(kernel_sizes[1] - 1) * dilations[1]// 2,
                                    #  padding=1
                                     )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        # print('144',h.size())
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # print('149',h.size())
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        # print('153',x.size(),h.size())
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,l = q.shape
        q = q.reshape(b,c,l)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,l) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,l)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,l)

        h_ = self.proj_out(h_)

        return x+h_


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    """
    Encoder of VQ-GAN to map input batch of images to latent space.
    Dimension Transformations:
    3x256x256 --Conv2d--> 32x256x256
    for loop:
              --ResBlock--> 64x256x256 --DownBlock--> 64x128x128
              --ResBlock--> 128x128x128 --DownBlock--> 128x64x64
              --ResBlock--> 256x64x64  --DownBlock--> 256x32x32
              --ResBlock--> 512x32x32
    --ResBlock--> 512x32x32
    --NonLocalBlock--> 512x32x32
    --ResBlock--> 512x32x32
    --GroupNorm-->
    --Swish-->
    --Conv2d-> 256x32x32
    16 16 16 32 64 64
    """

    def __init__(self, in_channels=3, channels=[16, 16, 16, 32, 64, 64], attn_resolutions=[32], resolution=32, dropout=0, num_res_blocks=2, z_channels=128, f_channel=1000,num_class=36, ks=None, ds=None):
        super(Encoder, self).__init__()
        # channels = [c * 5 for c in channels]
        layers = [nn.Conv1d(in_channels, channels[0], 31, 1, padding=15)]   # Conv1d(1024, 16, kernel_size=(31,), stride=(1,), padding=(15,))
        count=0
        for i in range(len(channels) - 1): # i=5
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks): # 2
                layers.append(ResnetBlock(in_channels=in_channels, out_channels=out_channels, dropout=0.0, ks=ks[count:count+2] if ks is not None else None, ds=ds[count:count+2] if ds is not None else None))
                count+=2
                in_channels = out_channels
                if True:#resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i < 0:#len(channels)- 2:
                layers.append(Downsample(channels[i + 1], with_conv=True))
                resolution //= 2
        layers.append(ResnetBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=0.0, ks=ks[count:count+2] if ks is not None else None, ds=ds[count:count+2] if ds is not None else None))
        count+=2
        layers.append(AttnBlock(channels[-1]))
        layers.append(ResnetBlock(in_channels=channels[-1], out_channels=channels[-1], dropout=0.0, ks=ks[count:count+2] if ks is not None else None, ds=ds[count:count+2] if ds is not None else None))
        count+=2
        layers.append(Normalize(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv1d(channels[-1], z_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        self.final = nn.Linear(z_channels,num_class)#f_channel, num_class)

        self.dnaemb=nn.Embedding(5,1024)
        print('vq',count,ks)

    def forward(self, x, return_embeddings=False):
        b, c, l = x.shape 
        x = self.dnaemb(x.transpose(1,2).reshape(-1,c).long()) # x: (64,1,500) -->transpose (64,500,1) -->reshape(32000,1) --> dnaembed (32000, 1, 1024) 
        x = x.squeeze(1).reshape(b,l,-1).transpose(1,2)
        # print('243',x.shape)
        x= self.model(x)
        # print('245',x.shape)
        out = x.transpose(1,2)
        # print('247',out.shape)
        out = self.final(out).mean(1)
        # print('249',out.shape)
        if return_embeddings:
            return out, x
        else:
            return out


# class Encoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, double_z=True, **ignore_kwargs):
#         super().__init__()
#         self.ch = ch
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#
#         curr_res = resolution
#         in_ch_mult = (1,)+tuple(ch_mult)
#         self.down = nn.ModuleList()
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         2*z_channels if double_z else z_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
#
#
#     def forward(self, x):
#         #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
#
#         # timestep embedding
#         temb = None
#
#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))
#
#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h


class Decoder(nn.Module):
    def __init__(self, out_channels=3, channels=[16, 16, 16, 32, 64, 64], attn_resolutions=[32], resolution=32, dropout=0.0, num_res_blocks=2, z_channels=12, **kwargs):
        super(Decoder, self).__init__()
        ch_mult = channels[1:]
        num_resolutions = len(ch_mult) 
        block_in = ch_mult[num_resolutions - 1] 
        curr_res = resolution// 2 ** (num_resolutions - 1)

        layers = [nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1),
                  ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0),
                  AttnBlock(block_in),
                  ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0)
                  ]

        for i in reversed(range(num_resolutions)):
            # print(curr_res)
            block_out = ch_mult[i]
            for i_block in range(num_res_blocks+1):
                layers.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=0.0))
                block_in = block_out
                if True:#curr_res in attn_resolutions:
                    layers.append(AttnBlock(block_in))
            if i >= num_resolutions - 1:
                layers.append(Upsample(block_in, with_conv=True))
                curr_res = curr_res * 2

        layers.append(Normalize(block_in))
        layers.append(Swish())
        layers.append(nn.Conv1d(block_in, out_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
#                  attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
#                  resolution, z_channels, **ignorekwargs):
#         super().__init__()
#         self.temb_ch = 0
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         block_in = ch*ch_mult[self.num_resolutions-1]
#         curr_res = resolution // 2**(self.num_resolutions-1)
#         self.z_shape = (1,z_channels,curr_res,curr_res)
#
#         # z to block_in
#         self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
#
#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up) # prepend to get consistent order
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, z):
#         self.last_z_shape = z.shape
#
#         # timestep embedding
#         temb = None
#
#         # z to block_in
#         h = self.conv_in(z)
#
#         # middle
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](h, temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)
#
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h

def int_to_bits(x, bits=12, dtype=torch.uint8):
    assert not(x.is_floating_point() or x.is_complex()), "x isn't an integer type"
    if bits is None: bits = x.element_size() * 8
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(dtype=dtype)

from timm.models.layers import trunc_normal_
import torch.nn.init as init

def init_module(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)
# class Codebook(nn.Module):
#     """
#     Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
#     avoids costly matrix multiplications and allows for post-hoc remapping of indices.
#     """
#     def __init__(self, codebook_size=512, codebook_dim=12, beta=1, init_steps=1000, reservoir_size=2e5):
#         super().__init__()
#         self.codebook_size = codebook_size
#         self.codebook_dim = codebook_dim
#         self.beta = beta

#         self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
#         #self.embedding.weight.data = int_to_bits(torch.arange(4096), dtype=torch.float32) #.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
#         #print(self.embedding.weight.data, self.embedding.weight.data.shape)
#         self.embedding.weight.data.uniform_(-0.002,0.002)#-1.0 / self.codebook_size, 1.0 / self.codebook_size)
#         self.embedding2 = nn.Embedding(self.codebook_size, self.codebook_dim)
#         self.embedding2.weight.data.uniform_(-0.002,0.002)
#         self.q_start_collect, self.q_init, self.q_re_end, self.q_re_step = init_steps, init_steps * 5, init_steps * 30, init_steps // 2
#         self.q_counter =50000# 2000#50000
#         self.reservoir_size = int(reservoir_size)
#         self.reservoir = None
#         self.reservoir2 = None
#         seqlen = 1469
#         self.get_w = nn.Linear(self.codebook_dim*seqlen,16*self.codebook_dim)
#         self.get_h = nn.Linear(self.codebook_dim*seqlen,16*self.codebook_dim)
#         init_module(self.get_w)
#         init_module(self.get_h)

#     def forward(self, z):
#         #print(z[0])
#         #z = self.lnorm(z)
#         z = rearrange(z, 'b c h -> b h c').contiguous()
#         #z = z-z.mean(-1,keepdim=True)
#         #print(z.mean(-1).shape)
#         #z = z.contiguous()
#         batch_size = z.size(0)
#         #z_flattened = z.view(-1, self.codebook_dim)
#         z_h = z.reshape(z.shape[0],-1)#z.shape[1],-1)
#         z_w = z.reshape(z.shape[0],-1)#.permute(0,2,1,3).reshape(z.shape[0],z.shape[2],-1)
#         #z_h = (torch.sign(z.mean(1)) * torch.sqrt(torch.abs(z.mean(1))+1e-8)).view(-1,self.codebook_dim)
#         #z_w = (torch.sign(z.mean(2)) * torch.sqrt(torch.abs(z.mean(2))+1e-8)).view(-1,self.codebook_dim)# permute(0,2,1,3).reshape(z.shape[0],32,-1).view(-1, 32*self.codebook_dim)
#         # print("1",z.mean(1),z.mean(2),z_h,z_w)
#         # exit()
#         z_h = self.get_h(z_h)#.reshape(z.shape[0],z.shape[1],self.codebook_dim)
#         z_w = self.get_w(z_w)#.reshape(z.shape[0],z.shape[2],self.codebook_dim)
#         #print("2",z_h.shape,z_w)
#         z_h_ = z_h.view(z.shape[0],1,z.shape[2],z.shape[3])
#         z_h_per = z_h_.permute(0,3,1,2)
#         z_w_ = z_w.view(z.shape[0],z.shape[1],1,z.shape[3])
#         z_w_per = z_w_.permute(0,3,1,2)
#         znew =torch.matmul(z_w_per, z_h_per)#**2
#         znew=znew.permute(0,2,3,1)
#         l=torch.mean((znew.detach()-z)**2)+torch.mean((znew-z.detach())**2)
#         # print("3",znew,z)

#         if self.training:
#             self.q_counter += 1
#             # x_flat = x.permute(0, 2, 3, 1).reshape(-1, z.shape(1))
#             if self.q_counter > self.q_start_collect:
#                 z_flattened = z_h#.squeeze(1), z_w_.squeeze(2)],1)
#                 z_new = z_flattened.clone().detach().view(batch_size, -1, self.codebook_dim)
#                 z_new = z_new[:, torch.randperm(z_new.size(1))][:, :10].reshape(-1, self.codebook_dim)
#                 self.reservoir = z_new if self.reservoir is None else torch.cat([self.reservoir, z_new], dim=0)
#                 self.reservoir = self.reservoir[torch.randperm(self.reservoir.size(0))[:self.reservoir_size]].detach()

#                 z_flattened = z_w#.squeeze(1), z_w_.squeeze(2)],1)
#                 z_new = z_flattened.clone().detach().view(batch_size, -1, self.codebook_dim)
#                 z_new = z_new[:, torch.randperm(z_new.size(1))][:, :10].reshape(-1, self.codebook_dim)
#                 self.reservoir2 = z_new if self.reservoir2 is None else torch.cat([self.reservoir2, z_new], dim=0)
#                 self.reservoir2 = self.reservoir2[torch.randperm(self.reservoir2.size(0))[:self.reservoir_size]].detach()
#             if self.q_counter <self.q_init:
#                 z_q = rearrange(z, 'b h w c -> b c h w').contiguous()
#                 #z_q= torch.sigmoid(z_q/0.01)
#                 #z_q = z
#                 #z_h = z.mean(1).view(-1,self.codebook_dim)
#                 #z_w = z.mean(2).view(-1,self.codebook_dim)# permute(0,2,1,3).reshape(z.shape[0],32,-1).view(-1, 32*self.codebook_dim)
#                 #z_h = self.get_h(z_h).view(z.shape[0],1,z.shape[2],z.shape[3])#.reshape(z.shape[0],z.shape[1],self.codebook_dim)
#                 #z_w = self.get_w(z_w).view(z.shape[0],z.shape[1],1,z.shape[3])
#                 #z_h = z_h.permute(0,3,1,2)
#                 #z_w = z_w.permute(0,3,1,2)
#                 #znew =torch.matmul(z_w, z_h)#**2
#                 #znew=znew.permute(0,2,3,1)
#                 #l=torch.mean((znew-z)**2)
#                 return z_q, l,None,None,None#z_q.new_tensor(0), None,None  # z_q, loss, min_encoding_indices
#             else:
#                 # if self.q_counter < self.q_init + self.q_re_end:
#                 if self.q_init <= self.q_counter < self.q_re_end:
#                     if (self.q_counter - self.q_init) % self.q_re_step == 0 or self.q_counter == self.q_init + self.q_re_end - 1:
#                         kmeans = KMeans(n_clusters=self.codebook_size)
#                         # world_size = dist.get_world_size()
#                         # print("Updating codebook from reservoir.")
#                         # if world_size > 1:
#                         #     global_reservoir = [torch.zeros_like(self.reservoir) for _ in range(world_size)]
#                         #     dist.all_gather(global_reservoir, self.reservoir.clone())
#                         #     global_reservoir = torch.cat(global_reservoir, dim=0)
#                         # else:
#                         global_reservoir = self.reservoir
#                         kmeans.fit_predict(global_reservoir)  # reservoir is 20k encoded latents
#                         self.embedding.weight.data = kmeans.centroids.detach()
                        
#                         kmeans = KMeans(n_clusters=self.codebook_size)
#                         global_reservoir = self.reservoir2
#                         kmeans.fit_predict(global_reservoir)  # reservoir is 20k encoded latents
#                         self.embedding2.weight.data = kmeans.centroids.detach()
#                         print("kmean update")
#         #z_flattened = torch.sigmoid(z_flattened)
#         #print((z.mean(1)>0).float().mean(),(z.mean(2)>0).float().mean())
        
#         #z_sig1 = torch.sigmoid(z_h/0.01)
#         #z_sig2 = torch.sigmoid(z_w/0.01)
#         #z_sig = torch.sigmoid(z.mean((1,2),keepdim=True)/0.01)
#         #print(z_sig)
#         #z_h = torch.cat([torch.clamp(z_sig1,0,1)]*z.shape[1],1)
#         #z_w = torch.cat([torch.clamp(z_sig2,0,1)]*z.shape[2],2)
#         #z_h=z_w
#         #print(z_h.shape,(z_h[0]>0.5).float().mean(),(z_w[0]>0.5).float().mean())
#         #z_h = torch.sigmoid(z_h)
#         #z_w=torch.sigmoid(z_w)
#         #z_q = z_h * z_w
#         #print(z_h[0],z_w[0])
#         if False:
#             z_ = np.load("img.npy")
#             z_ = torch.from_numpy(z_).to(z.device)
#             print(z_.shape,z_)
#             z_h = z_[:,:16,:]
#             z_w = z_[:,16:,:]
#         if True:
#             #z_h = torch.sigmoid(z_h)
#             #z_w=torch.sigmoid(z_w)
#             z_h= z_h.reshape(-1, self.codebook_dim)
#             d = torch.sum(z_h ** 2, dim=1, keepdim=True) + \
#             torch.sum(self.embedding.weight**2, dim=1) - 2 * \
#             torch.einsum('bd,dn->bn', z_h, rearrange(self.embedding.weight, 'n d -> d n'))
#             min_encoding_indices1 = torch.argmin(d, dim=1)
#             z_h = self.embedding(min_encoding_indices1).view(z.shape[0],1,z.shape[2],z.shape[3])
#             #z_h = torch.cat([z_h]*z.shape[1],1)
#             #print(min_encoding_indices) 
#             z_w= z_w.reshape(-1, self.codebook_dim)
#             d = torch.sum(z_w ** 2, dim=1, keepdim=True) + \
#             torch.sum(self.embedding2.weight**2, dim=1) - 2 * \
#             torch.einsum('bd,dn->bn', z_w, rearrange(self.embedding2.weight, 'n d -> d n'))
#             min_encoding_indices = torch.argmin(d, dim=1)
#             z_w = self.embedding2(min_encoding_indices).view(z.shape[0],z.shape[1],1,z.shape[3])
#             min_encoding_indices1 = min_encoding_indices1.reshape(z.shape[0],-1)
#             min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1)
#             zidx = torch.cat([min_encoding_indices1,min_encoding_indices+512],1)
#             #z_w  = torch.cat([z_w]*z.shape[2],2)
#             #print(min_encoding_indices)
#         elif False:#se:
#             z_h = (z_h > 0).float()
#             z_w = (z_w >0).float()

#         cbl = torch.mean((z_w_.detach()-z_w)**2) + self.beta * torch.mean((z_w_ - z_w.detach()) ** 2) + torch.mean((z_h_.detach()-z_h)**2) + self.beta * torch.mean((z_h_-z_h.detach()) ** 2)
#         z_h = z_h_ + (z_h-z_h_).detach()
#         z_w = z_w_ + (z_w-z_w_).detach()
#         #print((z>0).float().sum(1,keepdim=True).bool().float(),(z>0).float().sum(1,keepdim=True).bool().float().sum(1).sum(1),(z>0).float().sum(2,keepdim=True).bool().float())
#         z_h = z_h.permute(0,3,1,2)
#         z_w = z_w.permute(0,3,1,2)
#         z_q =torch.matmul(z_w, z_h)#**2
#         z_q=z_q.permute(0,2,3,1)
#         #z_q = z_h * z_w
#         #print(z_q.shape)
#         #print((z_q[0]>0.5).float().mean())
#         if self.q_counter % 1000 == 0: 
#             #print(z[0,0,0,...],z_q[0,0,0,...])
#             #print((z_q[0]>0.5).float().mean())
#             pass
#             #print(z_q.sum(1).sum(1),z_q.mean(0))
#         #d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
#         #    torch.sum(self.embedding.weight**2, dim=1) - 2 * \
#         #    torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

#         #min_encoding_indices = torch.argmin(d, dim=1)
#         #z_q = self.embedding(min_encoding_indices).view(z.shape)
#         #z_q = nn.Sigmoid()(z)
#         #print(z_q,z_q.shape)#, min_encoding_indices)
#         #z =torch.sigmoid(z/0.01)
#         # compute loss for embeddingi
#         #loss=torch.mean((z_q - z.detach()) ** 2)+l#.detach()) ** 2)i
#         loss = l+cbl
#         #loss =0*torch.mean((z_q.detach()-z)**2) + 1 * torch.mean((z_q - z.detach()) ** 2)+l
#         #print(z[0,0,0,...],z_q[0,0,0,...],loss)
#         # preserve gradients
#         #z_q = z + (z_q - z).detach()
        

#         # reshape back to match original input shape
#         z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
#         #min_encoding_indices = min_encoding_indices.reshape(batch_size,-1)

#         return z_q, loss,z_h_per,z_w_per,zidx# min_encoding_indices

#     def get_codebook_entry(self, indices, shape):
#         # get quantized latent vectors
#         z_q = self.embedding(indices)

#         if shape is not None:
#             z_q = z_q.view(shape)
#             # reshape back to match original input shape
#             z_q = z_q.permute(0, 3, 1, 2).contiguous()

#         return z_q


if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()
    print(sum([p.numel() for p in enc.parameters()]))
    print(sum([p.numel() for p in dec.parameters()]))
    x = torch.randn(1, 3, 512, 512)
    res = enc(x)
    print(res.shape)
    res = dec(res)
    print(res.shape)

