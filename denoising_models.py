import os
import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F
from models import SubPixelConvolutionalBlock, ConvolutionalBlock

class DnCNN(nn.Module):

    def __init__(self, D, C=64, output_channels=3):
        super(DnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, output_channels, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # # initialize the weights of the Batch normalization layers
        # for i in range(D):
        #     nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.leaky_relu(self.conv[0](x), 0.2)
        for i in range(D):
            h = F.leaky_relu(self.bn[i](self.conv[i+1](h)), 0.2)
        y = self.conv[D+1](h) + x
        return y

class SRDnCNN(nn.Module):
    def __init__(self, n_blocks=16, n_channels=64, scaling_factor=4, small_kernel_size=3, large_kernel_size=9):
        super(SRDnCNN, self).__init__()

        # Scaling factor must be 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        # DnCNN for denoising:
        self.denoising_backbone = DnCNN(n_blocks, n_channels, n_channels)

        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # The last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        denoised_lr_imgs = self.denoising_backbone(lr_imgs)
        upscaled_imgs = self.subpixel_convolutional_blocks(denoised_lr_imgs)
        sr_imgs = self.conv_block3(upscaled_imgs)

        return sr_imgs

# class UDnCNN(NNRegressor):

#     def __init__(self, D, C=64):
#         super(UDnCNN, self).__init__()
#         self.D = D

#         # convolution layers
#         self.conv = nn.ModuleList()
#         self.conv.append(nn.Conv2d(3, C, 3, padding=1))
#         self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
#         self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
#         # apply He's initialization
#         for i in range(len(self.conv[:-1])):
#             nn.init.kaiming_normal_(
#                 self.conv[i].weight.data, nonlinearity='relu')

#         # batch normalization
#         self.bn = nn.ModuleList()
#         self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
#         # initialize the weights of the Batch normalization layers
#         for i in range(D):
#             nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

#     def forward(self, x):
#         D = self.D
#         h = F.relu(self.conv[0](x))
#         h_buff = []
#         idx_buff = []
#         shape_buff = []
#         for i in range(D//2-1):
#             shape_buff.append(h.shape)
#             h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))),
#                                   kernel_size=(2, 2), return_indices=True)
#             h_buff.append(h)
#             idx_buff.append(idx)
#         for i in range(D//2-1, D//2+1):
#             h = F.relu(self.bn[i](self.conv[i+1](h)))
#         for i in range(D//2+1, D):
#             j = i - (D // 2 + 1) + 1
#             h = F.max_unpool2d(F.relu(self.bn[i](self.conv[i+1]((h+h_buff[-j])/np.sqrt(2)))),
#                                idx_buff[-j], kernel_size=(2, 2), output_size=shape_buff[-j])
#         y = self.conv[D+1](h) + x
#         return y


# class DUDnCNN(NNRegressor):

#     def __init__(self, D, C=64):
#         super(DUDnCNN, self).__init__()
#         self.D = D

#         # compute k(max_pool) and l(max_unpool)
#         k = [0]
#         k.extend([i for i in range(D//2)])
#         k.extend([k[-1] for _ in range(D//2, D+1)])
#         l = [0 for _ in range(D//2+1)]
#         l.extend([i for i in range(D+1-(D//2+1))])
#         l.append(l[-1])

#         # holes and dilations for convolution layers
#         holes = [2**(kl[0]-kl[1])-1 for kl in zip(k, l)]
#         dilations = [i+1 for i in holes]

#         # convolution layers
#         self.conv = nn.ModuleList()
#         self.conv.append(
#             nn.Conv2d(3, C, 3, padding=dilations[0], dilation=dilations[0]))
#         self.conv.extend([nn.Conv2d(C, C, 3, padding=dilations[i+1],
#                                     dilation=dilations[i+1]) for i in range(D)])
#         self.conv.append(
#             nn.Conv2d(C, 3, 3, padding=dilations[-1], dilation=dilations[-1]))
#         # apply He's initialization
#         for i in range(len(self.conv[:-1])):
#             nn.init.kaiming_normal_(
#                 self.conv[i].weight.data, nonlinearity='relu')

#         # batch normalization
#         self.bn = nn.ModuleList()
#         self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
#         # initialize the weights of the Batch normalization layers
#         for i in range(D):
#             nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

#     def forward(self, x):
#         D = self.D
#         h = F.relu(self.conv[0](x))
#         h_buff = []

#         for i in range(D//2 - 1):
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1](h)
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))
#             h_buff.append(h)

#         for i in range(D//2 - 1, D//2 + 1):
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1](h)
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))

#         for i in range(D//2 + 1, D):
#             j = i - (D//2 + 1) + 1
#             torch.backends.cudnn.benchmark = True
#             h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
#             torch.backends.cudnn.benchmark = False
#             h = F.relu(self.bn[i](h))

#         y = self.conv[D+1](h) + x
#         return y