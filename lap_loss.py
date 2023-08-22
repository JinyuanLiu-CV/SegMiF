import torch
import sys
import math
import torch.nn as nn


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def smoothing(kernel_size, sigma, channels, device):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          torch.tensor(-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance),dtype=torch.float)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False,
                                padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    gaussian_filter.to(device)
    return gaussian_filter


def laplacian_pyramid(img, kernels, max_levels=3):
    pyr = []
    for level in range(max_levels):
        filtered = kernels[level](img)
        diff = img - filtered
        pyr.append(diff)
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernels =[]
        # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.gauss_kernels.append(smoothing(3, 2, channels, device))
        self.gauss_kernels.append(smoothing(5, 2, channels, device))
        self.gauss_kernels.append(smoothing(7, 2, channels, device))

    def forward(self, input, target):
        pyr_input = laplacian_pyramid(img=input, kernels=self.gauss_kernels, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernels=self.gauss_kernels, max_levels=self.max_levels)
        loss = 10. * sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input[:-1], pyr_target[:-1]))
        loss = loss + torch.nn.functional.l1_loss(pyr_input[-1], pyr_target[-1])
        return loss

class LapLoss2(torch.nn.Module):
    def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
        super(LapLoss2, self).__init__()
        self.max_levels = max_levels
        # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.gauss_kernel = smoothing(5, 2, channels, device)
        self.gauss_kernels =[]
        # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.gauss_kernels.append(smoothing(3, 2, channels, device))
        self.gauss_kernels.append(smoothing(5, 2, channels, device))
        self.gauss_kernels.append(smoothing(7, 2, channels, device))

    def forward(self, input, ir, vis):
        pyr_input = laplacian_pyramid(img=input, kernels=self.gauss_kernels, max_levels=self.max_levels)
        pyr_ir = laplacian_pyramid(img=ir, kernels=self.gauss_kernels, max_levels=self.max_levels)
        pyr_vis = laplacian_pyramid(img=vis, kernels=self.gauss_kernels, max_levels=self.max_levels)
        loss = 10. * sum(torch.nn.functional.l1_loss(a, torch.maximum(b,c)) for a, b,c in zip(pyr_input[:-1], pyr_ir[:-1],pyr_vis[:-1] ))
        loss = loss + torch.nn.functional.l1_loss(pyr_input[-1], torch.maximum(pyr_ir[-1],pyr_vis[-1]))
        return loss

# class LapLoss3(torch.nn.Module):
#     def __init__(self, max_levels=3, channels=1, device=torch.device('cuda')):
#         super(LapLoss3, self).__init__()
#         self.max_levels = max_levels
#         # self.gauss_kernel = gauss_kernel(channels=channels, device=device)
#         self.gauss_kernel = smoothing(5, 2, channels, device)
#
#     def forward(self, input, ir, vis):
#         pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
#         pyr_ir = laplacian_pyramid(img=ir, kernel=self.gauss_kernel, max_levels=self.max_levels)
#         pyr_vis = laplacian_pyramid(img=vis, kernel=self.gauss_kernel, max_levels=self.max_levels)
#
#         # [print(torch.nn.functional.l1_loss(a, b)) for a, b in zip(pyr_input, pyr_target)]
#         # sys.exit()
#         loss = 10. * sum(torch.nn.functional.l1_loss(a, torch.maximum(b,c)) for a, b,c in zip(pyr_input[:-1], pyr_ir[:-1],pyr_vis[:-1] ))
#         loss = loss + torch.nn.functional.l1_loss(pyr_input[-1], torch.maximum(pyr_ir[-1],pyr_vis[-1]))
#         return loss