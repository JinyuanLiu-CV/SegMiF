import torch
from torch import nn, Tensor


class Entropy(nn.Sequential):
    def __init__(self, patch_size):
        super(Entropy, self).__init__()

        self.psize = patch_size
        # number of patches per image

        # unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int) -> torch.Tensor:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            batch: int, size of the batch
        Returns:
            torch.Tensor:
        """
        epsilon = 1e-40
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        entropy = torch.sum(entropy)
        return entropy

    def forward(self, inputs: Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        self.width = inputs.shape[3]
        self.height = inputs.shape[2]
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        # gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]
        gray_images = inputs
        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)
        # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(device=inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size)

        return entropy
