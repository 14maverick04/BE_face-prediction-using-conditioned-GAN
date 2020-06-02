import torch
import torch.nn as nn
from models.spectral_norm import SpectralNorm
from scripts.utils import Concat_embed


class discriminator(nn.Module):
    def __init__(self, image_size):
        super(discriminator, self).__init__()
        self.image_size = image_size
        self.num_channels = 3
        self.latent_space = 128
        self.ndf = 64

        self.netD_1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            SpectralNorm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            SpectralNorm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.image_size == 64:
            self.netD_2 = nn.Conv2d(self.ndf * 8 + self.latent_space, 1, 4, 1, 0, bias=False)

        elif self.image_size == 128:
            self.netD_1 = nn.Sequential(
                self.netD_1,
                SpectralNorm(nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.netD_2 = nn.Conv2d(self.ndf * 16 + self.latent_space, 1, 4, 1, 0, bias=False)


    def forward(self, input_image, z_vector):

        x_intermediate = self.netD_1(input_image)

       
        dimensions = list(x_intermediate.shape)
        x = torch.cat([x_intermediate, z_vector.repeat(1,1,dimensions[2],dimensions[3])], 1)

       
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1), x_intermediate
