import torch.nn as nn
from models.auxiliary_classifier import auxclassifier
from models.segan import Discriminator
from models.spectral_norm import SpectralNorm

class generator(nn.Module):
    def __init__(self, image_size, audio_samples):
        super(generator, self).__init__()

        self.audio_samples = audio_samples
        self.num_channels = 3
        self.latent_dim = 128
        self.ngf = 64
        self.image_size = image_size

        self.d_fmaps = [16, 32, 128, 256, 512, 1024]
        self.audio_embedding = Discriminator(1, self.d_fmaps, 15, nn.LeakyReLU(0.3), self.audio_samples)
        self.aux_classifier = auxclassifier()

        self.netG = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)),
            nn.Dropout(),
            nn.ReLU(True),
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)),
            nn.Dropout(),
            nn.ReLU(True),
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False)),
            nn.Dropout(),
            nn.ReLU(True),
            SpectralNorm(nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False)),
            nn.Tanh()
        )

        if self.image_size == 64:
            self.netG = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.latent_dim, self.ngf*8, 4, 1, 0, bias=False)),
            nn.Dropout(),
            nn.ReLU(True),
            self.netG
            )

        if self.image_size == 128:
            self.netG = nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(self.latent_dim, self.ngf*16, 4, 1, 0, bias=False)),
                nn.Dropout(),
                nn.ReLU(True),
                SpectralNorm(nn.ConvTranspose2d(self.ngf*16, self.ngf*8, 4, 2, 1, bias=False)),
                nn.Dropout(),
                nn.ReLU(True),
                self.netG
            )


    def forward(self, raw_wav):

        y, wav_embedding = self.audio_embedding(raw_wav.unsqueeze(1))

        softmax_scores = self.aux_classifier(y)

        z_vector = y.unsqueeze(2).unsqueeze(3)
        output = self.netG(z_vector)

        return output, z_vector, softmax_scores

