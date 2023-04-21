import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# -----------------------------------------------
#            Weights Initialization
# -----------------------------------------------

def weights_init(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            # Weight
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # Bias
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


# -----------------------------------------------
#            Spectral Normalization
# -----------------------------------------------

def SN_Conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs), eps=1e-04)


def SN_ConvTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs), eps=1e-04)


def SN_Linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs), eps=1e-04)


# -----------------------------------------------
#                 Basic Modules
# -----------------------------------------------

class CondEmbedSequential(nn.Sequential):
    """
    This sequential module composes of different modules and calls them with the matching signatures.
    """
    def forward(self, x, cond):
        for layer in self:
            if isinstance(layer, AdaIN):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


# Class embedding network
class EmbeddingNetwork(nn.Module):
    def __init__(self, n_class: int, cond_size: int, n_layer: int = 4):
        super().__init__()
        layers = []
        for i in range(n_layer):
            if i == 0:
                layers.append(SN_Linear(n_class, cond_size))
            else:
                layers.append(SN_Linear(cond_size, cond_size))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, label: torch.tensor):
        return self.net(label)


# Adaptive instance normalization
class AdaIN(nn.Module):
    def __init__(self, num_features: int, cond_size: int):
        super().__init__()
        # Parameter-free normalization
        self.param_free_norm = nn.InstanceNorm2d(num_features, affine=False)

        # Affine transform using condition
        self.affine_cond = SN_Linear(cond_size, num_features * 2)

    def forward(self, x: torch.tensor, cond: torch.tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param cond: is the condition embeddings with shape `[batch_size, cond_size]`
        """
        # Normalization
        norm = self.param_free_norm(x)

        # Scaling and bias factors
        h = self.affine_cond(cond).view(x.shape[0], -1, 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        return (1 + gamma) * norm + beta


# Project a 1-dim vector into a 4 × 4 feature map
class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.init = SN_ConvTranspose2d(nz, channel, 4, 1, 0, bias=False)

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_size):
        super().__init__()
        self.main = CondEmbedSequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            AdaIN(in_channels, cond_size),
            nn.LeakyReLU(0.2, inplace=True),
            SN_Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        )

    def forward(self, feat, cond):
        return self.main(feat, cond)


class DownBlockComp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            SN_Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SN_Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            SN_Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, feat):
        return self.main(feat) + self.shortcut(feat)


# Skip-Layer Global Context (SGC) Block
class SGCBlock(nn.Module):
    def __init__(self, ch_in, ch_out, r=4):
        super().__init__()
        self.to_context = SN_Conv2d(ch_in, 1, 1, 1, 0, bias=False)
        self.transform = nn.Sequential(
            SN_Conv2d(ch_in, ch_out // r, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SN_Conv2d(ch_out // r, ch_out, 1, 1, 0, bias=False),
        )

    def forward(self, feat_small, feat_big):
        # Context modeling
        context = self.to_context(feat_small).flatten(2).softmax(dim=-1)
        out = torch.einsum('bin, bcn -> bci', context, feat_small.flatten(2)).unsqueeze(-1)

        # Transform and fusion
        return feat_big + self.transform(out)


# Upsample layer of simple decoder
class SimDec_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimDec_UpBlock, self).__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            SN_Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        )

    def forward(self, feat):
        return self.main(feat)


# Simple decoder (auxiliary branch of discriminator)
class SimpleDecoder(nn.Module):
    """ A Simple Decoder for self-supervised reconstruction in Discriminator"""
    def __init__(self, nfc_in, nc=3):
        super().__init__()

        # Feature channels at each spatial resolution
        nfc_multi = {16: 4, 32: 2, 64: 2, 128: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        # Decoder structure
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            SimDec_UpBlock(nfc_in, nfc[16]),
            SimDec_UpBlock(nfc[16], nfc[32]),
            SimDec_UpBlock(nfc[32], nfc[64]),
            SimDec_UpBlock(nfc[64], nfc[128]),
            SN_Conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, feat):
        return self.main(feat)
