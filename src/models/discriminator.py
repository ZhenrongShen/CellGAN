import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .modules import DownBlockComp, SimpleDecoder, SN_Conv2d, SN_Linear
from .diffaug import DiffAugment


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=32, n_class=5, recon_mode='l1'):
        super().__init__()
        
        # Feature channels at each spatial resolution
        nfc_multi = {8: 16, 16: 8, 32: 8, 64: 4, 128: 2, 256: 1}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        # Network structure
        self.down_256 = SN_Conv2d(nc, nfc[256], 3, 1, 1, bias=False)
        self.down_128 = DownBlockComp(nfc[256], nfc[128])
        self.down_64 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_16 = DownBlockComp(nfc[32], nfc[16])
        self.down_8 = DownBlockComp(nfc[16], nfc[8])
        self.to_logits = SN_Conv2d(nfc[8], 1, 1, 1, 0, bias=False)

        # Projection head
        self.to_cls_embed = SN_Linear(n_class, nfc[8], bias=False)

        # Self-supervised reconstruction
        self.decoder_part = SimpleDecoder(nfc[16], nc)
        self.decoder_overall = SimpleDecoder(nfc[8], nc)
        if recon_mode == 'l1':
            self.recon_loss = nn.L1Loss()
        elif recon_mode == 'mse':
            self.recon_loss = nn.MSELoss()
        else:
            self.recon_loss = None

    def forward(self, img, label, is_real=True, policy=None):
        # Differentiable data augmentation
        img = DiffAugment(img, policy=policy)

        # Main branch
        feat_256 = self.down_256(img)
        feat_128 = self.down_128(feat_256)
        feat_64 = self.down_64(feat_128)
        feat_32 = self.down_32(feat_64)
        feat_16 = self.down_16(feat_32)
        feat_8 = self.down_8(feat_16)
        feat_out = feat_8
        logits = self.to_logits(feat_out)

        # Projection head
        cls_embed = self.to_cls_embed(label).view(label.shape[0], -1, 1, 1)
        logits += torch.sum(cls_embed * feat_out, dim=1, keepdim=True)

        # Self-supervision
        if is_real and self.recon_loss is not None:
            # Crop real image and 16×16 featur map on same portion
            img_part, feat_16_part = random_crop(img, feat_16)

            # Resize overall real image (keep same with cropped image size)
            img_overall = F.interpolate(img, size=128)

            # Self-supervised reconstruction
            recons_part = self.decoder_part(feat_16_part)
            recons_overall = self.decoder_overall(feat_8)

            # Calculate auxiliary reconstruction loss
            aux_loss_16 = self.recon_loss(recons_part, img_part)
            aux_loss_8 = self.recon_loss(recons_overall, img_overall)
            aux_loss = aux_loss_16 + aux_loss_8
        else:
            aux_loss = 0

        return {'logits': logits, 'aux_loss': aux_loss}


# Crop the same part from image and its feature map
def random_crop(image, feat, img_size=128):
    # Crop part of image
    img_h, img_w = image.shape[2:]
    img_ch = random.randint(0, img_h - img_size - 1)
    img_cw = random.randint(0, img_w - img_size - 1)
    img_part = image[:, :, img_ch:img_ch + img_size, img_cw:img_cw + img_size]

    # Crop part of feature map on the same portion
    feat_h, feat_w = feat.shape[2:]
    feat_part_h = int(img_size * feat_h / img_h)
    feat_part_w = int(img_size * feat_w / img_w)
    feat_ch = int(img_ch * feat_h / img_h)
    feat_cw = int(img_cw * feat_w / img_w)
    feat_part = feat[:, :, feat_ch:feat_ch + feat_part_h, feat_cw:feat_cw + feat_part_w]

    return img_part, feat_part
