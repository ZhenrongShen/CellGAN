import os
import random
import logging
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from .models.modules import weights_init
from .models.generator import Generator
from .models.discriminator import Discriminator


# ----------------------------------------
#             Reproducibility
# ----------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------
#                 Logging
# ----------------------------------------

# open the log file
def open_log(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_path, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_path, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_path, '{}.log'.format(log_name)))


# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        filename=logFilename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# ----------------------------------------
#    Load & Save Networks / Optimizers
# ----------------------------------------

def load_model(process_net, pretrained_file):
    pretrained_dict = torch.load(pretrained_file)['model']
    process_net.load_state_dict(pretrained_dict)

    return process_net


def save_model(net, optimizer, net_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = os.path.join(save_path, net_name)
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_name)


# ----------------------------------------
#            Create Optimizers
# ----------------------------------------

def load_optimizer(process_optim, pretrained_file):
    pretrained_dict = torch.load(pretrained_file)['optimizer']
    process_optim.load_state_dict(pretrained_dict)

    return process_optim


def create_optimizer(config, model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(config.BETA1, config.BETA2))
    if config.PRETRAIN:
        # 1. Generator
        if isinstance(model, Generator):
            load_name = os.path.join(config.CKPT_PATH, 'gen_{:d}.pth'.format(config.PRETRAIN_GEN))
        # 2. Discriminator
        elif isinstance(model, Discriminator):
            load_name = os.path.join(config.CKPT_PATH, 'dis_{:d}.pth'.format(config.PRETRAIN_DIS))
        else:
            raise ValueError("Unknown Model.")

        optimizer = load_optimizer(optimizer, load_name)

    return optimizer


# ----------------------------------------
#             Create Networks
# ----------------------------------------

def create_generator(config, model_path=None):
    generator = Generator(
        nc      = config.OUT_CHANNELS,
        nz      = config.LATENT_DIMS,
        ngf     = config.GEN_FEAT_NUM,
        n_class = config.CLASS_NUM,
        d_cond  = config.COND_SIZE,
        n_layer = config.MAP_LAYERS,
    )
    logging.info('Generator is created!')

    # Initialize the networks
    if model_path is not None:
        generator = load_model(generator, model_path)
        print(f'Load pre-trained generator from {model_path}')

    else:
        if config.MODE == 'train' and config.PRETRAIN_GEN:
            load_name = os.path.join(config.CKPT_PATH, 'gen_{:d}.pth'.format(config.PRETRAIN_GEN))
            generator = load_model(generator, load_name)
            logging.info('Load pre-trained generator from iteration %d' % config.PRETRAIN_GEN)

        elif config.MODE == 'test' and config.MODEL:
            load_name = os.path.join(config.CKPT_PATH, 'gen_{:d}.pth'.format(config.MODEL))
            generator = load_model(generator, load_name)
            logging.info('Load trained generator from iteration %d' % config.MODEL)

        else:
            weights_init(generator, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN)
            logging.info('Initialize generator with %s type' % config.INIT_TYPE)

    return generator


def create_discriminator(config, model_path=None):
    discriminator = Discriminator(
        nc = config.OUT_CHANNELS,
        ndf = config.DIS_FEAT_NUM,
        n_class = config.CLASS_NUM,
        recon_mode = config.RECON_MODE,
    )
    logging.info('Discriminator is created!')

    # Initialize the networks
    if model_path is not None:
        discriminator = load_model(discriminator, model_path)
        print(f'Load pre-trained discriminator from {model_path}')

    else:
        if config.MODE == 'train' and config.PRETRAIN_DIS:
            load_name = os.path.join(config.CKPT_PATH, 'dis_{:d}.pth'.format(config.PRETRAIN_DIS))
            discriminator = load_model(discriminator, load_name)
            logging.info('Load pre-trained discriminator from iteration %d' % config.PRETRAIN_DIS)

        else:
            weights_init(discriminator, init_type=config.INIT_TYPE, init_gain=config.INIT_GAIN)
            logging.info('Initialize discriminator with %s type' % config.INIT_TYPE)

    return discriminator


# ----------------------------------------
#       Exponential Moving Average
# ----------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ----------------------------------------
#               Save Results
# ----------------------------------------

def show_image(imgs, img_name, save_path, denormalize=True, grid_nrow=8):
    # Create save path
    save_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, img_name)

    # Denormalization [-1, 1] --> [0, 1]
    if isinstance(imgs, list):
        grid_nrow = imgs[0].shape[0]
        imgs = torch.cat(imgs, dim=0)

    if denormalize:
        out_imgs = torch.clamp((imgs + 1) / 2, min=0.0, max=1.0)
    else:
        out_imgs = torch.clamp(imgs, min=0.0, max=1.0)

    # Save images
    grid = make_grid(out_imgs, nrow=grid_nrow)
    save_image(grid, filename)
