import os
import argparse
import torch

from src.config import Config
from src.utils import set_seed
from src.cell_synthesis import train_model


if __name__ == "__main__":

    # Configuration
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('--config', type=str, default='default_config', help='configuration filename')
    args = parser.parse_args()
    config = Config(filename=args.config, mode='train')

    # Device
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    # Random seed
    set_seed(config.MANUAL_SEED)

    # Main
    config.print_info()
    train_model(config)
