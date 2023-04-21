import os
import argparse
from tqdm import tqdm
import torch

from src.utils import *
from src.config import Config
from src.dataset import categories


def main(args):

    # configuration
    config = Config(filename=args.config, mode='test')

    # Device
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(e) for e in config.GPU)
    torch.backends.cudnn.benchmark = True
    set_seed(config.MANUAL_SEED)

    # Build the generator
    netG = create_generator(config, args.model).cuda()

    # Create output directory
    save_path = os.path.join(args.output_dir, args.cell_type)
    os.makedirs(save_path, exist_ok=True)

    # Generate images
    with torch.no_grad():
        label = torch.tensor(categories[args.cell_type]).unsqueeze(0).cuda().float()
        for img_idx in tqdm(range(args.data_num), desc="generating {:s}: ".format(args.cell_type)):
            noise = torch.randn(1, config.LATENT_DIMS).cuda().float()
            gen_img = netG(noise, label)
            filename = "{:s}_{:05}.png".format(args.cell_type, img_idx + 1)
            show_image(gen_img, filename, save_path, config.DATA_NORM)


if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('--config', type=str, default='default_config', help='configuration filename')
    parser.add_argument('--model', type=str, default='checkpoints/model.pth', help='model weights')
    parser.add_argument('--output_dir', type=str, default='OUTPUT', help='output directory')
    parser.add_argument('--cell_type', type=str, default='NILM', help='cervical squamous cell type',
                        choices=['NILM', 'ASC_US', 'LSIL', 'ASC_H', 'HSIL'])
    parser.add_argument('--data_num', type=int, default=10, help='number of generated images')
    args = parser.parse_args()

    # Device
    main(args)
