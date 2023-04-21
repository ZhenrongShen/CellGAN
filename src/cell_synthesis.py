import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import *
from .dataset import CellDataset, categories, InfiniteSamplerWrapper
from .loss import GANLoss, R1_loss
from .fid import get_fid_fn


def train_model(config):
    # Loggings
    open_log(config.LOG_PATH)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # Build networks
    netG = create_generator(config).cuda()
    netD = create_discriminator(config).cuda()

    # Optimizers
    optimizer_G = create_optimizer(config, netG, config.LR_G)
    optimizer_D = create_optimizer(config, netD, config.LR_D)

    # Loss functions
    GAN_criterion = GANLoss(gan_mode=config.GAN_MODE).cuda()

    # Build EMA models
    netG_ema = EMA(netG, decay=config.EMA_DECAY)
    netG_ema.register()

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define training dataset
    train_data = CellDataset(dataroot=config.DATAROOT, normalize=config.DATA_NORM, augment=config.FLIP_AUG)
    logging.info('The overall number of training images is %d' % len(train_data))

    # Define the dataloader
    train_loader = iter(DataLoader(train_data, batch_size=config.BATCH_SIZE, drop_last=True, shuffle=False,
                                   sampler=InfiniteSamplerWrapper(train_data),
                                   num_workers=config.NUM_WORKERS))
    
    # ----------------------------------------
    #              Start Training
    # ----------------------------------------

    # Load pretrained models
    if config.PRETRAIN_GEN:
        start_iter = config.PRETRAIN_GEN
        logging.info("Resume training from iteration %d" % start_iter)
    else:
        start_iter = 0
        logging.info("Start training ......")

    # Training loop
    fix_noise = torch.randn(config.VIS_NUM, config.LATENT_DIMS).cuda().float()
    for iter_idx in tqdm(range(start_iter, config.TOTAL_ITERS), desc="Training"):
        cur_iter = iter_idx + 1  # current iteration

        # ----------------------------------------
        #            Train Discriminator
        # ----------------------------------------

        for _ in range(config.DIS_ITERS):
            # Load image ([B, 3, H, W]) and class label ([B, CLASS_NUM])
            real_img, label = next(train_loader)
            real_img, label = real_img.cuda().float(), label.cuda().float()

            # Sample p(z) and generate fake images
            noise = torch.randn(config.BATCH_SIZE, config.LATENT_DIMS).cuda().float()
            fake_img = netG(noise, label)

            # Discriminator output
            fake_out = netD(fake_img.detach(), label, is_real=False, policy=config.DIFF_AUG)  # Fake samples
            real_out = netD(real_img, label, is_real=True, policy=config.DIFF_AUG)            # Real samples

            # Calculate GAN loss
            loss_fake = GAN_criterion(fake_out['logits'], False, for_discriminator=True)
            loss_real = GAN_criterion(real_out['logits'], True, for_discriminator=True)
            GAN_loss_D = loss_fake + loss_real

            # Calculate reconstruction loss
            rec_loss = real_out['aux_loss']

            # Calculate R1 regularization
            if config.D_R1_WEIGHT is not None:
                real_img.requires_grad = True
                real_out = netD(real_img, label, is_real=True)
                R1 = config.D_R1_WEIGHT * R1_loss(real_out['logits'], real_img)
            else:
                R1 = torch.zeros_like(GAN_loss_D)

            # Update discriminator
            total_loss_D = (GAN_loss_D + rec_loss + R1).mean()
            optimizer_D.zero_grad()
            total_loss_D.backward()
            optimizer_D.step()

        # ----------------------------------------
        #              Train Generator
        # ----------------------------------------

        # Sample p(y) and p(z), and generate fake images
        real_img, label = next(train_loader)
        real_img, label = real_img.cuda().float(), label.cuda().float()
        noise = torch.randn(config.BATCH_SIZE, config.LATENT_DIMS).cuda().float()
        fake_img = netG(noise, label)

        # Calculate GAN loss
        fake_out = netD(fake_img, label, is_real=False, policy=config.DIFF_AUG)
        GAN_loss_G = GAN_criterion(fake_out['logits'], True, for_discriminator=False)

        # Update generator
        total_loss_G = GAN_loss_G.mean()
        optimizer_G.zero_grad()
        total_loss_G.backward()
        optimizer_G.step()

        if cur_iter >= config.EMA_START and config.USE_EMA:
            netG_ema.update()

        # ----------------------------------------
        #            Log training states
        # ----------------------------------------
        if config.LOG_INTERVAL and cur_iter % config.LOG_INTERVAL == 0:
            # Compute discriminator prediction scores
            with torch.no_grad():
                real_out = netD(real_img, label, is_real=True)
                fake_out = netD(fake_img, label, is_real=False)
                real_score = real_out['logits'].mean().item()
                fake_score = fake_out['logits'].mean().item()

            # Print training status
            logging.info(
                '[Iteration {:d}] D(x) / D(G(z)): {:.4f} / {:.4f}'
                .format(cur_iter, real_score, fake_score)
            )

        # ----------------------------------------
        #            Save Checkpoints
        # ----------------------------------------
        if cur_iter >= config.SAVE_START \
                and (cur_iter % config.SAVE_INTERVAL == 0 or cur_iter == config.TOTAL_ITERS):
            # 1. Save generator
            if cur_iter >= config.EMA_START and config.USE_EMA:
                netG_ema.apply_shadow()
                save_model(netG, optimizer_G, 'gen_{:d}.pth'.format(cur_iter), config.CKPT_PATH)
                netG_ema.restore()
            else:
                save_model(netG, optimizer_G, 'gen_{:d}.pth'.format(cur_iter), config.CKPT_PATH)
            logging.info('The trained generator is successfully saved at epoch {:d}'.format(cur_iter))

            # 2. Save discriminator
            save_model(netD, optimizer_D, 'dis_{:d}.pth'.format(cur_iter), config.CKPT_PATH)
            logging.info('The trained discriminator is successfully saved at epoch {:d}\n'.format(cur_iter))

        # ----------------------------------------
        #              Visualization
        # ----------------------------------------
        if cur_iter % config.VIS_INTERVAL == 0 or cur_iter == config.TOTAL_ITERS:
            if cur_iter >= config.EMA_START and config.USE_EMA:
                netG_ema.apply_shadow()

            img_list = []
            with torch.no_grad():
                for category in train_data.available_categories:
                    label = torch.tensor(categories[category]).unsqueeze(0).repeat(config.VIS_NUM, 1)
                    label = label.cuda().float()
                    gen_img = netG(fix_noise, label)
                    img_list.append(gen_img)
            filename = "{:d}.png".format(cur_iter)
            show_image(img_list, filename, config.SAMPLE_PATH, config.DATA_NORM)

            if cur_iter >= config.EMA_START and config.USE_EMA:
                netG_ema.restore()


def test_model(config):
    # ----------------------------------------
    #      Initialize testing parameters
    # ----------------------------------------

    # Build network
    netG = create_generator(config).cuda()
    netG.eval()

    # ----------------------------------------
    #              Start Testing
    # ----------------------------------------

    print("Start testing ......")
    eval_categories = list(categories.keys())
    with torch.no_grad():

        # Visualize generated images
        for category in eval_categories:
            save_path = os.path.join(config.TEST_PATH, category)
            label = torch.tensor(categories[category]).unsqueeze(0).cuda().float()
            for img_idx in tqdm(range(config.SAMPLE_NUM), desc="generating {:s}: ".format(category), leave=False):
                noise = torch.randn(1, config.LATENT_DIMS).cuda().float()
                gen_img = netG(noise, label)
                filename = "{:s}_{:05}.png".format(category, img_idx + 1)
                show_image(gen_img, filename, save_path, config.DATA_NORM)

        # FID evaluation
        if config.EVAL_METRICS:
            fid_scores = get_fid_fn(config, netG)
            eval_info = 'FID scores -'
            for category in eval_categories:
                eval_info += ' {:s}: {:.4f} |'.format(category, fid_scores[category])
            eval_info += ' Average: {:.4f}\n'.format(fid_scores['average'])
            print(eval_info)

    print("Testing finished !!")
