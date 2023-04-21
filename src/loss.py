# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# -----------------------------------------------
#                Adversarial Loss
# -----------------------------------------------

class GANLoss(nn.Module):

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. 
                               It currently supports 'vanilla', 'lsgan', 'wgan', and 'hinge'.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator. LSGAN needs no sigmoid. 
              Vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('zero_tensor', torch.tensor(1).fill_(0))

        self.gan_mode = gan_mode
        assert self.gan_mode in ['vanilla', 'lsgan', 'wgan', 'hinge'], \
            ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, prediction, target_is_real):
        """ Create label tensors with the same size as Discriminator's output.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from Discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A ground truth label tensor with the size of Discriminator's output
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def get_zero_tensor(self, prediction):
        """ Create zero tensor with the same size as Discriminator's output.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from Discriminator
        Returns:
            A zero tensor with the size of Discriminator's output
        """
        self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """ Compute GAN loss given Discriminator's outputs and ground-truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from Discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            for_discriminator (bool) - - if the loss is computed for Discriminator
        Returns:
            the computed loss.
        """
        # Vanilla GAN loss (non-saturating loss)
        if self.gan_mode == 'vanilla':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(prediction, target_tensor)

        # LSGAN loss
        elif self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss_fn = nn.MSELoss()
            loss = loss_fn(prediction, target_tensor)

        # WGAN loss
        elif self.gan_mode == 'wgan':
            loss = -prediction.mean() if target_is_real else prediction.mean()

        # Hinge loss
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(prediction)

        else:
            raise ValueError("Unsupported GAN mode: {:s}".format(self.gan_mode))

        return loss


# -----------------------------------------------
#                R1 Regularization
# -----------------------------------------------

def R1_loss(prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
    """ Forward pass to compute the regularization
    :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
    :param real_sample: (torch.Tensor) Batch of the corresponding real images
    :return: Loss value: (torch.Tensor)
    """
    # Calculate gradient
    grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]

    # Calculate regularization
    regularization_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return regularization_loss
