import time
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
from tqdm import tqdm

# Data parameters
data_folder = './data_lists'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "checkpoints/srresnet/new_poisson_denoise_less_noise/checkpoint_srresnet.pt"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 32  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader

vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 5e-3  # the coefficient to weight the adversarial loss in the perceptual loss
color_loss_alpha = 2.0
print_freq = 100  # print training status once every __ batches
lr = 0.5e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

noise_prob = 0.75 # Probability that Gaussian noise will be added to LR image

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

# Where to save checkpoints
ckpt_dir = "checkpoints/srgan_denoising_beta=5e-3_half_lr_color_correction_2_generator_train_10"

def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)

        # Initialize generator network with pretrained SRResNet
        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

        # Initialize generator's optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr)

        # Discriminator
        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)

        # Initialize discriminator's optimizer
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='[0, 1]',
                              hr_img_type='[0, 1]',
                              noise_probability=noise_prob,
                              random_flips=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, prefetch_factor=2)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    
    writer = SummaryWriter(ckpt_dir,flush_secs=30)
    # Epochs
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch's training
        loss_a, loss_c, loss_d, loss_color = train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              truncated_vgg19=truncated_vgg19,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch,
              writer=writer)
        figure = visualize_superres(['../Super-resolution/Image Super-Resolution/Classical/Set14/original/lenna.png', 
                                     '../Super-resolution/Image Super-Resolution/Classical/Set14/original/baboon.png'], 
                                    generator, device, noise=True)
        writer.add_figure('val/image', figure=figure, global_step=epoch, close=True)
        writer.add_scalar('train/adversarial_loss', loss_a, epoch)
        writer.add_scalar('train/content_loss', loss_c, epoch)
        writer.add_scalar('train/mse_loss', loss_color, epoch)
        writer.add_scalar('train/discriminator_loss', loss_d, epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                   f'{ckpt_dir}/checkpoint_epoch_{epoch}_srgan.pt')
    writer.close()


def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch, writer):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator
    losses_color = AverageMeter()  # adversarial loss in the discriminator
    start = time.time()
    mse_color = nn.MSELoss().to(device)

    # Batches
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (lr_imgs, hr_imgs) in enumerate(pbar):
        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        # sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they're constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        color_loss = mse_color(sr_imgs, hr_imgs)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss + color_loss_alpha * color_loss

        # Back-prop.
        if epoch==0 and i>10:
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer_g, grad_clip)

            # Update generator
            optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))
        losses_color.update(color_loss.item(), lr_imgs.size(0))
    
        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Print status
        if i % print_freq == 0:
            pbar.set_postfix(cont=losses_c.get(), adv=losses_a.get(), disc=losses_d.get(), mse=losses_color.get())
            writer.add_scalar('train/avg_cont', losses_c.get(), epoch*len(train_loader)+i)
            writer.add_scalar('train/avg_adv', losses_a.get(), epoch*len(train_loader)+i)
            writer.add_scalar('train/avg_disc', losses_d.get(), epoch*len(train_loader)+i)
            writer.add_scalar('train/avg_mse', losses_color.get(), epoch*len(train_loader)+i)

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored
    pbar.close()
    print(f'\tEpoch time:\t{time.time()-start:.1f} seconds.')

    return losses_a.avg, losses_c.avg, losses_d.avg, losses_color.avg

if __name__ == '__main__':
    main()
