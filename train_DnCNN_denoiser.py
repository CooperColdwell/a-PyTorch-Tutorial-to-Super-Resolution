import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models import ResNetDenoiser, TruncatedVGG19
from denoising_models import DnCNN
from datasets import SRDataset
from utils import *
from tqdm import tqdm

# Data parameters
data_folder = './data_lists'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 48  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e6  # number of training iterations
workers = 6  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status once every __ batches
lr = 1e-3  # learning rate
grad_clip = None  # clip if gradients are exploding
patience = 15

noise_prob = 0.9 # Probability that Gaussian noise will be added to LR image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py

cudnn.benchmark = True

# Where to save checkpoints
ckpt_dir = "checkpoints/denoiser/DnCNN_3_higher_lr_leaky"

def gram(x, normalize=False):
    b, c, h, w = x.size()
    x = x.view(b, c, -1)
    gram_matrices = torch.bmm(x, x.transpose(1, 2)) # I'm almost positive the order of these arguments is backwards for calculating the GM
    
    if normalize:
        gram_matrices = gram_matrices.div(c * h * w)
    
    return gram_matrices


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # model = UNet(3, 3, 5, 6, False, False, 'upconv')
        model = DnCNN(3)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.to(device).eval()

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='[0, 1]',
                            #   hr_img_type='[-1, 1]')
                              hr_img_type='[0, 1]',
                              noise_probability=noise_prob,
                              return_orig_lr=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True, prefetch_factor=2)  # note that we're passing the collate function here
    # val_dataset = SRDataset(data_folder,
    #                           split='test',
    #                           crop_size=crop_size,
    #                           scaling_factor=scaling_factor,
    #                           lr_img_type='imagenet-norm',
    #                           hr_img_type='[-1, 1]', test_data_name='Set14')
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
    #                                            pin_memory=False)  # note that we're passing the collate function here
    
    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    best_val_loss = 1e6 # default validation loss: really high
    epochs_no_improvement = 0

    writer = SummaryWriter(ckpt_dir, flush_secs=15)
    # Epochs
    for epoch in range(start_epoch, epochs):
        if epochs_no_improvement>=patience:
            print(f'Training has gone {epochs_no_improvement} epochs without improvement. Early stopping...')
            break
        # One epoch's training
        loss = train(train_loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        epoch=epoch,
                        writer=writer,
                        truncated_vgg=None,
                        style_coeffs=[0.25, 1])
        
        # loss = validate(val_loader, model=model, criterion=criterion)
        figure = visualize_denoise(['../Super-resolution/Image Super-Resolution/Classical/Set14/original/lenna.png', 
                                     '../Super-resolution/Image Super-Resolution/Classical/Set14/original/baboon.png'], 
                                    model, device, noise=True)
        writer.add_figure('val/image', figure=figure, global_step=epoch, close=True)
        writer.add_scalar('train/loss', loss, epoch)
        if loss < best_val_loss:
        #     print(f'\tValidation loss improved! Old value: {best_val_loss:.4f}, new value: {loss:.4f}')
            best_val_loss = loss
            epochs_no_improvement = 0
        #     torch.save({'epoch': epoch,
        #             'model': model,
        #             'optimizer': optimizer},
        #            f'checkpoints/srresnet/ckpt_{best_val_loss:.4f}.pth.tar')
                # Save checkpoint
            torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer},
                    ckpt_dir+'/checkpoint_srresnet.pt')
        else:
            epochs_no_improvement += 1



    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, truncated_vgg=None, style_coeffs=None):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    losses = AverageMeter()  # loss

    start = time.time()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (lr_img_noised, hr_imgs, lr_imgs) in enumerate(pbar):
        # Move to default device
        lr_img_noised = lr_img_noised.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        # hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        denoised_imgs = model(lr_img_noised)  # (N, 3, 96, 96), in [-1, 1]
        # print(denoised_imgs.shape, lr_imgs.shape)
        loss = criterion(denoised_imgs, lr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Print status
        if i % print_freq == 0:
            writer.add_scalar('train/inst_loss', loss.item(), epoch*len(train_loader)+i)
            pbar.set_postfix(loss=losses.get())

    del lr_imgs, hr_imgs  # free some memory since their histories may be stored

    pbar.close()
    print(f'\tEpoch time:\t{time.time()-start:.1f} seconds.')

    return losses.avg

def validate(val_loader, model, criterion):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.eval()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    iter=0
    # Batches
    with torch.no_grad():
        try:
            for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
                data_time.update(time.time() - start)

                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

                # Loss
                loss = criterion(sr_imgs, hr_imgs)  # scalar

                # Keep track of loss
                losses.update(loss.item(), lr_imgs.size(0))

                # Keep track of batch time
                batch_time.update(time.time() - start)

                # Reset start time
                start = time.time()

                # Print status
                if i % print_freq == 0:
                    print('\t Validation:'
                        'Loss {loss.val:.4f} (avg. {loss.avg:.4f})'.format(loss=losses))
                iter = i
            del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
        except:
            print(iter)
            raise ZeroDivisionError
    return losses.avg

def visualize_denoise(img, denoiser, device, halve=False, noise=False):
    """
    """
    denoiser.eval()
    if not isinstance(img, list):
        images = [img]
    else:
        images = img
    # Create plot
    fig, axs = plt.subplots(len(img), 3, figsize=(4.25*len(img), 3*len(img)))
    fig.suptitle('Denoising Comparison', fontsize=16, fontweight='bold')
    for iter, img in enumerate(images):
        # Load image, downsample to obtain low-res version
        hr_img = plt.imread(img)
        if hr_img.dtype == np.uint8:
            hr_img = hr_img.astype(np.float32) / 255.0
        
        if halve:
            hr_img = resize(hr_img, (int(hr_img.shape[0] / 2), int(hr_img.shape[1] / 2)), 
                            anti_aliasing=True)
        
        # For downsampling
        lr_img = resize(hr_img, (int(hr_img.shape[0] / 4), int(hr_img.shape[1] / 4)), 
                        anti_aliasing=True, mode='reflect')

        # Add noise to the image if applicable
        if noise:
            lr_img = torch.Tensor(lr_img)
            noiser = transforms.GaussianNoise(sigma=0.04)
            lr_img_noise = noiser(lr_img)
            lr_img_noise = torch.clamp(lr_img_noise, min=0.0, max=1.0)
            # lr_img = torch.clamp(lr_img + torch.randn(lr_img.size())*0.05, min=0.0, max=1.0)
            lr_tensor = lr_img_noise.permute(2,0,1).unsqueeze(0).to(device)
        else:
            lr_tensor = convert_image(lr_img, source='pil', target='[0, 1]').unsqueeze(0).to(device)

        # Super-resolution (SR) with SRResNet
        denoised_img = denoiser(lr_tensor)
        denoised_img = denoised_img.squeeze(0).cpu().detach()
        denoised_img = convert_image(denoised_img, source='[0, 1]', target='pil')
        # sr_img_srresnet = np.array(sr_img_srresnet) / 255.0

        axs[iter, 0].imshow(lr_img)
        if noise:
            axs[iter, 0].set_title('Low-Res')    
        else:
            axs[iter, 0].set_title('Downsampled LR')  
        axs[iter, 0].axis('off')

        axs[iter, 1].imshow(lr_img_noise)
        axs[iter, 1].set_title('Noised LR')
        axs[iter, 1].axis('off')
        axs[iter, 2].imshow(denoised_img)
        axs[iter, 2].set_title('Denoised LR')
        axs[iter, 2].axis('off')
        

    plt.tight_layout()
    # plt.show()

    denoiser.train()

    return fig

if __name__ == '__main__':
    main()
