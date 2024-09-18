from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as FT
import torch
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torchvision.transforms.v2 as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    Create lists for images in the training set and each of the test sets.

    :param train_folders: folders containing the training images; these will be merged
    :param test_folders: folders containing the test images; each test folder will form its own test set
    :param min_size: minimum width and height of images to be considered
    :param output_folder: save data lists here
    """
    print("\nCreating data lists... this may take some time.\n")
    train_images = list()
    if train_folders is not None:
        for d in train_folders:
            for i in tqdm(os.listdir(d)):
                img_path = os.path.join(d, i)
                img = Image.open(img_path, mode='r')
                if img.width >= min_size and img.height >= min_size:
                    train_images.append(img_path)
        print("There are %d images in the training data.\n" % len(train_images))
        with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
            json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in tqdm(os.listdir(d)):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("There are %d images in the %s test data.\n" % (len(test_images), test_name))
        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    print("JSONS containing lists of Train and Test images have been saved to %s\n" % output_folder)


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class PoissonNoise(transforms.Transform):
    def __init__(self, scale=10.0):
        super().__init__()
        self.scale = scale

    def forward(self, img):
        # img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        scaled_img = img * self.scale
        noisy_img = torch.poisson(scaled_img) / self.scale
        # noisy_img = torch.poisson(img) 
        noisy_img = torch.clamp(noisy_img, 0, 1)  # Ensure values are within [0, 1]
        # return (noisy_img * 255).to(torch.uint8)  # Convert back to [0, 255]
        return noisy_img

class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type, 
                 noise_probability:float=0.0, return_orig_lr=False, random_flips=False):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        :param noise_probability: probability that noise will be added to LR image
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.noise_probability = noise_probability
        # self.noise_gen = transforms.GaussianNoise(sigma=30/255.)
        self.noise_gen = PoissonNoise(200.0)
        self.h_flip = transforms.RandomHorizontalFlip(0.5)
        self.v_flip = transforms.RandomVerticalFlip(0.5)
        assert self.split in {'train', 'test'}
        self.return_orig_lr=return_orig_lr
        self.random_flips = random_flips

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            try:
                left = random.randint(1, img.width - self.crop_size)
            except ValueError as e:
                print(f'\tImage width ({img.width}) too small to crop with crop_size {self.crop_size}')
                left = img.width
            try:
                top = random.randint(1, img.height - self.crop_size)
            except ValueError as e:
                print(f'\tImage height ({img.height}) too small to crop with crop_size {self.crop_size}')
                top = img.height
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        if self.random_flips:
            hr_img = self.h_flip(hr_img)
            hr_img = self.v_flip(hr_img)
        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
      
        # Randomly determine if noise should be added to LR image
        if torch.rand(1).item() < self.noise_probability:
            lr_img_noised = self.noise_gen(lr_img)
        else:
            lr_img_noised = lr_img
        if self.return_orig_lr:
            return lr_img_noised, hr_img, lr_img

        return lr_img_noised, hr_img


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get(self):
        return self.avg


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def visualize_superres(img, srresnet, device, halve=False, noise=False):
    """
    Visualizes the super-resolved images from the SRResNet for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, using matplotlib.

    :param img: filepath of the HR image
    :param srresnet: the SRResNet model
    :param device: the device to run the model on
    :param halve: halve each dimension of the HR image
    """
    srresnet.eval()
    if not isinstance(img, list):
        images = [img]
    else:
        images = img
    # Create plot
    fig, axs = plt.subplots(len(img), 4, figsize=(4.25*len(img), 3*len(img)))
    fig.suptitle('Super-Resolution Comparison', fontsize=16, fontweight='bold')
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
            # noiser = transforms.GaussianNoise(sigma=30/255.)
            noiser = PoissonNoise(200.0)
            lr_img = noiser(lr_img)
            lr_img = torch.clamp(lr_img, min=0.0, max=1.0)
            # lr_img = torch.clamp(lr_img + torch.randn(lr_img.size())*0.05, min=0.0, max=1.0)
            # lr_tensor = convert_image(lr_img.permute(2,0,1), source='[0, 1]', target='imagenet-norm').unsqueeze(0).to(device)
            lr_tensor = lr_img.permute(2,0,1).unsqueeze(0).to(device)
        else:
            lr_tensor = convert_image(lr_img, source='pil', target='[0, 1]').unsqueeze(0).to(device)

        if noise:
            # For bicubic upsampling
            bicubic_img = resize(lr_img, (hr_img.shape[0], hr_img.shape[1]), 
                            anti_aliasing=True, mode='reflect', order=3)
        else:
            # For bicubic upsampling
            bicubic_img = resize(lr_img, (hr_img.shape[0], hr_img.shape[1]), 
                            anti_aliasing=True, mode='reflect', order=3)
        # Super-resolution (SR) with SRResNet
        sr_img_srresnet = srresnet(lr_tensor)
        sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
        # sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
        sr_img_srresnet = convert_image(sr_img_srresnet, source='[0, 1]', target='pil')
        # sr_img_srresnet = np.array(sr_img_srresnet) / 255.0

        axs[iter, 0].imshow(lr_img)
        if noise:
            axs[iter, 0].set_title('DS LR + Noise')    
        else:
            axs[iter, 0].set_title('Downsampled LR')  
        axs[iter, 0].axis('off')

        axs[iter, 1].imshow(bicubic_img)
        axs[iter, 1].set_title('Bicubic')
        axs[iter, 1].axis('off')

        axs[iter, 2].imshow(sr_img_srresnet)
        axs[iter, 2].set_title('SRResNet')
        axs[iter, 2].axis('off')

        axs[iter, 3].imshow(hr_img)
        axs[iter, 3].set_title('Original HR')
        axs[iter, 3].axis('off')

    plt.tight_layout()
    # plt.show()

    srresnet.train()

    return fig

