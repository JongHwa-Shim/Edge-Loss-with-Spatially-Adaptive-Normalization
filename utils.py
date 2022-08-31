import numpy as np
from PIL import Image

from models.networks import SPADEGenerator, Pix2PixHDGenerator, MultiscaleDiscriminator, NLayerDiscriminator

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

def img_show(img):
    from matplotlib import pyplot as plt
    img_dim = len(img.shape)

    if img_dim == 4:
        plt.imshow(img.cpu().detach().squeeze(dim=0).permute([1,2,0]).numpy())
    elif img_dim == 3:
        plt.imshow(img.cpu().detach().numpy().permute([1,2,0]))

    plt.show()
    return None

def modify_commandline_options(opt):
    """
    In case of SPADE Generator
    norm_G: (spectralspadesyncbatch3x3)
    num_unpsampling_layers: (normal|more|most), \
    "If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator"
    
    In case of pix2pix HD
    resnet_n_downsample: ('4'), "number of downsampling layers in netG"
    resnet_n_blocks: ('9'), "number of residual blocks in the global generator network"
    resnet_kernel_size: ('3'), "kernel size of the resnet block"
    resnet_initial_kernel_size: ('7'), "kernel size of the first convolution"
    norm_G: ('spectralinstance')
    """
    if opt.netG == 'spade':
        opt.norm_G = 'spectralspadesyncbatch3x3'
        opt.num_upsampling_layers = 'normal'
    elif opt.netG == 'spadeplus':
        opt.norm_G = 'spectralspadesyncbatch3x3'
        opt.num_upsampling_layers = 'normal'
    elif opt.netG == 'pix2pixhd':
        opt.resnet_n_downsample = 4
        opt.resnet_n_blocks = 9
        opt.resnet_kernel_size = 3
        opt.resnet_initial_kernel_size = 7
        opt.norm_G = 'spectralinstance'
    
    if opt.netD == 'multiscale':
        opt.netD_subarch = 'n_layer'
        opt.num_D = 2
        opt.n_layers_D = 4
    elif opt.netD == 'n_layers':
        opt.n_layers_D = 4
    
    if opt.load_checkpoint is not None:
        opt.epoch_count = int(opt.load_checkpoint) + 1
    
    return opt