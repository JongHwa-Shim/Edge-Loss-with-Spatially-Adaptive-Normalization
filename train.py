from __future__ import print_function
import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from models.networks import EdgeLoss, VGGLoss, define_G, define_D, GANLoss, get_scheduler, update_learning_rate, FeatLoss, SPADEGenerator, MultiscaleDiscriminator, Pix2PixHDGenerator, NLayerDiscriminator
from data import get_training_set, get_test_set
from utils import modify_commandline_options

# Training settings
parser = argparse.ArgumentParser(description='spade with edge loss')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate for adam, pix2pix=0.0002, spade=0.0001')
parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate for adam, pix2pix=0.0002, spade=0.0004')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam. default=0.0 for spade, default=0.5 for pix2pix')
parser.add_argument('--cuda', type=str, default='cuda:0', help='(cuda:n|cpu')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb_l1', type=int, default=10, help='weight on L1 term in objective')

# add new
parser.add_argument('--img_width', type=int, default=286, help='image width after resize (not final size)')
parser.add_argument('--img_height', type=int, default=286, help='image height after resize (not final size)')
parser.add_argument('--crop_size_width', type=int, default=256, help='final image width, Crop to the width of crop_size (after initially scaling the images to load_size.)')
parser.add_argument('--crop_size_height', type=int, default=256, help='final image height, Crop to the height of crop_size (after initially scaling the images to load_size.)')

parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pix|pix2pixhd|spade|spadeplus)')
parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale)')
parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
parser.add_argument('--lamb_feat', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
parser.add_argument('--no_edge_loss', action='store_true', help='if specified, edge loss will not apllied')
parser.add_argument('--lambda_edge', type=int, default=100, help='weight for edge matching loss')
parser.add_argument('--label_nc', type=int, default=3, help='discriminator input segmap channels, usually same with input_nc')
parser.add_argument('--contain_dontcare_label', action='store_true', help='is there dont care label segmap in D input? if true, discriminator input dimension 1++')
parser.add_argument('--no_instance', action='store_true', help='is there no instance segmap in D input? if true, discriminator input dimension 1++')
parser.add_argument('--load_checkpoint', type=str, default=None, help=r'load model with "checkpoint/{--dataset}/netX_{--netG|--netD}_epoch_{--load_checkpoint}"')
parser.add_argument('--memo', type=str, default='', help='additional memo for checkpoint folder')
parser.add_argument('--lower_threshold', type=float, default=10.0, help='lower bound threshold of edge detection')
parser.add_argument('--upper_threshold', type=float, default=50.0, help='upper bound threshold of edge detection')
opt = parser.parse_args('--dataset facades --cuda cuda:0 --netG spadeplus --netD multiscale --no_instance --no_edge_loss --memo one_more_try'.split())
opt = modify_commandline_options(opt)
print(opt)

# if opt.cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if 'cuda' in opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction, opt)
test_set = get_test_set(root_path + opt.dataset, opt.direction, opt)
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device(opt.cuda)

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device, opt=opt)
net_d = define_D(opt.label_nc + opt.output_nc, opt.ndf, opt.netD, gpu_id=device, opt=opt)

criterionGAN = GANLoss().to(device)
#criterionL1 = nn.L1Loss().to(device)
criterionL1 = VGGLoss().to(device)
if not opt.no_ganFeat_loss:
    criterionFeat = FeatLoss(opt).to(device)
criterionMSE = nn.MSELoss().to(device) # only for validation
if not opt.no_edge_loss:
    criterionEdge = EdgeLoss(opt).to(device)

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

for epoch in range(opt.epoch_count, opt.epoch_count + opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator, adversarial loss
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, feature matching loss and L1 loss
        
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb_l1

        loss_g = loss_g_gan + loss_g_l1

        # Thired, Feature Matching Loss
        if not opt.no_ganFeat_loss:
            loss_g_feat = criterionFeat(pred_fake, pred_real) * opt.lamb_feat
            loss_g = loss_g + loss_g_feat

        # Fourth, Edge matching loss
        if not opt.no_edge_loss:
            loss_g_edge = criterionEdge(fake_b, real_b) * opt.lambda_edge
            loss_g = loss_g + loss_g_edge
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 1 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        checkpoint_dir = os.path.join("checkpoint", opt.dataset, 'netG={}, netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        net_g_model_out_path = os.path.join(checkpoint_dir, 'netG_{}_epoch_{}.pth'.format(opt.netG, epoch))
        net_d_model_out_path = os.path.join(checkpoint_dir, 'netD_{}_epoch_{}.pth'.format(opt.netD, epoch))
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format(checkpoint_dir))
