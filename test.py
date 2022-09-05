from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='spade with edge loss')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', type=str, default='cuda:0', help='(cuda:n|cpu')

# add new
parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pix|pix2pixhd|spade|spadeplus)')
parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale)')
parser.add_argument('--no_edge_loss', action='store_true', help='if specified, edge loss will not apllied')
parser.add_argument('--memo', type=str, default='', help='additional memo for checkpoint folder')
opt = parser.parse_args('--dataset facades --cuda cpu --netG spadeplus --netD multiscale --nepochs 1 --no_edge_loss --memo one_more_try'.split())
print(opt)

device = torch.device(opt.cuda)

checkpoint_dir = os.path.join("checkpoint", opt.dataset, 'netG={}, netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo))
model_path = os.path.join(checkpoint_dir, 'netG_{}_epoch_{}.pth'.format(opt.netG, opt.nepochs))
# model_path = "checkpoint/{}/netG_{}_epoch_{}.pth".format(opt.dataset, opt.netG, opt.nepochs) # will be deprecated

net_g = torch.load(model_path).to(device)

if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    save_dir = os.path.join('result', opt.dataset, 'netG={},netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo), str(opt.nepochs))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_img(out_img, "{}/{}".format(save_dir, image_name))
