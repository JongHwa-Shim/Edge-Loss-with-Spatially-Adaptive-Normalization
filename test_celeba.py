from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
from dataset import DatasetFromFolderCelebATest
from torch.utils.data import DataLoader

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
parser.add_argument('--input_h', type=int, default=256, help='input height')
parser.add_argument('--input_w', type=int, default=256, help='input width')
opt = parser.parse_args('--dataset celeba --cuda cpu --netG spadepluslite --netD multiscale --nepochs 60 --no_edge_loss'.split())
print(opt)

device = torch.device(opt.cuda)

checkpoint_dir = os.path.join("checkpoint", opt.dataset, 'netG={}, netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo))
model_path = os.path.join(checkpoint_dir, 'netG_{}_epoch_{}.pth'.format(opt.netG, opt.nepochs))
# model_path = "checkpoint/{}/netG_{}_epoch_{}.pth".format(opt.dataset, opt.netG, opt.nepochs) # will be deprecated

net_g = torch.load(model_path).to(device)

# if opt.direction == "a2b":
#     image_dir = "dataset/{}/test/a/".format(opt.dataset)
# else:
#     image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_dir = "dataset/{}/test".format(opt.dataset)

dataset = DatasetFromFolderCelebATest(image_dir, opt.direction, opt)
testing_data_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False)

with torch.no_grad():
    for mask, file_name in testing_data_loader:
        mask = mask.to(device)
        out = net_g(mask)
        out_img = out.detach().cpu().squeeze(0)

        save_dir = os.path.join('result', opt.dataset, 'netG={},netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo), str(opt.nepochs))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_img(out_img, "{}/{}".format(save_dir, file_name[0]))

# for image_name in image_filenames:
#     img = load_img(image_dir + image_name)
#     img = transform(img)
#     input = img.unsqueeze(0).to(device)
#     out = net_g(input)
#     out_img = out.detach().squeeze(0).cpu()

#     save_dir = os.path.join('result', opt.dataset, 'netG={},netD={}, edgeloss={}{}'.format(opt.netG, opt.netD, str(not(opt.no_edge_loss)), opt.memo), str(opt.nepochs))
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     save_img(out_img, "{}/{}".format(save_dir, image_name))
