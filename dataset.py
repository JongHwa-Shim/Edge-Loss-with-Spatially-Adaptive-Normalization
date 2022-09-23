from os import listdir
from os.path import join
import os
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, opt):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        self.opt = opt

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((self.opt.img_width, self.opt.img_height), Image.BICUBIC)
        b = b.resize((self.opt.img_width, self.opt.img_height), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)

        # random crop augmentation
        w_offset = random.randint(0, max(0, self.opt.img_width - self.opt.crop_size_width - 1))
        h_offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
        offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
    
        a = a[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]
        b = b[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]

        # -1~1 image scaling
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderCelebA(data.Dataset):
    def __init__(self, image_dir, direction, opt):
        super(DatasetFromFolderCelebA, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        
        self.mask_anno_list = self.get_anno(join(image_dir, "mask_anno.txt"))
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        self.opt = opt

    def get_anno(self, anno_txt):
        with open(anno_txt, 'r') as f:
            anno_list = f.read().split(' ')
        return anno_list
    
    def get_mask(self, mask_dir, image_filename):
        # image_filename = '1234.jpg'
        # mask_filename = 'xxxxx_attr.png'
        img_num, _ = image_filename.split('.')

        if len(img_num) < 5:
            num_zero = 5 - len(img_num)
            zeros = '0'*num_zero
            img_num = zeros + img_num
        
        mask_tensor_list = []
        for i, mask_anno in enumerate(self.mask_anno_list):
            mask_filename = '{}_{}.png'.format(img_num, mask_anno)
            mask_path = join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((self.opt.img_width, self.opt.img_height), Image.BICUBIC)
                mask = transforms.ToTensor()(mask) # mask 0~1
                mask = transforms.Normalize(0.5, 0.5)(mask) # -1~1 scaling
            else:
                mask = torch.full((1, self.opt.img_width, self.opt.img_height), -1.0) # fill -1

            mask_tensor_list.append(mask)
        
        mask_tensor = torch.cat(mask_tensor_list, dim=0) # (C, H, W)

        return mask_tensor

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((self.opt.img_width, self.opt.img_height), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = self.get_mask(self.b_path, self.image_filenames[index])

        # random crop augmentation
        w_offset = random.randint(0, max(0, self.opt.img_width - self.opt.crop_size_width - 1))
        h_offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
        offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
    
        a = a[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]
        b = b[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]

        # -1~1 image scaling
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        #b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # random vertical flip
        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderCelebATest(data.Dataset):
    def __init__(self, image_dir, direction, opt):
        super(DatasetFromFolderCelebATest, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        
        self.mask_anno_list = self.get_anno(join(image_dir, "mask_anno.txt"))
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        self.opt = opt

    def get_anno(self, anno_txt):
        with open(anno_txt, 'r') as f:
            anno_list = f.read().split(' ')
        return anno_list
    
    def get_mask(self, mask_dir, image_filename):
        # image_filename = '1234.jpg'
        # mask_filename = 'xxxxx_attr.png'
        img_num, _ = image_filename.split('.')

        if len(img_num) < 5:
            num_zero = 5 - len(img_num)
            zeros = '0'*num_zero
            img_num = zeros + img_num
        
        mask_tensor_list = []
        for i, mask_anno in enumerate(self.mask_anno_list):
            mask_filename = '{}_{}.png'.format(img_num, mask_anno)
            mask_path = join(mask_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((self.opt.input_h, self.opt.input_w), Image.BICUBIC)
                mask = transforms.ToTensor()(mask) # mask 0~1
                mask = transforms.Normalize(0.5, 0.5)(mask) # -1~1 scaling
            else:
                mask = torch.full((1, self.opt.input_h, self.opt.input_w), -1.0) # fill -1

            mask_tensor_list.append(mask)
        
        mask_tensor = torch.cat(mask_tensor_list, dim=0) # (C, H, W)

        return mask_tensor

    def __getitem__(self, index):
        # a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        # a = a.resize((self.opt.img_width, self.opt.img_height), Image.BICUBIC)
        # a = transforms.ToTensor()(a)
        filename = self.image_filenames[index]
        b = self.get_mask(self.b_path, filename)

        # random crop augmentation
        # w_offset = random.randint(0, max(0, self.opt.img_width - self.opt.crop_size_width - 1))
        # h_offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
        # offset = random.randint(0, max(0, self.opt.img_height - self.opt.crop_size_height - 1))
    
        # a = a[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]
        # b = b[:, h_offset:h_offset + self.opt.crop_size_height, w_offset:w_offset + self.opt.crop_size_width]

        # -1~1 image scaling
        # a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # random vertical flip
        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        return b, filename

    def __len__(self):
        return len(self.image_filenames)
