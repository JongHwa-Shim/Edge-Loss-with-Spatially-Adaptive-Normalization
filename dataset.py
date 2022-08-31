from os import listdir
from os.path import join
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
