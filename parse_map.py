import os
import numpy as np
from PIL import Image


def parse_folder(data_dir):
    a_dir = os.path.join(save_dir, 'a')
    b_dir = os.path.join(save_dir, 'b')

    if not os.path.exists(a_dir):
        os.makedirs(a_dir)
    if not os.path.exists(b_dir):
        os.makedirs(b_dir)

    file_name_list = os.listdir(data_dir)

    for file_name in file_name_list:
        file_path = os.path.join(data_dir, file_name)

        np_img = np.array(Image.open(file_path)) # (600, 1200, 3)
        map_img, map_seg = np.split(np_img, 2, axis=1) # (600, 600, 3)
        img_pil = Image.fromarray(map_img)
        seg_pil = Image.fromarray(map_seg)

        img_pil.save(os.path.join(b_dir, file_name))
        seg_pil.save(os.path.join(a_dir, file_name))

        
if __name__ == '__main__':
    save_dir = './dataset/cityscape/test'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_dir = './dataset/unprocessed/cityscape/val'
    parse_folder(data_dir)