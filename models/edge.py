import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
from imageio import imread, imsave
import torch
from torch.autograd import Variable

from utils import img_show, show_hist

torch.autograd.set_detect_anomaly(True)

class Net(nn.Module):
    def __init__(self, std=1.0, lower_threshold=6.0, upper_threshold=50.0, device='cpu'):
        super(Net, self).__init__()

        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.device = device

        filter_size = 5
        generated_filters = gaussian(filter_size,std=std)

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight = nn.Parameter(torch.FloatTensor(generated_filters.reshape([1,1,1,filter_size])))
        self.gaussian_filter_horizontal.bias = nn.Parameter(torch.FloatTensor(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight = nn.Parameter(torch.FloatTensor(generated_filters.T.reshape([1,1,filter_size,1])))
        self.gaussian_filter_vertical.bias = nn.Parameter(torch.FloatTensor(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=np.float32)

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight = nn.Parameter(torch.FloatTensor(sobel_filter.reshape([1,1,3,3])))
        self.sobel_filter_horizontal.bias = nn.Parameter(torch.FloatTensor(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight = nn.Parameter(torch.FloatTensor(sobel_filter.T.reshape([1,1,3,3])))
        self.sobel_filter_vertical.bias = nn.Parameter(torch.FloatTensor(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]], dtype=np.float32)
        filter_0 = filter_0.reshape([1,3,3])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]], dtype=np.float32)
        filter_45 = filter_45.reshape([1,3,3])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]], dtype=np.float32)
        filter_90 = filter_90.reshape([1,3,3])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]], dtype=np.float32)
        filter_135 = filter_135.reshape([1,3,3])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]], dtype=np.float32)
        filter_180 = filter_180.reshape([1,3,3])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]], dtype=np.float32)
        filter_225 = filter_225.reshape([1,3,3])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]], dtype=np.float32)
        filter_270 = filter_270.reshape([1,3,3])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]], dtype=np.float32)
        filter_315 = filter_315.reshape([1,3,3])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight = nn.Parameter(torch.FloatTensor(all_filters))
        self.directional_filter.bias = nn.Parameter(torch.FloatTensor(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):

        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])
        """
        max = blurred_img.max()
        min = blurred_img.min()
        nor_img = (blurred_img - blurred_img.min())/(max - min)
        """
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(torch.pow(grad_x_r,2) + torch.pow(grad_y_r,2) + 1e-8)
        grad_mag = grad_mag + torch.sqrt(torch.pow(grad_x_g,2) + torch.pow(grad_y_g,2) + 1e-8)
        grad_mag = grad_mag + torch.sqrt(torch.pow(grad_x_b,2) + torch.pow(grad_y_b,2) + 1e-8)


        # THIN EDGES (NON-MAX SUPPRESSION)
        # not differentiable
        with torch.no_grad():
            grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
            grad_orientation += 180.0
            grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

            all_filtered = self.directional_filter(grad_mag)

            indices_positive = (grad_orientation / 45) % 8
            indices_negative = ((grad_orientation / 45) + 4) % 8

            height = indices_positive.size()[2]
            width = indices_positive.size()[3]
            pixel_count = height * width
            pixel_range = torch.FloatTensor([range(pixel_count)]).to(self.device)            

            indices = (indices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
            channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

            indices = (indices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
            channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

            channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

            is_max = channel_select_filtered.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag
        thin_edges[is_max==0] = 0.0

        # THRESHOLD
        thresholded = thin_edges
        thresholded[thin_edges<self.lower_threshold] = 0.0
        thresholded[thin_edges>self.upper_threshold] = 0.0

        thresholded[thresholded > 0.0] = 1.0
        
        # do not need
        """
        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0
        
        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()
        """
        # return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold
        return thresholded

def canny(raw_img, img_name, device):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(std=1.0, lower_threshold=6.0, upper_threshold=50.0, device=device).to(device)
    net.eval()

    data = Variable(batch).to(device)

    data.requires_grad = True

    optimizer = torch.optim.SGD([data], lr=0.001, momentum=0.9)

    x = imread('./sample/seg_map.jpg')/255.0
    x = torch.from_numpy(x.transpose((2,0,1)))
    x = torch.stack([x]).float().to(device)
    x = torch.nn.functional.interpolate(x, (600,600))
    x_ = net(x).clone().detach()
    L1Loss = nn.L1Loss()

    for i in range(1000):

        thresholded = net(data)
        # loss = -torch.mean(thresholded)
        loss = 10000 * L1Loss(thresholded, x_)
        optimizer.zero_grad()
        loss.backward()
        print(loss.data)
        optimizer.step()
        if i%100 == 0:
            # if img -1 ~ 1
            # a = (data.clone().detach()+ 1) / 2 
            # if img 0 ~ 1
            img_show(data)
    
    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

    result_dir = './result'
    imsave(os.path.join(result_dir, img_name+'gradient_magnitude.png'), grad_mag.data.cpu().numpy()[0,0])
    imsave(os.path.join(result_dir, img_name+'thin_edges.png'), thresholded.data.cpu().numpy()[0, 0])
    imsave(os.path.join(result_dir, img_name+'final.png'), (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    imsave(os.path.join(result_dir, img_name+'threshold.png'), early_threshold.data.cpu().numpy()[0, 0])


if __name__ == '__main__':
    img_dir = './sample'
    img_file = 'cmp_b0005.png'
    img_name = img_file.split('.')[0]
    img = imread(os.path.join(img_dir, img_file)) / 255.0

    # canny(img, use_cuda=False)
    canny(img, img_name, device='cuda:0')
