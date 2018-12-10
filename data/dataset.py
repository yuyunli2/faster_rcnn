import torch
import numpy as np
from data import util
from utils.config import opt
from skimage import transform
from torchvision import transforms
from data.voc_dataset import VOCBboxDataset
from __future__ import  absolute_import
from __future__ import  division

# Constants
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def inverse_normalize(image):
    return (image*0.225+0.45).clip(min=0, max=1)*255

def pytorch_normalize(image):
    normalize = transforms.Normalize(mean=mean, std=std)
    image = (normalize(torch.from_numpy(image))).numpy()
    return image

def preprocess(image, min_size=600, max_size=1000):
    C, H, W = image.shape
    scale = min(min_size/min(H, W), max_size/max(H, W))
    image = image/255
    image = transform.resize(image, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)

    normalize = pytorch_normalize
    return normalize(image)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        image, bbox, label = in_data
        _, H, W = image.shape
        image = preprocess(image, self.min_size, self.max_size)
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # Flip horizontally
        image, params = util.random_flip(image, x_random=True, return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return image, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        ori_image, bbox, label, difficult = self.db.get_example(index)
        image, bbox, label, scale = self.tsf((ori_image, bbox, label))

        return image.copy(), bbox.copy(), label.copy(), scale


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)
    
    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        ori_image, bbox, label, difficult = self.db.get_example(index)
        image = preprocess(ori_image)

        return image, ori_image.shape[1:], bbox, label, difficult


