from __future__ import division
from __future__ import  absolute_import
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cupy as cp
from dataset import preprocess
from . import array_tool
from config import opt
from model.utils.nms import non_maximum_suppression
from model.utils.bbox_tools import loc2bbox


def nograd(f):
    def new_f(*args,**kwargs):
        with torch.no_grad():
            return f(*args,**kwargs)
    return new_f


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, loc_normalize_mean = (0., 0., 0., 0.), loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        # Head returns class-dependent localization params and class scores
        self.head = head
        # Extractor returns feature maps
        self.extractor = extractor
        self.rpn = rpn
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    # Number of classes
    def n_class(self):
        return self.head.n_class

    def use_preset(self, preset):
        if(preset == 'visualize'):
            self.nms_thresholdold = 0.3
            self.score_threshold = 0.7
        elif(preset == 'evaluate'):
            self.nms_threshold = 0.3
            self.score_threshold = 0.05
        else:
            raise ValueError('Preset is not visualized or evaluated.')

    def _suppress(self, raw_cls_bbox, raw_prob):
        # Initialization
        bbox = list()
        label = list()
        confidence = list()

        # Background class is at index 0
        for cls_idx in range(1, self.n_class):
            cls_bbox_idx = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, cls_idx, :]
            
            # Fetch probability
            prob_idx = raw_prob[:, cls_idx]
            
            mask = prob_idx > self.score_threshold
            cls_bbox_idx = cls_bbox_idx[mask]
            prob_idx = prob_idx[mask]

            # Update info
            keep = cp.asnumpy(non_maximum_suppression(cp.array(cls_bbox_idx), self.nms_threshold, prob_idx))
            bbox.append(cls_bbox_idx[keep])
            label.append((cls_idx-1) * np.ones((len(keep), )))
            confidence.append(prob_idx[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        confidence = np.concatenate(confidence, axis=0).astype(np.float32)

        return bbox, label, confidence


    def forward(self, x, scale=1.0):
        # forward propagation of faster rcnn
        img_size = x.shape[2: ]
        height = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(height, img_size, scale)
        roi_cls_locs, roi_scores = self.head(height, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices


    # Select the optimizer for backward propagation
    def select_optimizer(self):
        lr = opt.lr
        params = []

        # Get parameters for optimizer
        for key, value in dict(self.named_parameters()).items():
            if (value.requires_grad):
                if ('bias' in key):
                    params += [{'params': [value], 'lr': lr*2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

        # Choose the type of optimizer
        if (opt.use_adam):
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)

        return self.optimizer

    # Adapt learning rate with decay value
    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr']*decay
        return self.optimizer


    @nograd
    # Predict possible objects in each image
    def predict(self, imgs,sizes=None,visualize=False):
        self.eval()

        # Enable visualization
        if(visualize):
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()

            for img in imgs:
                size = img.shape[1:]
                img = preprocess(array_tool.tonumpy(img))
                sizes.append(size)
                prepared_imgs.append(img)
        else:
             prepared_imgs = imgs

        bboxes = list()
        labels = list()
        confidences = list()

        for img, size in zip(prepared_imgs, sizes):
            img = array_tool.totensor(img[None]).float()
            scale = img.shape[3]/size[1]

            # Fetch data
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = array_tool.totensor(rois)/scale

            # Get mean and standard deviation
            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc*std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

            cls_bbox = loc2bbox(array_tool.tonumpy(roi).reshape((-1, 4)), array_tool.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = array_tool.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)

            # Clip the bounding boxes
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            prob = array_tool.tonumpy(F.softmax(array_tool.totensor(roi_score), dim=1))
            raw_cls_bbox = array_tool.tonumpy(cls_bbox)
            raw_prob = array_tool.tonumpy(prob)

            # Update info
            bbox, label, confidence = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            confidences.append(confidence)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, confidences




