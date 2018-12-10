from __future__ import absolute_import
import torch
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool
from utils.config import opt


def decom_vgg16():
    model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]

    if(not opt.use_drop):
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # Freeze convolution
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

# Faster R-CNN based on VGG-16
class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()
        rpn = RegionProposalNetwork(512, 512, ratios=ratios, anchor_scales=anchor_scales, feat_stride=self.feat_stride,)
        head = VGG16RoIHead(n_class=n_fg_class+1, roi_size=7, spatial_scale=(1.0/self.feat_stride), classifier=classifier)
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head,)


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = array_tool.totensor(roi_indices).float()
        rois = array_tool.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(x, mean, std, truncated=False):
    if(truncated):
        x.weight.data.normal_().fmod_(2).mul_(std).add_(mean)
    else:
        x.weight.data.normal_(mean, std)
        x.bias.data.zero_()
