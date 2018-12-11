from __future__ import  absolute_import
import os
import ipdb
import resource
import cupy as cp
from tqdm import tqdm
import matplotlib

from config import opt
from dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from . import array_tool
from vis_tool import visdom_bbox
from eval_tool import eval_detection_voc

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for i, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if i == test_num:
        	break
    return eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, use_07_metric=True)


def train(**kwargs):
    opt._parse(kwargs)

    data_set = Dataset(opt)
    test_set = TestDataset(opt)
    data_loader = data_.DataLoader(data_set, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = data_.DataLoader(test_set,batch_size=1, num_workers=opt.test_num_workers, shuffle=False, pin_memory=True)
    faster_rcnn = FasterRCNNVGG16()

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if(opt.load_path):
        trainer.load(opt.load_path)
    trainer.vis.text(data_set.db.label_names, win='labels')

    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for i, (img, bbox_, label_, scale) in tqdm(enumerate(data_loader)):
            scale = array_tool.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if((i+1) % opt.plot_every == 0):
                if(os.path.exists(opt.debug_file)):
                    ipdb.set_trace()

                # Plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # Plot groud truth bounding boxes
                ori_img_ = inverse_normalize(array_tool.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_, array_tool.tonumpy(bbox_[0]), array_tool.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # Plot predict bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_, array_tool.tonumpy(_bboxes[0]), array_tool.tonumpy(_labels[0]).reshape(-1), array_tool.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # RPN confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')

                # ROI confusion matrix
                trainer.vis.img('roi_cm', array_tool.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_), str(eval_result['map']), str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if(eval_result['map'] > best_map):
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if(epoch == 9):
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire
    fire.Fire()
