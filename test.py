import os
import argparse
from datetime import datetime

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data

from datasets.coco import COCO_eval
from datasets.pascal import PascalVOC_eval

from nets.hourglass import get_hourglass
# from nets.resdcn import get_pose_net

from utils.keypoint import _decode, _rescale_dets
from utils.summary import create_logger

from lib.nms.nms import soft_nms, soft_nms_merge

# Training settings
parser = argparse.ArgumentParser(description='cornernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='large_hourglass')

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')

parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--ae_threshold', type=float, default=0.5)
parser.add_argument('--nms_threshold', type=float, default=0.5)
parser.add_argument('--w_exp', type=float, default=10)

parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def main():
  logger = create_logger(save_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = False
  cfg.device = torch.device('cuda')

  print('Setting up data...')
  Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval
  dataset = Dataset_eval(cfg.data_dir, 'val', test_scales=cfg.test_scales, test_flip=cfg.test_flip)
  val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=False, num_workers=1, pin_memory=True,
                                           collate_fn=dataset.collate_fn)

  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=dataset.num_classes)
  else:
    raise NotImplementedError

  model = model.to(cfg.device)
  model.load_state_dict(torch.load(cfg.pretrain_dir))
  print('loaded pretrained model from %s !' % cfg.pretrain_dir)

  print('validation starts at %s' % datetime.now())
  model.eval()
  results = {}
  with torch.no_grad():
    for inputs in val_loader:
      img_id, inputs = inputs[0]

      detections = []
      for scale in inputs:
        inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
        output = model(inputs[scale]['image'])[-1]
        dets = _decode(*output, ae_threshold=cfg.ae_threshold, K=cfg.topk, kernel=3)
        dets = dets.reshape(dets.shape[0], -1, 8).detach().cpu().numpy()
        if dets.shape[0] == 2:
          dets[1, :, [0, 2]] = inputs[scale]['fmap_size'][0, 1] - dets[1, :, [2, 0]]
        dets = dets.reshape(1, -1, 8)

        _rescale_dets(dets, inputs[scale]['ratio'], inputs[scale]['border'], inputs[scale]['size'])
        dets[:, :, 0:4] /= scale
        detections.append(dets)

      detections = np.concatenate(detections, axis=1)[0]
      # reject detections with negative scores
      detections = detections[detections[:, 4] > -1]
      classes = detections[..., -1]

      results[img_id] = {}
      for j in range(dataset.num_classes):
        keep_inds = (classes == j)
        results[img_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        soft_nms_merge(results[img_id][j + 1], Nt=cfg.nms_threshold, method=2, weight_exp=cfg.w_exp)
        # soft_nms(results[img_id][j + 1], Nt=0.5, method=2)
        results[img_id][j + 1] = results[img_id][j + 1][:, 0:5]

      scores = np.hstack([results[img_id][j][:, -1] for j in range(1, dataset.num_classes + 1)])
      if len(scores) > dataset.max_objs:
        kth = len(scores) - dataset.max_objs
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, dataset.num_classes + 1):
          keep_inds = (results[img_id][j][:, -1] >= thresh)
          results[img_id][j] = results[img_id][j][keep_inds]

  eval_results = dataset.run_eval(results, save_dir=cfg.ckpt_dir)
  print(eval_results)
  print('validation ends at %s' % datetime.now())


if __name__ == '__main__':
  main()
