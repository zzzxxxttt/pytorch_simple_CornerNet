import os
import cv2
import json
import math
import pickle
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
import pycocotools.coco as coco

from utils.image import random_crop, crop_image
from utils.image import color_jittering_, lighting_
from utils.image import draw_gaussian, gaussian_radius

VOC_NAMES = ['__background__', "aeroplane", "bicycle", "bird", "boat",
             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
             "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
             "train", "tvmonitor"]

VOC_MEAN = [0.485, 0.456, 0.406]
VOC_STD = [0.229, 0.224, 0.225]

VOC_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
VOC_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                     [-0.5832747, 0.00994535, -0.81221408],
                     [-0.56089297, 0.71832671, 0.41158938]]


class PascalVOC(data.Dataset):
  def __init__(self, data_dir, split, gaussian=True, img_size=384, **kwargs):
    super(PascalVOC, self).__init__()
    self.num_classes = 20
    self.class_names = VOC_NAMES
    self.valid_ids = np.arange(1, 21, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

    self.data_rng = np.random.RandomState(123)
    self.mean = np.array(VOC_MEAN, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(VOC_STD, dtype=np.float32).reshape(1, 1, 3)
    self.eig_val = np.array(VOC_EIGEN_VALUES, dtype=np.float32)
    self.eig_vec = np.array(VOC_EIGEN_VECTORS, dtype=np.float32)

    self.split = split
    self.data_dir = os.path.join(data_dir, 'voc')
    self.img_dir = os.path.join(self.data_dir, 'images')
    _ann_name = {'train': 'trainval0712', 'val': 'test2007'}
    self.annot_path = os.path.join(self.data_dir, 'annotations', 'pascal_%s.json' % _ann_name[split])

    self.max_objs = 100
    self.padding = 32  # 128 for hourglass
    self.down_ratio = 4
    self.img_size = {'h': img_size, 'w': img_size}
    self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian = gaussian
    self.gaussian_iou = 0.7

    print('==> initializing pascal %s data.' % _ann_name[split])
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)
    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    annotations = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=[img_id]))

    labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
    bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([[0]])
    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

    # random crop (for training) or center crop (for validation)
    if self.split == 'train':
      image, bboxes = random_crop(image,
                                  bboxes,
                                  random_scales=self.rand_scales,
                                  new_size=self.img_size,
                                  padding=self.padding)
    else:
      image, border, offset = crop_image(image,
                                         center=[image.shape[0] // 2, image.shape[1] // 2],
                                         new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
      bboxes[:, 0::2] += border[2]
      bboxes[:, 1::2] += border[0]

    # resize image and bbox
    height, width = image.shape[:2]
    image = cv2.resize(image, (self.img_size['w'], self.img_size['h']))
    bboxes[:, 0::2] *= self.img_size['h'] / height
    bboxes[:, 1::2] *= self.img_size['w'] / width

    # discard non-valid bboxes
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_size['w'] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_size['h'] - 1)
    keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
                               (bboxes[:, 3] - bboxes[:, 1]) > 0)
    bboxes = bboxes[keep_inds]
    labels = labels[keep_inds]

    # randomly flip image and bboxes
    if self.split == 'train' and np.random.uniform() > 0.5:
      image[:] = image[:, ::-1, :]
      bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]] - 1

    image = image.astype(np.float32) / 255.

    # randomly change color and lighting
    if self.split == 'train':
      color_jittering_(self.data_rng, image)
      lighting_(self.data_rng, image, 0.1, self.eig_val, self.eig_vec)

    image -= self.mean
    image /= self.std
    image = image.transpose((2, 0, 1))  # from [H, W, C] to [C, H, W]

    hmap_tl = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)
    hmap_br = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)

    regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)
    regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)

    inds_tl = np.zeros((self.max_objs,), dtype=np.int64)
    inds_br = np.zeros((self.max_objs,), dtype=np.int64)

    num_objs = np.array(min(bboxes.shape[0], self.max_objs))
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
    ind_masks[:num_objs] = 1

    for i, ((xtl, ytl, xbr, ybr), label) in enumerate(zip(bboxes, labels)):
      fxtl = (xtl * self.fmap_size['w'] / self.img_size['w'])
      fytl = (ytl * self.fmap_size['h'] / self.img_size['h'])
      fxbr = (xbr * self.fmap_size['w'] / self.img_size['w'])
      fybr = (ybr * self.fmap_size['h'] / self.img_size['h'])

      ixtl = int(fxtl)
      iytl = int(fytl)
      ixbr = int(fxbr)
      iybr = int(fybr)

      if self.gaussian:
        width = xbr - xtl
        height = ybr - ytl

        width = math.ceil(width * self.fmap_size['w'] / self.img_size['w'])
        height = math.ceil(height * self.fmap_size['h'] / self.img_size['h'])

        radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))

        draw_gaussian(hmap_tl[label], [ixtl, iytl], radius)
        draw_gaussian(hmap_br[label], [ixbr, iybr], radius)
      else:
        hmap_tl[label, iytl, ixtl] = 1
        hmap_br[label, iybr, ixbr] = 1

      regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]
      regs_br[i, :] = [fxbr - ixbr, fybr - iybr]
      inds_tl[i] = iytl * self.fmap_size['w'] + ixtl
      inds_br[i] = iybr * self.fmap_size['w'] + ixbr

    return {'image': image,
            'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
            'regs_tl': regs_tl, 'regs_br': regs_br,
            'inds_tl': inds_tl, 'inds_br': inds_br,
            'ind_masks': ind_masks}

  def __len__(self):
    return self.num_samples


class PascalVOC_eval(PascalVOC):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=True):
    super(PascalVOC_eval, self).__init__(data_dir, split)
    self.test_flip = test_flip
    self.test_scales = test_scales
    self.fix_size = fix_size

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      in_height = new_height | 31  # 127 for hourglass
      in_width = new_width | 31  # 127 for hourglass

      fmap_height, fmap_width = (in_height + 1) // self.down_ratio, (in_width + 1) // self.down_ratio
      height_ratio = fmap_height / in_height
      width_ratio = fmap_width / in_width

      resized_image = cv2.resize(image, (new_width, new_height))
      resized_image, border, offset = crop_image(image=resized_image,
                                                 center=[new_height // 2, new_width // 2],
                                                 new_size=[in_height, in_width])

      resized_image = resized_image / 255.
      resized_image -= self.mean
      resized_image /= self.std
      resized_image = resized_image.transpose((2, 0, 1))[None, :, :, :]  # [H, W, C] to [C, H, W]

      if self.test_flip:
        resized_image = np.concatenate((resized_image, resized_image[..., ::-1].copy()), axis=0)

      out[scale] = {'image': resized_image,
                    'border': border,
                    'size': [new_height, new_width],
                    'fmap_size': [fmap_height, fmap_width],
                    'ratio': [height_ratio, width_ratio]}

    return img_id, out

  def convert_eval_format(self, all_bboxes):
    # all_bboxes: num_samples x num_classes x 5
    detections = [[] for _ in self.class_names[1:]]
    for i in range(self.num_samples):
      img_id = self.images[i]
      img_name = self.coco.loadImgs(ids=[img_id])[0]['file_name'].split('.')[0]
      for j in range(1, self.num_classes + 1):
        if len(all_bboxes[img_id][j]) > 0:
          for bbox in all_bboxes[img_id][j]:
            detections[j - 1].append((img_name, bbox[-1], *bbox[:-1]))
    detections = {cls: det for cls, det in zip(self.class_names[1:], detections)}
    return detections

  def run_eval(self, results, save_dir=None):
    detections = self.convert_eval_format(results)
    if save_dir is not None:
      torch.save(detections, os.path.join(save_dir, 'results.t7'))
    eval_map = eval_mAP(os.path.join(self.data_dir, 'VOCdevkit'))
    aps, map = eval_map.do_python_eval(detections)
    return map, aps

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k])[None, ...] for k in sample[s]} for s in sample}))
    return out


class eval_mAP:
  def __init__(self, VOC_test_root, YEAR='2007', set='test'):
    self.VOC_root = VOC_test_root
    self.YEAR = YEAR
    self.set_type = set
    self.annopath = os.path.join(VOC_test_root, 'VOC2007', 'Annotations', '{:s}.xml')
    self.imgpath = os.path.join(VOC_test_root, 'VOC2007', 'JPEGImages', '%s.jpg')
    self.imgsetpath = os.path.join(VOC_test_root, 'VOC2007', 'ImageSets', 'Main', '%s.txt')
    self.devkit_path = os.path.join(VOC_test_root, 'VOC' + YEAR)

  def parse_record(self, filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
      obj_struct = {}
      obj_struct['name'] = obj.find('name').text
      obj_struct['pose'] = obj.find('pose').text
      obj_struct['truncated'] = int(obj.find('truncated').text)
      obj_struct['difficult'] = int(obj.find('difficult').text)
      bbox = obj.find('bndbox')
      obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                            int(bbox.find('ymin').text) - 1,
                            int(bbox.find('xmax').text) - 1,
                            int(bbox.find('ymax').text) - 1]
      objects.append(obj_struct)

    return objects

  def do_python_eval(self, detections, use_07=True):
    cachedir = os.path.join(self.devkit_path, 'annotations_cache')

    aps = []
    # The PASCAL VOC metric changed in 2010
    print('use VOC07 metric ' if use_07 else 'use VOC12 metric ')

    for i, cls in enumerate(VOC_NAMES[1:]):
      rec, prec, ap = self.voc_eval(detections[cls], self.annopath,
                                    self.imgsetpath % self.set_type,
                                    cls, cachedir, ovthresh=0.5, use_07_metric=use_07)
      aps += [ap]
      print('AP for %s = %.2f%%' % (cls, ap * 100))

    print('Mean AP = %.2f%%' % (np.mean(aps) * 100))
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    return aps, np.mean(aps)

  def voc_ap(self, recall, precision, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
      # 11 point metric
      ap = 0.
      for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
          p = 0
        else:
          p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    else:
      # correct AP calculation
      # first append sentinel values at the end
      mrec = np.concatenate(([0.], recall, [1.]))
      mpre = np.concatenate(([0.], precision, [0.]))

      # compute the precision envelope
      for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

      # to calculate area under PR curve, look for points
      # where X axis (recall) changes value
      i = np.where(mrec[1:] != mrec[:-1])[0]

      # and sum (\Delta recall) * prec
      ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

  def voc_eval(self,
               cls_detections,
               annopath,
               imagesetfile,
               classname,
               cachedir,
               ovthresh=0.5,
               use_07_metric=False,
               use_difficult=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
      os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
      lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
      # load annotations
      recs = {}
      for i, imagename in enumerate(imagenames):
        recs[imagename] = self.parse_record(annopath.format(imagename))
        if i % 100 == 0:
          print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
      # save
      print('Saving cached annotations to {:s}'.format(cachefile))
      with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
    else:
      # load
      with open(cachefile, 'rb') as f:
        try:
          recs = pickle.load(f)
        except:
          recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
      R = [obj for obj in recs[imagename] if obj['name'] == classname]
      bbox = np.array([x['bbox'] for x in R])
      if use_difficult:
        difficult = np.array([False for x in R]).astype(np.bool)
      else:
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
      det = [False] * len(R)
      npos = npos + sum(~difficult)
      class_recs[imagename] = {'bbox': bbox,
                               'difficult': difficult,
                               'det': det}

    # read dets
    image_ids = [x[0] for x in cls_detections]
    confidence = np.array([float(x[1]) for x in cls_detections])
    BB = np.array([[float(z) for z in x[2:]] for x in cls_detections])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
      # sort by confidence
      sorted_ind = np.argsort(-confidence)
      sorted_scores = np.sort(-confidence)
      BB = BB[sorted_ind, :]
      image_ids = [image_ids[x] for x in sorted_ind]

      # go down dets and mark TPs and FPs
      for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
          # compute overlaps
          # intersection
          ixmin = np.maximum(BBGT[:, 0], bb[0])
          iymin = np.maximum(BBGT[:, 1], bb[1])
          ixmax = np.minimum(BBGT[:, 2], bb[2])
          iymax = np.minimum(BBGT[:, 3], bb[3])
          iw = np.maximum(ixmax - ixmin + 1., 0.)
          ih = np.maximum(iymax - iymin + 1., 0.)
          inters = iw * ih

          # union
          uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                 (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                 (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

          overlaps = inters / uni
          ovmax = np.max(overlaps)
          jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
          if not R['difficult'][jmax]:
            if not R['det'][jmax]:
              tp[d] = 1.
              R['det'][jmax] = 1
            else:
              fp[d] = 1.
        else:
          fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = self.voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


if __name__ == '__main__':
  from tqdm import tqdm

  train_dataset = PascalVOC('E:\\voc', 'train')
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2,
                                             shuffle=True, num_workers=0,
                                             pin_memory=True, drop_last=True)

  # for b in tqdm(train_dataset):
  #   pass

  val_dataset = PascalVOC_eval('E:\\voc', 'val')
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=True, num_workers=0,
                                           pin_memory=True, drop_last=True,
                                           collate_fn=val_dataset.collate_fn)

  # for d in tqdm(dataset):
  #   pass

  for b in tqdm(val_loader):
    pass
