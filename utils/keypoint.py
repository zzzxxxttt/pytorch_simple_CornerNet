import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nms(heat, kernel=1):
  hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep


def _gather_feat(feat, ind, mask=None):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat


def _tranpose_and_gather_feature(feature, ind):
  feature = feature.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] => [B, H, W, C]
  feature = feature.view(feature.size(0), -1, feature.size(3))  # [B, H, W, C] => [B, H x W, C]
  ind = ind[:, :, None].expand(ind.shape[0], ind.shape[1], feature.shape[-1])  # [B, num_obj] => [B, num_obj, C]
  feature = feature.gather(1, ind)  # [B, H x W, C] => [B, num_obj, C]
  return feature


def _topk(score_map, K=20):
  batch, cat, height, width = score_map.size()

  topk_scores, topk_inds = torch.topk(score_map.view(batch, -1), K)

  topk_classes = (topk_inds / (height * width)).int()
  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()
  return topk_scores, topk_inds, topk_classes, topk_ys, topk_xs


def _decode(hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br,
            K, kernel, ae_threshold, num_dets=1000):
  batch, cat, height, width = hmap_tl.shape

  hmap_tl = torch.sigmoid(hmap_tl)
  hmap_br = torch.sigmoid(hmap_br)

  # perform nms on heatmaps
  hmap_tl = _nms(hmap_tl, kernel=kernel)
  hmap_br = _nms(hmap_br, kernel=kernel)

  scores_tl, inds_tl, clses_tl, ys_tl, xs_tl = _topk(hmap_tl, K=K)
  scores_br, inds_br, clses_br, ys_br, xs_br = _topk(hmap_br, K=K)

  xs_tl = xs_tl.view(batch, K, 1).expand(batch, K, K)
  ys_tl = ys_tl.view(batch, K, 1).expand(batch, K, K)
  xs_br = xs_br.view(batch, 1, K).expand(batch, K, K)
  ys_br = ys_br.view(batch, 1, K).expand(batch, K, K)

  if regs_tl is not None and regs_br is not None:
    regs_tl = _tranpose_and_gather_feature(regs_tl, inds_tl)
    regs_br = _tranpose_and_gather_feature(regs_br, inds_br)
    regs_tl = regs_tl.view(batch, K, 1, 2)
    regs_br = regs_br.view(batch, 1, K, 2)

    xs_tl = xs_tl + regs_tl[..., 0]
    ys_tl = ys_tl + regs_tl[..., 1]
    xs_br = xs_br + regs_br[..., 0]
    ys_br = ys_br + regs_br[..., 1]

  # all possible boxes based on top k corners (ignoring class)
  bboxes = torch.stack((xs_tl, ys_tl, xs_br, ys_br), dim=3)

  embd_tl = _tranpose_and_gather_feature(embd_tl, inds_tl)
  embd_br = _tranpose_and_gather_feature(embd_br, inds_br)
  embd_tl = embd_tl.view(batch, K, 1)
  embd_br = embd_br.view(batch, 1, K)
  dists = torch.abs(embd_tl - embd_br)

  scores_tl = scores_tl.view(batch, K, 1).expand(batch, K, K)
  scores_br = scores_br.view(batch, 1, K).expand(batch, K, K)
  scores = (scores_tl + scores_br) / 2

  # reject boxes based on classes
  clses_tl = clses_tl.view(batch, K, 1).expand(batch, K, K)
  clses_br = clses_br.view(batch, 1, K).expand(batch, K, K)
  cls_inds = (clses_tl != clses_br)

  # reject boxes based on distances
  dist_inds = (dists > ae_threshold)

  # reject boxes based on widths and heights
  width_inds = (xs_br < xs_tl)
  height_inds = (ys_br < ys_tl)

  scores[cls_inds] = -1
  scores[dist_inds] = -1
  scores[width_inds] = -1
  scores[height_inds] = -1

  scores = scores.view(batch, -1)
  scores, inds = torch.topk(scores, num_dets)
  scores = scores.unsqueeze(2)

  bboxes = bboxes.view(batch, -1, 4)
  bboxes = _gather_feat(bboxes, inds)

  classes = clses_tl.contiguous().view(batch, -1, 1)
  classes = _gather_feat(classes, inds).float()

  scores_tl = scores_tl.contiguous().view(batch, -1, 1)
  scores_br = scores_br.contiguous().view(batch, -1, 1)
  scores_tl = _gather_feat(scores_tl, inds).float()
  scores_br = _gather_feat(scores_br, inds).float()

  detections = torch.cat([bboxes, scores, scores_tl, scores_br, classes], dim=2)
  return detections


def _rescale_dets(detections, ratios, borders, sizes):
  xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
  xs /= ratios[:, 1][:, None, None]
  ys /= ratios[:, 0][:, None, None]
  xs -= borders[:, 2][:, None, None]
  ys -= borders[:, 0][:, None, None]
  np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
  np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
