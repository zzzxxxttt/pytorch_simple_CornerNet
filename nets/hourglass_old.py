import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpool import TopPool, BottomPool, LeftPool, RightPool

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MyUpsample2(nn.Module):
  def forward(self, x):
    return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2). \
      reshape(x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2)


class pool(nn.Module):
  def __init__(self, dim, pool1, pool2):
    super(pool, self).__init__()
    self.p1_conv1 = convolution(3, dim, 128)
    self.p2_conv1 = convolution(3, dim, 128)

    self.p_conv1 = nn.Conv2d(128, dim, 3, padding=1, bias=False)
    self.p_bn1 = nn.BatchNorm2d(dim)

    self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)

    self.conv2 = convolution(3, dim, dim)

    self.pool1 = pool1()
    self.pool2 = pool2()

  def forward(self, x):
    pool1 = self.pool1(self.p1_conv1(x))
    pool2 = self.pool2(self.p2_conv1(x))

    p_bn1 = self.p_bn1(self.p_conv1(pool1 + pool2))
    bn1 = self.bn1(self.conv1(x))

    out = self.conv2(F.relu(p_bn1 + bn1, inplace=True))
    return out


class convolution(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()

    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu


class residual(nn.Module):
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()

    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)

    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                              nn.BatchNorm2d(out_dim)) \
      if stride != 1 or inp_dim != out_dim else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ... -> out_dim
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)


# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))


class kp_module(nn.Module):
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()

    self.n = n

    curr_modules = modules[0]
    next_modules = modules[1]

    curr_dim = dims[0]
    next_dim = dims[1]

    # 上支路：重复curr_mod次residual，curr_dim -> curr_dim -> ... -> curr_dim
    self.up1 = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
    # 分辨率本来应该在这里减半...
    self.down = nn.Sequential()
    # 重复curr_mod次residual，curr_dim -> next_dim -> ... -> next_dim
    # 实际上分辨率是在这里的第一个卷积层层降的
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)
    # hourglass中间还是一个hourglass
    # 直到递归结束，重复next_mod次residual，next_dim -> next_dim -> ... -> next_dim
    if self.n > 1:
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
    # 重复curr_mod次residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
    # 分辨率在这里X2
    # self.up = nn.Upsample(scale_factor=2)
    self.up=MyUpsample2()

  def forward(self, x):
    up1 = self.up1(x)  # 上支路residual
    down = self.down(x)  # 下支路downsample(并没有)
    low1 = self.low1(down)  # 下支路residual
    low2 = self.low2(low1)  # 下支路hourglass
    low3 = self.low3(low2)  # 下支路residual
    up2 = self.up(low3)  # 下支路upsample
    return up1 + up2  # 合并上下支路


class exkp(nn.Module):
  def __init__(self, n, nstack, dims, modules, num_classes=80, cnv_dim=256):
    super(exkp, self).__init__()

    self.nstack = nstack

    curr_dim = dims[0]

    self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))

    self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])

    self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

    self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

    self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                nn.BatchNorm2d(curr_dim))
                                  for _ in range(nstack - 1)])
    self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])

    self.tl_cnvs = nn.ModuleList([pool(cnv_dim, TopPool, LeftPool) for _ in range(nstack)])
    self.br_cnvs = nn.ModuleList([pool(cnv_dim, BottomPool, RightPool) for _ in range(nstack)])

    # heatmap layers
    self.tl_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
    self.br_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])

    # embedding layers
    self.tl_tags = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
    self.br_tags = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])

    for hmap_tl, hmap_br in zip(self.tl_heats, self.br_heats):
      hmap_tl[-1].bias.data.fill_(-2.19)
      hmap_br[-1].bias.data.fill_(-2.19)

    # regression layers
    self.tl_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
    self.br_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

    self.relu = nn.ReLU(inplace=True)

  def forward(self, inputs):
    inter = self.pre(inputs)

    outs = []
    for ind in range(self.nstack):
      kp = self.kps[ind](inter)
      cnv = self.cnvs[ind](kp)

      if self.training or ind == self.nstack - 1:
        cnv_tl = self.tl_cnvs[ind](cnv)
        cnv_br = self.br_cnvs[ind](cnv)

        hmap_tl, hmap_br = self.tl_heats[ind](cnv_tl), self.br_heats[ind](cnv_br)
        embd_tl, embd_br = self.tl_tags[ind](cnv_tl), self.br_tags[ind](cnv_br)
        regs_tl, regs_br = self.tl_regrs[ind](cnv_tl), self.br_regrs[ind](cnv_br)

        outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])

      if ind < self.nstack - 1:
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
        inter = self.relu(inter)
        inter = self.inters[ind](inter)
    return outs


def large_hourglass():
  return exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4])


def small_hourglass():
  return exkp(n=5, nstack=1, dims=[256, 128, 256, 256, 256, 384], modules=[2, 2, 2, 2, 2, 4])


if __name__ == '__main__':
  import time
  import pickle
  from collections import OrderedDict
  from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature
  from utils.losses import _neg_loss, _tranpose_and_gather_feature, _reg_loss, _ae_loss
  import os

  # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  # os.environ["CUDA_VISIBLE_DEVICES"] = '0'


  class Loss(nn.Module):
    def __init__(self, model):
      super(Loss, self).__init__()
      self.model = model

    def forward(self, batch):
      y = self.model(batch['xs'][0].cuda())

      hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*y)

      embd_tl = [_tranpose_and_gather_feature(e, batch['xs'][1].cuda()) for e in embd_tl]
      embd_br = [_tranpose_and_gather_feature(e, batch['xs'][2].cuda()) for e in embd_br]
      regs_tl = [_tranpose_and_gather_feature(r, batch['xs'][1].cuda()) for r in regs_tl]
      regs_br = [_tranpose_and_gather_feature(r, batch['xs'][2].cuda()) for r in regs_br]

      focal_loss = _neg_loss(hmap_tl, batch['ys'][0].cuda()) + \
                   _neg_loss(hmap_br, batch['ys'][1].cuda())
      reg_loss = _reg_loss(regs_tl, batch['ys'][3].cuda(), batch['ys'][2].cuda()) + \
                 _reg_loss(regs_br, batch['ys'][4].cuda(), batch['ys'][2].cuda())
      pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ys'][2].cuda())

      print(focal_loss * 2, pull_loss * 2, push_loss * 2, reg_loss * 2)
      # loss=0.1*pull_loss
      loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
      return loss.unsqueeze(0)


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = large_hourglass()

  # net.load_state_dict(ckpt)
  net.load_state_dict(torch.load('model_gt.t7'))
  net = nn.DataParallel(Loss(net)).cuda()
  net.train()

  optimizer = torch.optim.Adam(net.parameters(), 2.5e-4)
  data = torch.load('data_gt.t7')

  print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))

  # for m in net.modules():
  #   if isinstance(m, nn.Conv2d):
  #     m.register_forward_hook(hook)

  for k in data:
    for i in range(len(data[k])):
      data[k][i] = data[k][i].cuda()

  # with torch.no_grad():
  loss = net(data)

  print(loss.mean().item())
  print('')
  print('')
  optimizer.zero_grad()
  loss.mean().backward()
  torch.save([(n, v.grad) for n, v in net.named_parameters()], 'grads.t7')
  optimizer.step()
  torch.save(net.module.model.state_dict(), 'model_new.t7')
  # print(y.size())

if __name__ == 'x__main__':
  import time
  import pickle
  import argparse
  from collections import OrderedDict
  from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature
  from utils.losses import _neg_loss, _tranpose_and_gather_feature, _reg_loss, _ae_loss
  import torch.distributed as dist

  # Training settings
  parser = argparse.ArgumentParser(description='cornernet')
  parser.add_argument('--local_rank', type=int, default=0)
  cfg = parser.parse_args()


  class Loss(nn.Module):
    def __init__(self, model):
      super(Loss, self).__init__()
      self.model = model

    def forward(self, batch):
      y = self.model(batch['xs'][0].cuda())

      hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*y)

      embd_tl = [_tranpose_and_gather_feature(e, batch['xs'][1].cuda()) for e in embd_tl]
      embd_br = [_tranpose_and_gather_feature(e, batch['xs'][2].cuda()) for e in embd_br]
      regs_tl = [_tranpose_and_gather_feature(r, batch['xs'][1].cuda()) for r in regs_tl]
      regs_br = [_tranpose_and_gather_feature(r, batch['xs'][2].cuda()) for r in regs_br]

      focal_loss = _neg_loss(hmap_tl, batch['ys'][0].cuda()) + \
                   _neg_loss(hmap_br, batch['ys'][1].cuda())
      reg_loss = _reg_loss(regs_tl, batch['ys'][3].cuda(), batch['ys'][2].cuda()) + \
                 _reg_loss(regs_br, batch['ys'][4].cuda(), batch['ys'][2].cuda())
      pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ys'][2].cuda())

      print(focal_loss * 2, pull_loss * 2, push_loss * 2, reg_loss * 2)
      loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
      return loss.unsqueeze(0)

  num_gpus = torch.cuda.device_count()
  cfg.device = torch.device('cuda:%d' % cfg.local_rank)
  torch.cuda.set_device(cfg.local_rank)
  dist.init_process_group(backend='nccl', init_method='env://',
                          world_size=num_gpus, rank=cfg.local_rank)


  net = large_hourglass()

  # net.load_state_dict(ckpt)
  net.load_state_dict(torch.load('model_gt.t7'))

  net = Loss(net).to(cfg.device)
  net = nn.parallel.DistributedDataParallel(net,
                                            device_ids=[cfg.local_rank, ],
                                            output_device=cfg.local_rank)

  if cfg.local_rank==0:
    data = torch.load('db_gt_8_1.t7')
  else:
    data = torch.load('db_gt_8_2.t7')

  print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))

  # for m in net.modules():
  #   if isinstance(m, nn.Conv2d):
  #     m.register_forward_hook(hook)

  for k in data[0]:
    for i in range(len(data[0][k])):
      data[0][k][i] = data[0][k][i].cuda()

  # with torch.no_grad():
  loss = net(data)

  print(loss.sum())
  # print(y.size())

if __name__ == 'x__main__':
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'

  import time
  import pickle
  import argparse
  from collections import OrderedDict
  from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature
  from utils.losses import _neg_loss, _tranpose_and_gather_feature, _reg_loss, _ae_loss
  import torch.distributed as dist

  # Training settings
  parser = argparse.ArgumentParser(description='cornernet')
  parser.add_argument('--local_rank', type=int, default=0)
  cfg = parser.parse_args()

  num_gpus = torch.cuda.device_count()
  cfg.device = torch.device('cuda:%d' % cfg.local_rank)
  torch.cuda.set_device(cfg.local_rank)
  dist.init_process_group(backend='nccl', init_method='env://',
                          world_size=num_gpus, rank=cfg.local_rank)


  net = large_hourglass()

  # net.load_state_dict(ckpt)
  net.load_state_dict(torch.load('model_gt.t7'))

  net = net.to(cfg.device)
  net = nn.parallel.DistributedDataParallel(net,
                                            device_ids=[cfg.local_rank, ],
                                            output_device=cfg.local_rank)

  net.train()

  optimizer = torch.optim.Adam(net.parameters(), 2.5e-4)

  if cfg.local_rank==0:
    data = torch.load('data1.t7')
  else:
    data = torch.load('data2.t7')

  print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))

  # for m in net.modules():
  #   if isinstance(m, nn.Conv2d):
  #     m.register_forward_hook(hook)

  for k in data:
    for i in range(len(data[k])):
      data[k][i] = data[k][i].cuda()
  print('')

  # with torch.no_grad():
  y = net(data['xs'][0].cuda())

  hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*y)

  embd_tl = [_tranpose_and_gather_feature(e, data['xs'][1].cuda()) for e in embd_tl]
  embd_br = [_tranpose_and_gather_feature(e, data['xs'][2].cuda()) for e in embd_br]
  regs_tl = [_tranpose_and_gather_feature(r, data['xs'][1].cuda()) for r in regs_tl]
  regs_br = [_tranpose_and_gather_feature(r, data['xs'][2].cuda()) for r in regs_br]

  focal_loss = _neg_loss(hmap_tl, data['ys'][0].cuda()) + \
               _neg_loss(hmap_br, data['ys'][1].cuda())
  reg_loss = _reg_loss(regs_tl, data['ys'][3].cuda(), data['ys'][2].cuda()) + \
             _reg_loss(regs_br, data['ys'][4].cuda(), data['ys'][2].cuda())
  pull_loss, push_loss = _ae_loss(embd_tl, embd_br, data['ys'][2].cuda())

  print(focal_loss.item() * 2, pull_loss.item() * 2, push_loss.item() * 2, reg_loss.item() * 2)
  loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss

  print(loss.sum().item())
  print('')
  print('')
  optimizer.zero_grad()
  loss.mean().backward()
  if cfg.local_rank==0:
    torch.save([(n, v.grad) for n, v in net.named_parameters()], 'grads.t7')
  optimizer.step()
  if cfg.local_rank==0:
    torch.save(net.module.state_dict(), 'model_new.t7')

  # print(y.size())
