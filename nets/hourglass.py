import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.cpool import TopPool, BottomPool, LeftPool, RightPool


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
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)
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
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x):
    up1 = self.top(x)  # 上支路residual
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

    self.cnvs_tl = nn.ModuleList([pool(cnv_dim, TopPool, LeftPool) for _ in range(nstack)])
    self.cnvs_br = nn.ModuleList([pool(cnv_dim, BottomPool, RightPool) for _ in range(nstack)])

    # heatmap layers
    self.hmap_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])
    self.hmap_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])

    # embedding layers
    self.embd_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])
    self.embd_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)])

    for hmap_tl, hmap_br in zip(self.hmap_tl, self.hmap_br):
      hmap_tl[-1].bias.data.fill_(-2.19)
      hmap_br[-1].bias.data.fill_(-2.19)

    # regression layers
    self.regs_tl = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])
    self.regs_br = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])

    self.relu = nn.ReLU(inplace=True)

  def forward(self, inputs):
    inter = self.pre(inputs)

    outs = []
    for ind in range(self.nstack):
      kp = self.kps[ind](inter)
      cnv = self.cnvs[ind](kp)

      if self.training or ind == self.nstack - 1:
        cnv_tl = self.cnvs_tl[ind](cnv)
        cnv_br = self.cnvs_br[ind](cnv)

        hmap_tl, hmap_br = self.hmap_tl[ind](cnv_tl), self.hmap_br[ind](cnv_br)
        embd_tl, embd_br = self.embd_tl[ind](cnv_tl), self.embd_br[ind](cnv_br)
        regs_tl, regs_br = self.regs_tl[ind](cnv_tl), self.regs_br[ind](cnv_br)

        outs.append([hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br])

      if ind < self.nstack - 1:
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
        inter = self.relu(inter)
        inter = self.inters[ind](inter)
    return outs


# tiny hourglass is for f**king debug
get_hourglass = \
  {'large_hourglass':
     exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
   'small_hourglass':
     exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
   'tiny_hourglass':
     exkp(n=5, nstack=1, dims=[256, 128, 256, 256, 256, 384], modules=[2, 2, 2, 2, 2, 4])}

if __name__ == '__main__':
  import time
  import pickle
  from collections import OrderedDict


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_hourglass['tiny_hourglass'].cuda()

  # ckpt = torch.load('./ckpt/pretrain/checkpoint.t7', map_location='cpu')
  # new_ckpt = OrderedDict()
  # for k in ckpt:
  #   if 'up1' in k:
  #     new_ckpt[k.replace('up1', 'top')] = ckpt[k]
  #   elif 'tl_cnvs' in k:
  #     new_ckpt[k.replace('tl_cnvs', 'cnvs_tl')] = ckpt[k]
  #   elif 'br_cnvs' in k:
  #     new_ckpt[k.replace('br_cnvs', 'cnvs_br')] = ckpt[k]
  #   elif 'tl_heats' in k:
  #     new_ckpt[k.replace('tl_heats', 'hmap_tl')] = ckpt[k]
  #   elif 'br_heats' in k:
  #     new_ckpt[k.replace('br_heats', 'hmap_br')] = ckpt[k]
  #   elif 'tl_tags' in k:
  #     new_ckpt[k.replace('tl_tags', 'embd_tl')] = ckpt[k]
  #   elif 'br_tags' in k:
  #     new_ckpt[k.replace('br_tags', 'embd_br')] = ckpt[k]
  #   elif 'tl_regrs' in k:
  #     new_ckpt[k.replace('tl_regrs', 'regs_tl')] = ckpt[k]
  #   elif 'br_regrs' in k:
  #     new_ckpt[k.replace('br_regrs', 'regs_br')] = ckpt[k]
  #   else:
  #     new_ckpt[k] = ckpt[k]
  # torch.save(new_ckpt, './ckpt/pretrain/checkpoint.t7')

  # net.load_state_dict(ckpt)

  print("Total param size = %f MB" % (sum(v.numel() for v in net.parameters()) / 1024 / 1024))

  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.register_forward_hook(hook)

  with torch.no_grad():
    y = net(torch.randn(1, 3, 384, 384).cuda())
  # print(y.size())
