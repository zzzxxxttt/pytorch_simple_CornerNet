import torch
import torch.nn as nn


def count_parameters(model):
  num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
  print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=224):
  flops = []
  handles = []

  def conv_hook(self, input, output):
    flops.append(output.shape[2] ** 2 *
                 self.kernel_size[0] ** 2 *
                 self.in_channels *
                 self.out_channels /
                 self.groups / 1e6)

  def fc_hook(self, input, output):
    flops.append(self.in_features * self.out_features / 1e6)

  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      handles.append(m.register_forward_hook(conv_hook))
    if isinstance(m, nn.Linear):
      handles.append(m.register_forward_hook(fc_hook))

  with torch.no_grad():
    _ = model(torch.randn(1, 3, input_size, input_size))
  print("Total FLOPs = %f M" % sum(flops))
