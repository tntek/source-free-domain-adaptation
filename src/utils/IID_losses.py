import sys

import torch


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  if x_out.dim() == 1:
    x_out = x_out.unsqueeze(0)
    x_tf_out = x_tf_out.unsqueeze(0)
  _, k = x_out.size()

  bn_, k_ = x_out.size()
  assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # 
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # 
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j+1e-5) \
                    - lamb * torch.log(p_j+1e-5) \
                    - lamb * torch.log(p_i+1e-5))

  loss = loss.sum()

  # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
  #                           - torch.log(p_j) \
  #                           - torch.log(p_i))

  # loss_no_lamb = loss_no_lamb.sum()

  return loss


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j