# Copyright (c) Kakao Brain. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# from domainbed.optimizers import get_optimizer
# from domainbed.networks.ur_networks import URFeaturizer
# from domainbed.lib import misc
# from domainbed.algorithms import Algorithm


# class ForwardModel(nn.Module):
#     """Forward model is used to reduce gpu memory usage of SWAD.
#     """
#     def __init__(self, network):
#         super().__init__()
#         self.network = network

#     def forward(self, x):
#         return self.predict(x)

#     def predict(self, x):
#         return self.network(x)


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=False, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps
        # if(init == 0.1):
        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))
        # print(1)
    def forward(self, x):
        return F.softplus(self.b) + self.eps


# def get_shapes(model, input_shape):
#     # get shape of intermediate features
#     with torch.no_grad():
#         dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
#         _, feats = model(dummy, ret_feats=True)
#         shapes = [f.shape for f in feats]

#     return shapes


class MIRO(nn.Module):
    """Mutual-Information Regularization with Oracle"""
    def __init__(self,shape):
        # super().__init__(target_fea,pre_fea)
        super(MIRO, self).__init__()
        # self.pre_featurizer = URFeaturizer(
        #     input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        # )
        # self.featurizer = URFeaturizer(
        #     input_shape, self.hparams, feat_layers=hparams.feat_layers
        # )
        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        # self.ld = hparams.ld

        # # build mean/var encoders
        # shapes = get_shapes(self.pre_featurizer, self.input_shape)
        # tar_shape = target_fea.shape
        # pre_shape = pre_fea.shape
        self.mean_encoders = MeanEncoder(shape)
        self.var_encoders = VarianceEncoder(shape)
        # print(2)
        # # optimizer
        # parameters = [
        #     {"params": self.network.parameters()},
        #     {"params": self.mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        #     {"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        # ]
        # self.optimizer = get_optimizer(
        #     hparams["optimizer"],
        #     parameters,
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams["weight_decay"],
        # )

    def update(self, pre_f, f):
        # all_x = torch.cat(x)
        # all_y = torch.cat(y)
        # feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        # logit = self.classifier(feat)
        # loss = F.cross_entropy(logit, all_y)

        # MIRO
        # with torch.no_grad():
        #     _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.
        # for f, pre_f, mean_enc, var_enc in misc.zip_strict(
        #     inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        # ):
            # mutual information regularization

        # self.mean_encoders = self.mean_encoders
        # self.var_encoders = self.var_encoders
        mean_enc = self.mean_encoders
        var_enc = self.var_encoders
        # var_enc.forward()
        # parm = self.new
        mean = mean_enc(pre_f)
        var = var_enc(pre_f)
        vlb = (mean - f).pow(2).div(var) + var.log()
        reg_loss += vlb.mean() / 2.

        # loss += reg_loss * self.ld

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        return reg_loss,var

    # def predict(self, x):
    #     return self.network(x)

    # def get_forward_model(self):
    #     forward_model = ForwardModel(self.network)
    #     return forward_model
