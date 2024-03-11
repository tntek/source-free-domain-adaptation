import torch
from torch import nn
from .WideResNet import WideResNet

from src.utils import split_up_model


class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """
    def __init__(self, model, arch_name):
        super().__init__()
        if isinstance(model, WideResNet):
            self.nChannels = model.nChannels
        self.arch_name = arch_name
        if arch_name!='vit':
            self.encoder, self.fc = split_up_model(model)
            if isinstance(self.fc, nn.Sequential):
                for module in self.fc.modules():
                    if isinstance(module, nn.Linear):
                        self._num_classes = module.out_features
                        self._output_dim = module.in_features
            elif isinstance(self.fc, nn.Linear):
                self._num_classes = self.fc.out_features
                self._output_dim = self.fc.in_features
            else:
                raise ValueError("Unable to detect output dimensions")
        else:
            self.model=model
            self._num_classes = model.head.out_features
            self._output_dim = model.head.in_features

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        if self.arch_name!='vit':
            feat = self.encoder(x)
            if self.arch_name == 'WideResNet':
                feat = torch.nn.functional.avg_pool2d(feat, 8)
                feat = feat.view(-1, self.nChannels)
            elif self.arch_name == 'vit':
                feat = feat[:, 0]
            feat = torch.flatten(feat, 1)
            logits = self.fc(feat)
        else:
            feat=self.model.forward_features(x)
            feat = self.model.fc_norm(feat[:, 0])
            logits=self.model.head(feat)

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim