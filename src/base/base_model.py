from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
