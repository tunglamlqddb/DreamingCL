import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(BN, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)

        # calculate running estimates
        if bn_training:
            n = input.numel() / input.size(1)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class ReBN(nn.BatchNorm2d):
    def __init__(self, num_features, r=1., momentum=0.1,
                 affine=True, track_running_stats=True):
        super(ReBN, self).__init__(
            num_features, momentum, affine, track_running_stats)
        self.r = r

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)

        # calculate running estimates
        if bn_training:
            n = input.numel() / input.size(1)
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None]).clamp(min=self.r)

        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

    def __repr__(self):
        return f"ReBN({self.num_features}, r={self.r}, momentum={self.momentum}, " \
            f"affine={self.affine}, track_running_stats={self.track_running_stats})"


class GN(nn.GroupNorm):
    def __init__(self, num_channels, group_size=2, eps=1e-05, affine=True):
        num_groups = num_channels // group_size
        super(GN, self).__init__(
            num_groups, num_channels, eps, affine)
        self.group_size = group_size

    def forward(self, input):
        b = input.size(0)
        init_size = input.size()
        input = input.reshape(b, self.num_groups, -1)
        mean = input.mean(2)
        var = input.var(2, unbiased=False)

        input = (input - mean[:, :, None]) / (torch.sqrt(var[:, :, None] + self.eps))

        input = input.reshape(init_size)
        if self.affine:
            if len(init_size) == 2:
                input = input * self.weight[None, :] + self.bias[None, :]
            elif len(init_size) == 4:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

    def __repr__(self):
        return f"GN({self.num_channels}, group_size={self.group_size}, " \
            f"eps={self.eps}, affine={self.affine})"


class ReGN(nn.GroupNorm):
    def __init__(self, num_channels, group_size=2, r=1., affine=True):
        num_groups = num_channels // group_size
        super(ReGN, self).__init__(
            num_groups, num_channels, affine)
        self.r = r
        self.group_size = group_size

    def forward(self, input):
        b = input.size(0)
        init_size = input.size()
        input = input.reshape(b, self.num_groups, -1)
        s = input.size(2)
        mean = input.mean(2)
        var = input.var(2, unbiased=False)

        input = (input - mean[:, :, None]) / torch.sqrt(var[:, :, None]).clamp(min=self.r)

        input = input.reshape(init_size)
        if self.affine:
            if len(init_size) == 2:
                input = input * self.weight[None, :] + self.bias[None, :]
            elif len(init_size) == 4:
                input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        input = input * s / (s - 1)
        return input

    def __repr__(self):
        return f"ReGN({self.num_channels}, group_size={self.group_size}, " \
            f"r={self.r}, affine={self.affine})"


class ReIN(ReGN):
    def __init__(self, num_channels, r=1., affine=True):
        super(ReGN, self).__init__(num_channels, num_groups=1, r=r, affine=affine)

    def __repr__(self):
        return f"ReIN({self.num_channels}, r={self.r}, affine={self.affine})"


class ReLN(ReGN):
    def __init__(self, num_channels, r=1., affine=True):
        super(ReGN, self).__init__(num_channels, num_groups=num_channels, r=r, affine=affine)

    def __repr__(self):
        return f"ReLN({self.num_channels}, r={self.r}, affine={self.affine})"
 

def get_norm_layer(norm_layer=None, **kwargs):
    norm_dict = {
        "bn": BN,
        "gn": partial(GN, **kwargs),
        "rebn": partial(ReBN, **kwargs),
        "regn": partial(ReGN, **kwargs),
        "reln": partial(ReLN, **kwargs),
        "rein": partial(ReIN, **kwargs),
    }

    if norm_layer is None or norm_layer not in norm_dict:
        raise NotImplementedError

    return norm_dict.get(norm_layer)
