import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class enhance_net_nopool(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)

    def forward(self, x_original):
        _, _, h, w = x_original.shape
        max_dim = max(h, w)
        scale_factor = 1.0

        if max_dim > 800:
            scale_factor = 800.0 / max_dim
            x_net = F.interpolate(x_original, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            x_net = x_original

        x1 = self.relu(self.e_conv1(x_net))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if scale_factor != 1.0:
            x_r = F.interpolate(x_r, size=(h, w), mode='bilinear', align_corners=False)

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x_original + r1 * (torch.pow(x_original, 2) - x_original)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        return x + r8 * (torch.pow(x, 2) - x)


class End2EndLowLightModel(nn.Module):
    """Task-driven Zero-DCE + YOLOv7 wrapper for online enhancement."""

    def __init__(self, yolo_model, dce_weights=None, clamp_output=True):
        super().__init__()
        self.dce_net = enhance_net_nopool()
        self.yolo_net = yolo_model
        self.clamp_output = clamp_output
        self.latest_enhanced = None

        if dce_weights:
            dce_weights = Path(dce_weights)
            if dce_weights.is_file():
                state_dict = torch.load(str(dce_weights), map_location='cpu')
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict and hasattr(state_dict['model'], 'state_dict'):
                        state_dict = state_dict['model'].state_dict()
                self.dce_net.load_state_dict(state_dict, strict=True)
                logger.info('Loaded Zero-DCE weights from %s', dce_weights)
            else:
                logger.warning('Zero-DCE weights not found: %s, using random initialization.', dce_weights)

    @property
    def model(self):
        return self.yolo_net.model

    @property
    def stride(self):
        return self.yolo_net.stride

    @property
    def names(self):
        return self.yolo_net.names

    @names.setter
    def names(self, value):
        self.yolo_net.names = value

    @property
    def yaml(self):
        return self.yolo_net.yaml

    def forward(self, x, augment=False, profile=False):
        enhanced_x = self.dce_net(x)
        if self.clamp_output:
            enhanced_x = enhanced_x.clamp(0.0, 1.0)
        self.latest_enhanced = enhanced_x.detach()
        return self.yolo_net(enhanced_x, augment=augment, profile=profile)

    def fuse(self):
        self.yolo_net.fuse()
        return self

    def nms(self, mode=True):
        self.yolo_net.nms(mode=mode)
        return self

    def autoshape(self):
        return self.yolo_net.autoshape()

    def info(self, verbose=False, img_size=640):
        return self.yolo_net.info(verbose=verbose, img_size=img_size)


def is_lowlight_model(model):
    return hasattr(model, 'dce_net') and hasattr(model, 'yolo_net')


def maybe_wrap_with_dce(model, dce_weights):
    if not dce_weights or is_lowlight_model(model):
        return model
    return End2EndLowLightModel(model, dce_weights=dce_weights)
