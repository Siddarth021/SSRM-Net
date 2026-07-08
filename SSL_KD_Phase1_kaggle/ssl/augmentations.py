import torch
import numpy as np
import random

class RandomMask(object):
    """
    Masks a random continuous sequence segment of 10% to 30% of temporal steps.
    """
    def __init__(self, min_ratio=0.1, max_ratio=0.3):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, x):
        # x is a tensor of shape (B, 12, T) or (12, T)
        # We perform batch-wise or element-wise augmentation
        if len(x.shape) == 3:
            B, C, T = x.shape
            x_aug = x.clone()
            for b in range(B):
                ratio = random.uniform(self.min_ratio, self.max_ratio)
                mask_len = int(T * ratio)
                start = random.randint(0, T - mask_len)
                x_aug[b, :, start:start+mask_len] = 0
            return x_aug
        else:
            C, T = x.shape
            x_aug = x.clone()
            ratio = random.uniform(self.min_ratio, self.max_ratio)
            mask_len = int(T * ratio)
            start = random.randint(0, T - mask_len)
            x_aug[:, start:start+mask_len] = 0
            return x_aug


class GaussianNoise(object):
    """
    Adds Gaussian noise to the ECG sequence.
    """
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class AmplitudeScaling(object):
    """
    Randomly scales amplitude of the leads by a factor between 0.8 and 1.2.
    """
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        if len(x.shape) == 3:
            B, C, T = x.shape
            scales = torch.empty(B, 1, 1).uniform_(self.min_scale, self.max_scale).to(x.device)
            return x * scales
        else:
            scale = random.uniform(self.min_scale, self.max_scale)
            return x * scale


class TimeShift(object):
    """
    Shifts sequence in time dimension by up to ±5%.
    """
    def __init__(self, max_shift_ratio=0.05):
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, x):
        if len(x.shape) == 3:
            B, C, T = x.shape
            x_aug = torch.zeros_like(x)
            for b in range(B):
                shift = int(T * random.uniform(-self.max_shift_ratio, self.max_shift_ratio))
                if shift > 0:
                    x_aug[b, :, shift:] = x[b, :, :T-shift]
                elif shift < 0:
                    x_aug[b, :, :T+shift] = x[b, :, -shift:]
                else:
                    x_aug[b] = x[b]
            return x_aug
        else:
            C, T = x.shape
            x_aug = torch.zeros_like(x)
            shift = int(T * random.uniform(-self.max_shift_ratio, self.max_shift_ratio))
            if shift > 0:
                x_aug[:, shift:] = x[:, :T-shift]
            elif shift < 0:
                x_aug[:, :T+shift] = x[:, -shift:]
            else:
                x_aug = x
            return x_aug


class LeadDropout(object):
    """
    Randomly drops non-critical leads (Lead II at index 1 is kept).
    """
    def __init__(self, drop_prob=0.3):
        self.drop_prob = drop_prob

    def __call__(self, x):
        # x is (B, C, T) or (C, T)
        if len(x.shape) == 3:
            B, C, T = x.shape
            if C <= 1:
                return x
            x_aug = x.clone()
            for b in range(B):
                for c in range(C):
                    if c == 1: # Lead II is critical
                        continue
                    if random.random() < self.drop_prob:
                        x_aug[b, c, :] = 0
            return x_aug
        else:
            C, T = x.shape
            if C <= 1:
                return x
            x_aug = x.clone()
            for c in range(C):
                if c == 1:
                    continue
                if random.random() < self.drop_prob:
                    x_aug[c, :] = 0
            return x_aug


class ComposeSSL(object):
    """
    Composes several SSL augmentations.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
