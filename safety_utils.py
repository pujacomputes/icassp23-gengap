import pdb
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
import argparse
import torch.nn as nn
import sklearn.metrics as sk
if torch.cuda.is_available():
    DEVICE = 'cuda'
else: 
    DEVICE = 'cpu'

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout", 
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur", 
    "lines", "sparkles", "transverse_chromatic_abberation"]

CBAR_CORRUPTIONS_SEV = {
    "caustic_refraction": [2.35, 3.2, 4.9, 6.6, 9.15],
    "inverse_sparkles": [1.0, 2.0, 4.0, 9.0, 10.0],
    "sparkles": [1.0, 2.0, 3.0, 5.0, 6.0],
    "perlin_noise": [4.6, 5.2, 5.8, 7.6, 8.8],
    "blue_noise_sample": [0.8, 1.6, 2.4, 4.0, 5.6],
    "plasma_noise": [4.75, 7.0, 8.5, 9.25, 10.0],
    "checkerboard_cutout": [2.0, 3.0, 4.0, 5.0, 6.0],
    "cocentric_sine_waves": [3.0, 5.0, 8.0, 9.0, 10.0],
    "single_frequency_greyscale": [1.0, 1.5, 2.0, 4.5, 5.0],
    "brownish_noise": [1.0, 2.0, 3.0, 4.0, 5.0],
}
CBAR_CORRUPTIONS_IMAGENET = {
    "caustic_refraction",
    "inverse_sparkles",
    "sparkles",
    "perlin_noise", 
    "blue_noise_sample",
    "plasma_noise",
    "checkerboard_cutout",
    "cocentric_sine_waves",
    "single_frequency_greyscale",
    "brownish_noise"
}

"""
PixMix Augmentation Code!
"""

def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  return (out + 1) / 2

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)
  return out / 2

mixings = [add, multiply]
def pixmix(orig, mixing_pic, preprocess,k, beta, severity,use_all_ops): 
    tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
    if np.random.random() < 0.5:
        mixed = tensorize(augment_input(orig,severity,use_all_ops))
    else:
        mixed = tensorize(orig)
  
    for _ in range(np.random.randint(k + 1)):
        if np.random.random() < 0.5:
            aug_image_copy = tensorize(augment_input(orig,severity,use_all_ops))
        else:
            aug_image_copy = tensorize(mixing_pic)

        mixed_op = np.random.choice(mixings)
        mixed = mixed_op(mixed, aug_image_copy, beta)
        mixed = torch.clip(mixed, 0, 1)
    return normalize(mixed)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess,severity,use_all_ops=False):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess
    self.use_all_ops = use_all_ops
    self.severity = severity

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

def normalize_l2(x):
  """
  Expects x.shape == [N, C, H, W]
  """
  norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
  norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  return x / norm

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

