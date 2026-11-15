##=========================================================================

#  Program:   Bronchoscopic Depth Map Generation - Slicer Module

#  Copyright (c) Brigham and Women's Hospital. All rights reserved.

#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.

#=========================================================================
# Network definitions. This file imports torch, so it is only imported once PyTorch is installed.
# The layer structure must stay exactly the same as when the models were trained, otherwise the saved
# weights (state dicts) no longer load.
import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super().__init__()
    self.block = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features),
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features),
    )

  def forward(self, x):
    return x + self.block(x)

class GeneratorResNet(nn.Module):
  """ResNet generator of the Three Cycle-Consistent GAN (3cGAN) that translates a grayscale bronchoscopic image into
  a depth map of the same size. The output is in the range [-1, 1] (Tanh); depth values are in [0, 1].

  Banach A, King F, Masaki F, Tsukada H, Hata N. Visually Navigated Bronchoscopy using three cycle-Consistent
  generative adversarial network for depth estimation. Medical Image Analysis 73 (2021) 102164.
  https://doi.org/10.1016/j.media.2021.102164"""

  def __init__(self, channels, num_residual_blocks):
    super().__init__()

    # Initial convolution block
    out_features = 64
    model = [
      nn.ReflectionPad2d(channels),
      nn.Conv2d(channels, out_features, 6),
      nn.InstanceNorm2d(out_features),
      nn.ReLU(inplace=True),
    ]
    in_features = out_features

    # Downsampling
    for _ in range(2):
      out_features *= 2
      model += [
        nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True),
      ]
      in_features = out_features

    # Residual blocks
    for _ in range(num_residual_blocks):
      model += [ResidualBlock(out_features)]

    # Upsampling
    for _ in range(2):
      out_features //= 2
      model += [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
        nn.InstanceNorm2d(out_features),
        nn.ReLU(inplace=True),
      ]
      in_features = out_features

    # Output layer
    model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7, padding=2), nn.Tanh()]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)

# Networks that can be referenced by the "network" field of a model in Resources/Models.json
NETWORKS = {
  'GeneratorResNet': lambda model: GeneratorResNet(channels=1, num_residual_blocks=model['residualBlocks']),
}
