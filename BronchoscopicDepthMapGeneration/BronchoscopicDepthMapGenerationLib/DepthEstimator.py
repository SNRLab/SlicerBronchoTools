##=========================================================================

#  Program:   Bronchoscopic Depth Map Generation - Slicer Module

#  Copyright (c) Brigham and Women's Hospital. All rights reserved.

#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.

#=========================================================================
# Runs a depth estimation model on single images. This file imports torch, so it is only imported once
# PyTorch is installed. It does not depend on Slicer, so it can also be used from plain Python.
import numpy as np
import torch
import torch.nn.functional as F

from BronchoscopicDepthMapGenerationLib.Networks import NETWORKS

class DepthEstimator:
  def __init__(self, model, weightsPath, device):
    """
    :param model: model description from Resources/Models.json
    :param weightsPath: path of the saved state dict
    :param device: torch device name, e.g. 'cpu' or 'cuda'
    """
    if model['network'] not in NETWORKS:
      raise ValueError(f"Unknown network type: {model['network']}")
    self.model = model
    self.device = torch.device(device)
    # The size (width, height) the model was trained with; images are resized to it
    self.inputSize = tuple(model['inputSize'])
    self.network = NETWORKS[model['network']](model)
    self.network.load_state_dict(torch.load(weightsPath, map_location=self.device, weights_only=True))
    self.network.to(self.device)
    self.network.eval()

  def predict(self, image):
    """Estimates the depth map of an image.

    :param image: 2D float32 array (height, width) with values in [0, 1]
    :return: 2D uint8 array with the same size as the input image
    """
    height, width = image.shape
    modelWidth, modelHeight = self.inputSize
    with torch.inference_mode():
      x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(self.device)
      if (height, width) != (modelHeight, modelWidth):
        x = F.interpolate(x, size=(modelHeight, modelWidth), mode='bilinear', align_corners=False, antialias=True)
      y = self.network(x)
      if (height, width) != (modelHeight, modelWidth):
        y = F.interpolate(y, size=(height, width), mode='bilinear', align_corners=False)
      # Depth is in [0, 1]; the Tanh output can be slightly negative
      y = (y.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
      return y[0, 0].cpu().numpy()
