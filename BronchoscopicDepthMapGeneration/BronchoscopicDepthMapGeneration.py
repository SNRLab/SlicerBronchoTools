##=========================================================================

#  Program:   Bronchoscopic Depth Map Generation - Slicer Module

#  Copyright (c) Brigham and Women's Hospital. All rights reserved.

#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.

#=========================================================================
import os
import sys
import json
import math
import time
import hashlib
import logging
import importlib
import collections
import numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLModelNode, vtkMRMLLinearTransformNode
from vtk.util import numpy_support

class BronchoscopicDepthMapGeneration(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Bronchoscopic Depth Map Generation"
    self.parent.categories = ["Endoscopy"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King (Brigham and Women's Hospital)", "Artur Banach (Brigham and Women's Hospital)"]
    self.parent.helpText = """
    Estimates a depth map from a 2D bronchoscopic image using a deep learning model. It is intended for monoscopic
    (single camera) bronchoscope footage, so depth is estimated from one image at a time. It can process a single
    image, or run continuously on an image that is updated live, e.g. by the Video Input module. The depth map can be
    converted into a point cloud using the camera intrinsics.
"""
    self.parent.acknowledgementText = """This module was developed by Franklin King (Brigham and Women's Hospital).
    The depth estimation model is described in: Banach A, King F, Masaki F, Tsukada H, Hata N. Visually Navigated
    Bronchoscopy using three cycle-Consistent generative adversarial network for depth estimation. Medical Image
    Analysis 73 (2021) 102164. <a href="https://doi.org/10.1016/j.media.2021.102164">doi:10.1016/j.media.2021.102164</a>.
    The model weights are licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>."""
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

# ------------------------------------ Parameter Node ---------------------------------------
@parameterNodeWrapper
class BronchoscopicDepthMapGenerationParameterNode:
  # Input / Output
  # Empty or unknown ids (e.g. a model renamed or removed since the scene was saved) select the first model
  modelId: str = ''
  inputVolume: vtkMRMLScalarVolumeNode | None
  outputVolume: vtkMRMLScalarVolumeNode | None

  # Parameters
  # Upper limit on how often the model runs while running continuously, as the model can be expensive to run
  maxFrameRate: float = 10.0
  useGPU: bool = True

  # Point Cloud Generation (from the depth map in the output volume)
  # When enabled, the point cloud is regenerated with each depth map and when these settings change
  enablePointCloud: bool = False
  pointCloudModel: vtkMRMLModelNode | None
  # Optional camera pose: the point cloud is placed under this transform
  cameraTransform: vtkMRMLLinearTransformNode | None
  # Camera intrinsics, in pixels of the depth map
  focalLengthX: float = 200.0
  focalLengthY: float = 200.0
  principalPointAtCenter: bool = True
  principalPointX: float = 100.0
  principalPointY: float = 100.0
  # Size (mm) of the point cloud along each axis, for the farthest possible depth (100%). Equal values keep its shape.
  scaleX: float = 100.0
  scaleY: float = 100.0
  scaleZ: float = 100.0
  # Points outside this depth range, in percent of the farthest possible depth, are left out
  minimumDepth: float = 0.0
  maximumDepth: float = 60.0
  # Percentage of the depth map pixels that become points
  pixelDensity: float = 100.0
  colorPointCloud: bool = True
# -------------------------------------- ----------  --------------------------------------

class WrappingLabel(qt.QLabel):
  def __init__(self, text=""):
    qt.QLabel.__init__(self, text)
    self.wordWrap = True
    self.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)

  def setText(self, text):
    qt.QLabel.setText(self, text)
    self.updateHeight()

  def resizeEvent(self, event):
    qt.QLabel.resizeEvent(self, event)
    self.updateHeight()

  def updateHeight(self):
    # Measure the text at the current width. QLabel.heightForWidth is not used, as it can return a height
    # calculated for an earlier width.
    height = 0
    if self.text:
      document = qt.QTextDocument()
      document.setDefaultFont(self.font)
      document.setDocumentMargin(0)
      if qt.Qt.mightBeRichText(self.text):
        document.setHtml(self.text)
      else:
        document.setPlainText(self.text)
      margins = self.contentsMargins()
      document.setTextWidth(max(self.width - margins.left() - margins.right(), 1))
      height = int(math.ceil(document.size.height())) + margins.top() + margins.bottom()
    if self.minimumHeight != height or self.maximumHeight != height:
      self.setFixedHeight(height)

class BronchoscopicDepthMapGenerationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  START_BUTTON_STYLE = "QPushButton {background-color: #2E7D32; color: white;} QPushButton:hover {background-color: #388E3C;}"
  STOP_BUTTON_STYLE = "QPushButton {background-color: #F9A825; color: black;} QPushButton:hover {background-color: #FBC02D;}"

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._parameterNodeGuiTag = None
    self._observedInputVolume = None
    self._lastPointCloudSettings = None
    self._pointCloudUpdatePending = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = BronchoscopicDepthMapGenerationLogic()
    self.logic.statusCallback = self.onLogicStatusChanged

    # ------------------------------------ IO ---------------------------------------
    self.inputOutputCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputOutputCollapsibleButton.text = "Input / Output"
    self.inputOutputCollapsibleButton.collapsed = False
    self.layout.addWidget(self.inputOutputCollapsibleButton)
    inputOutputLayout = qt.QFormLayout(self.inputOutputCollapsibleButton)

    self.modelComboBox = qt.QComboBox()
    self.modelComboBox.setToolTip("Depth estimation model")
    for model in self.logic.models:
      self.modelComboBox.addItem(model['title'], model['id'])
      self.modelComboBox.setItemData(self.modelComboBox.count - 1, model['description'], qt.Qt.ToolTipRole)
    self.modelComboBox.connect('activated(int)', self.onModelSelected)

    self.openModelsFileButton = qt.QToolButton()
    self.openModelsFileButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_DirOpenIcon))
    self.openModelsFileButton.setToolTip("Open the list of models (Models.json) in the default application for JSON files. "
                                         "Changes take effect after reloading the module or restarting Slicer.")
    self.openModelsFileButton.connect('clicked()', self.onOpenModelsFileButtonClicked)

    modelLayout = qt.QHBoxLayout()
    modelLayout.addWidget(self.modelComboBox, 1)
    modelLayout.addWidget(self.openModelsFileButton)
    inputOutputLayout.addRow("Model:", modelLayout)

    # Read-only box that scrolls, so long descriptions (e.g. with a citation) do not take up much space
    self.modelDescriptionTextBrowser = qt.QTextBrowser()
    self.modelDescriptionTextBrowser.setToolTip("Description of the selected model")
    self.modelDescriptionTextBrowser.setOpenExternalLinks(True)
    self.modelDescriptionTextBrowser.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
    self.modelDescriptionTextBrowser.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    # About 4 lines of text
    lineSpacing = qt.QFontMetrics(self.modelDescriptionTextBrowser.font).lineSpacing()
    documentMargin = int(self.modelDescriptionTextBrowser.document.documentMargin)
    self.modelDescriptionTextBrowser.setFixedHeight(4 * lineSpacing + 2 * documentMargin + 2 * self.modelDescriptionTextBrowser.frameWidth)
    inputOutputLayout.addRow("Description:", self.modelDescriptionTextBrowser)

    self.inputVolumeSelector = slicer.qMRMLNodeComboBox()
    self.inputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputVolumeSelector.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.inputVolumeSelector.selectNodeUponCreation = True
    self.inputVolumeSelector.addEnabled = False
    self.inputVolumeSelector.removeEnabled = False
    self.inputVolumeSelector.noneEnabled = True
    self.inputVolumeSelector.showHidden = False
    self.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.inputVolumeSelector.setToolTip("2D image (a volume with a single slice) to estimate depth from, e.g. the "
                                        "output of the Video Input module. Color images are converted to grayscale.")
    inputOutputLayout.addRow("Input Volume:", self.inputVolumeSelector)

    self.inputInfoLabel = WrappingLabel()
    inputOutputLayout.addRow("", self.inputInfoLabel)

    self.outputVolumeSelector = slicer.qMRMLNodeComboBox()
    self.outputVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputVolumeSelector.showChildNodeTypes = False
    self.outputVolumeSelector.selectNodeUponCreation = True
    self.outputVolumeSelector.addEnabled = True
    self.outputVolumeSelector.removeEnabled = True
    self.outputVolumeSelector.renameEnabled = True
    self.outputVolumeSelector.noneEnabled = True
    self.outputVolumeSelector.showHidden = False
    self.outputVolumeSelector.baseName = "DepthMap"
    self.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputVolumeSelector.setToolTip("Volume that receives the depth map: an 8-bit grayscale image with the same size "
                                         "and position as the input. If none is selected, one is created.")
    inputOutputLayout.addRow("Output Volume:", self.outputVolumeSelector)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Run UI ---------------------------------------
    runFont = qt.QFont()
    runFont.setPointSize(12)
    self.applyButton = qt.QPushButton("Generate Depth Map")
    self.applyButton.setToolTip("Generate the depth map of the current input image once")
    self.applyButton.setFont(runFont)
    self.applyButton.setMinimumHeight(40)
    self.applyButton.connect('clicked()', self.onApplyButtonClicked)

    self.continuousButton = qt.QPushButton()
    self.continuousButton.setCheckable(True)
    self.continuousButton.setToolTip("Generate the depth map whenever the input image changes, at most at the maximum frame rate")
    self.continuousButton.setFont(runFont)
    self.continuousButton.setMinimumHeight(40)
    self.continuousButton.connect('clicked()', self.onContinuousButtonClicked)

    runLayout = qt.QHBoxLayout()
    runLayout.addWidget(self.applyButton)
    runLayout.addWidget(self.continuousButton)
    self.layout.addLayout(runLayout)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Parameters UI ---------------------------------------
    self.parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    self.parametersCollapsibleButton.text = "Depth Map Parameters"
    self.parametersCollapsibleButton.collapsed = False
    self.layout.addWidget(self.parametersCollapsibleButton)
    parametersLayout = qt.QFormLayout(self.parametersCollapsibleButton)

    self.maxFrameRateSliderWidget = ctk.ctkSliderWidget()
    self.maxFrameRateSliderWidget.setToolTip("Maximum number of depth maps generated per second while running continuously. "
                                             "Lower it to reduce the load on the computer. The actual rate is also limited "
                                             "by how often the input changes and how fast the model runs.")
    self.maxFrameRateSliderWidget.setDecimals(1)
    self.maxFrameRateSliderWidget.minimum = 0.5
    self.maxFrameRateSliderWidget.maximum = 60.0
    self.maxFrameRateSliderWidget.singleStep = 0.5
    self.maxFrameRateSliderWidget.suffix = " fps"
    parametersLayout.addRow("Maximum Frame Rate:", self.maxFrameRateSliderWidget)

    self.useGPUCheckBox = qt.QCheckBox("Use GPU if available")
    self.useGPUCheckBox.setToolTip("Run the model on a GPU if a supported one is available")
    parametersLayout.addRow(self.useGPUCheckBox)

    self.deviceLabel = qt.QLabel()
    self.deviceLabel.setToolTip("Device the model runs on")
    parametersLayout.addRow("Device:", self.deviceLabel)

    self.inferenceTimeLabel = qt.QLabel()
    self.inferenceTimeLabel.setToolTip("Time taken to generate the most recent depth map")
    parametersLayout.addRow("Inference Time:", self.inferenceTimeLabel)

    self.frameRateLabel = qt.QLabel()
    self.frameRateLabel.setToolTip("Depth maps generated per second while running continuously")
    parametersLayout.addRow("Frame Rate:", self.frameRateLabel)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Point Cloud UI ---------------------------------------
    self.pointCloudCollapsibleButton = ctk.ctkCollapsibleButton()
    self.pointCloudCollapsibleButton.text = "Point Cloud Generation"
    self.pointCloudCollapsibleButton.collapsed = False
    self.layout.addWidget(self.pointCloudCollapsibleButton)
    pointCloudLayout = qt.QFormLayout(self.pointCloudCollapsibleButton)

    self.enablePointCloudCheckBox = qt.QCheckBox("Enable Point Cloud")
    self.enablePointCloudCheckBox.setToolTip("Generate a point cloud from the depth map in the output volume. It is "
                                             "regenerated with each depth map (including while running continuously) "
                                             "and when the settings below change.")
    pointCloudLayout.addRow(self.enablePointCloudCheckBox)

    self.pointCloudModelSelector = slicer.qMRMLNodeComboBox()
    self.pointCloudModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.pointCloudModelSelector.showChildNodeTypes = False
    self.pointCloudModelSelector.selectNodeUponCreation = True
    self.pointCloudModelSelector.addEnabled = True
    self.pointCloudModelSelector.removeEnabled = True
    self.pointCloudModelSelector.renameEnabled = True
    self.pointCloudModelSelector.noneEnabled = True
    self.pointCloudModelSelector.showHidden = False
    self.pointCloudModelSelector.baseName = "PointCloud"
    self.pointCloudModelSelector.setMRMLScene(slicer.mrmlScene)
    self.pointCloudModelSelector.setToolTip("Model that receives the point cloud, generated from the depth map in the "
                                            "output volume. If none is selected, one is created.")
    pointCloudLayout.addRow("Point Cloud:", self.pointCloudModelSelector)

    self.cameraTransformSelector = slicer.qMRMLNodeComboBox()
    self.cameraTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.cameraTransformSelector.selectNodeUponCreation = True
    self.cameraTransformSelector.addEnabled = True
    self.cameraTransformSelector.removeEnabled = False
    self.cameraTransformSelector.renameEnabled = True
    self.cameraTransformSelector.noneEnabled = True
    self.cameraTransformSelector.noneDisplay = "None"
    self.cameraTransformSelector.showHidden = False
    self.cameraTransformSelector.baseName = "CameraPose"
    self.cameraTransformSelector.setMRMLScene(slicer.mrmlScene)
    self.cameraTransformSelector.setToolTip("Optional camera pose. The point cloud is placed under this transform.")
    pointCloudLayout.addRow("Camera Transform:", self.cameraTransformSelector)

    self.intrinsicsGroupBox = ctk.ctkCollapsibleGroupBox()
    self.intrinsicsGroupBox.title = "Camera Intrinsics"
    intrinsicsLayout = qt.QFormLayout(self.intrinsicsGroupBox)
    pointCloudLayout.addRow(self.intrinsicsGroupBox)

    self.focalLengthXSpinBox = ctk.ctkDoubleSpinBox()
    self.focalLengthXSpinBox.setToolTip("Horizontal focal length (fx), in pixels of the depth map.")
    self.focalLengthXSpinBox.decimals = 2
    self.focalLengthXSpinBox.minimum = 0.01
    self.focalLengthXSpinBox.maximum = 100000.0
    self.focalLengthXSpinBox.singleStep = 1.0
    self.focalLengthXSpinBox.suffix = " px"
    intrinsicsLayout.addRow("Focal Length X (fx):", self.focalLengthXSpinBox)

    self.focalLengthYSpinBox = ctk.ctkDoubleSpinBox()
    self.focalLengthYSpinBox.setToolTip("Vertical focal length (fy), in pixels of the depth map.")
    self.focalLengthYSpinBox.decimals = 2
    self.focalLengthYSpinBox.minimum = 0.01
    self.focalLengthYSpinBox.maximum = 100000.0
    self.focalLengthYSpinBox.singleStep = 1.0
    self.focalLengthYSpinBox.suffix = " px"
    intrinsicsLayout.addRow("Focal Length Y (fy):", self.focalLengthYSpinBox)

    self.principalPointAtCenterCheckBox = qt.QCheckBox("Principal point at image center")
    self.principalPointAtCenterCheckBox.setToolTip("Use the center of the depth map as the principal point instead of the values below")
    intrinsicsLayout.addRow(self.principalPointAtCenterCheckBox)

    self.principalPointXSpinBox = ctk.ctkDoubleSpinBox()
    self.principalPointXSpinBox.setToolTip("Column of the principal point (cx), in pixels from the left edge of the depth map")
    self.principalPointXSpinBox.decimals = 2
    self.principalPointXSpinBox.minimum = -100000.0
    self.principalPointXSpinBox.maximum = 100000.0
    self.principalPointXSpinBox.singleStep = 1.0
    self.principalPointXSpinBox.suffix = " px"
    intrinsicsLayout.addRow("Principal Point X (cx):", self.principalPointXSpinBox)

    self.principalPointYSpinBox = ctk.ctkDoubleSpinBox()
    self.principalPointYSpinBox.setToolTip("Row of the principal point (cy), in pixels from the top edge of the depth map")
    self.principalPointYSpinBox.decimals = 2
    self.principalPointYSpinBox.minimum = -100000.0
    self.principalPointYSpinBox.maximum = 100000.0
    self.principalPointYSpinBox.singleStep = 1.0
    self.principalPointYSpinBox.suffix = " px"
    intrinsicsLayout.addRow("Principal Point Y (cy):", self.principalPointYSpinBox)

    # Depths from the model are relative, not true distances, so the size of the point cloud is set by these scales.
    # Each is the size along its axis for the farthest possible depth (100%, white in the depth map).
    self.scaleGroupBox = ctk.ctkCollapsibleGroupBox()
    self.scaleGroupBox.title = "Scale"
    scaleLayout = qt.QFormLayout(self.scaleGroupBox)
    pointCloudLayout.addRow(self.scaleGroupBox)

    self.scaleXSpinBox = ctk.ctkDoubleSpinBox()
    self.scaleXSpinBox.setToolTip("Size of the point cloud along X (toward the bottom of the image) for the farthest "
                                  "possible depth. Use the same value for all axes to keep the shape of the point cloud.")
    self.scaleXSpinBox.decimals = 1
    self.scaleXSpinBox.minimum = 0.1
    self.scaleXSpinBox.maximum = 10000.0
    self.scaleXSpinBox.singleStep = 1.0
    self.scaleXSpinBox.suffix = " mm"
    scaleLayout.addRow("X (image vertical):", self.scaleXSpinBox)

    self.scaleYSpinBox = ctk.ctkDoubleSpinBox()
    self.scaleYSpinBox.setToolTip("Size of the point cloud along Y (toward the left of the image) for the farthest "
                                  "possible depth. Use the same value for all axes to keep the shape of the point cloud.")
    self.scaleYSpinBox.decimals = 1
    self.scaleYSpinBox.minimum = 0.1
    self.scaleYSpinBox.maximum = 10000.0
    self.scaleYSpinBox.singleStep = 1.0
    self.scaleYSpinBox.suffix = " mm"
    scaleLayout.addRow("Y (image horizontal):", self.scaleYSpinBox)

    self.scaleZSpinBox = ctk.ctkDoubleSpinBox()
    self.scaleZSpinBox.setToolTip("Distance along Z (the viewing direction) of the farthest possible depth. Use the "
                                  "same value for all axes to keep the shape of the point cloud.")
    self.scaleZSpinBox.decimals = 1
    self.scaleZSpinBox.minimum = 0.1
    self.scaleZSpinBox.maximum = 10000.0
    self.scaleZSpinBox.singleStep = 1.0
    self.scaleZSpinBox.suffix = " mm"
    scaleLayout.addRow("Z (depth):", self.scaleZSpinBox)

    self.depthRangeWidget = ctk.ctkRangeWidget()
    self.depthRangeWidget.setToolTip("Only pixels with a depth in this range become points, in percent of the farthest "
                                     "possible depth. Lower the maximum to leave out the far parts of the airway, where "
                                     "the depth is least accurate. Pixels with zero depth are always left out.")
    self.depthRangeWidget.decimals = 0
    self.depthRangeWidget.minimum = 0.0
    self.depthRangeWidget.maximum = 100.0
    self.depthRangeWidget.singleStep = 1.0
    self.depthRangeWidget.suffix = "%"
    self.depthRangeWidget.connect('valuesChanged(double,double)', self.onDepthRangeChanged)
    pointCloudLayout.addRow("Depth Range:", self.depthRangeWidget)

    self.pixelDensitySliderWidget = ctk.ctkSliderWidget()
    self.pixelDensitySliderWidget.setToolTip("Percentage of the depth map pixels that become points, spread evenly over "
                                             "the image. Lower it to reduce the number of points.")
    self.pixelDensitySliderWidget.decimals = 0
    self.pixelDensitySliderWidget.minimum = 1.0
    self.pixelDensitySliderWidget.maximum = 100.0
    self.pixelDensitySliderWidget.singleStep = 1.0
    self.pixelDensitySliderWidget.suffix = "%"
    pointCloudLayout.addRow("Pixel Density:", self.pixelDensitySliderWidget)

    self.colorPointCloudCheckBox = qt.QCheckBox("Color points from the input image")
    self.colorPointCloudCheckBox.setToolTip("Color each point with its pixel in the input image, if the input image has "
                                            "the same size as the depth map. Otherwise the points have a single color.")
    pointCloudLayout.addRow(self.colorPointCloudCheckBox)

    self.pointCloudInfoLabel = qt.QLabel()
    self.pointCloudInfoLabel.setToolTip("Number of points in the most recent point cloud, and the time taken to generate it")
    pointCloudLayout.addRow("Points:", self.pointCloudInfoLabel)
    # -------------------------------------- ----------  --------------------------------------

    self.layout.addStretch(1)

    # Widgets bound directly to parameter node values
    self.parameterWidgets = {
      'inputVolume': self.inputVolumeSelector,
      'outputVolume': self.outputVolumeSelector,
      'useGPU': self.useGPUCheckBox,
      'pointCloudModel': self.pointCloudModelSelector,
      'cameraTransform': self.cameraTransformSelector,
      'enablePointCloud': self.enablePointCloudCheckBox,
      'principalPointAtCenter': self.principalPointAtCenterCheckBox,
      'colorPointCloud': self.colorPointCloudCheckBox,
    }
    # Widgets with ranges set above are synced manually, since binding them would replace their ranges
    self.valueWidgets = {
      'maxFrameRate': self.maxFrameRateSliderWidget,
      'focalLengthX': self.focalLengthXSpinBox,
      'focalLengthY': self.focalLengthYSpinBox,
      'principalPointX': self.principalPointXSpinBox,
      'principalPointY': self.principalPointYSpinBox,
      'scaleX': self.scaleXSpinBox,
      'scaleY': self.scaleYSpinBox,
      'scaleZ': self.scaleZSpinBox,
      'pixelDensity': self.pixelDensitySliderWidget,
    }
    # Point cloud settings, which are disabled while the point cloud is not enabled
    self.pointCloudSettingWidgets = [
      self.pointCloudModelSelector, self.cameraTransformSelector, self.intrinsicsGroupBox, self.scaleGroupBox,
      self.depthRangeWidget, self.pixelDensitySliderWidget, self.colorPointCloudCheckBox,
    ]
    for name, widget in self.valueWidgets.items():
      signal = 'valueChanged(int)' if isinstance(widget, qt.QSpinBox) else 'valueChanged(double)'
      widget.connect(signal, lambda value, name=name: self.onValueWidgetChanged(name, value))

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

    self.initializeParameterNode()
    self.onLogicStatusChanged()

  def cleanup(self):
    if self.logic:
      self.logic.stopContinuous()
      self.logic.statusCallback = None
    self.removeObservers()

  def onReload(self):
    # Slicer only reloads the main module file, so also reload the helper package to pick up changes in it
    for moduleName in list(sys.modules):
      if moduleName == 'BronchoscopicDepthMapGenerationLib' or moduleName.startswith('BronchoscopicDepthMapGenerationLib.'):
        del sys.modules[moduleName]
    ScriptedLoadableModuleWidget.onReload(self)

  def enter(self):
    self.initializeParameterNode()

  def exit(self):
    self.setParameterNode(None)

  def onSceneStartClose(self, caller, event):
    self.logic.stopContinuous()
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.initializeParameterNode()

  def onSceneEndImport(self, caller, event):
    # The node selections of a loaded scene are only available once all of its nodes are loaded. The refresh is
    # delayed until the node selectors list the loaded nodes too (they are updated after the import).
    qt.QTimer.singleShot(0, self.refreshParameterNode)

  def refreshParameterNode(self):
    self.logic.refreshParameterNode()
    # The GUI can be updated before the parameter node wrapper reads the parameters again, so it is updated again
    if self._parameterNode:
      self.updateGUIFromParameterNode()
  def initializeParameterNode(self):
    self.setParameterNode(self.logic.getParameterNode())
    # Select the first suitable volume as input, as most modules do
    if self._parameterNode and not self._parameterNode.inputVolume:
      firstVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolume and not firstVolume.IsA("vtkMRMLLabelMapVolumeNode"):
        self._parameterNode.inputVolume = firstVolume

  def setParameterNode(self, inputParameterNode):
    if self._parameterNode == inputParameterNode:
      return
    if self._parameterNode:
      self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode:
      self._parameterNodeGuiTag = self._parameterNode.connectParametersToGui(self.parameterWidgets)
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    if not self._parameterNode:
      self.observeInputVolume(None)
      return

    model = self.logic.getSelectedModel()
    itemIndex = self.modelComboBox.findData(model['id'])
    if itemIndex >= 0 and self.modelComboBox.currentIndex != itemIndex:
      wasBlocked = self.modelComboBox.blockSignals(True)
      self.modelComboBox.currentIndex = itemIndex
      self.modelComboBox.blockSignals(wasBlocked)
    if self.modelDescriptionTextBrowser.toPlainText() != model['description']:
      self.modelDescriptionTextBrowser.setPlainText(model['description'])

    for name, widget in self.valueWidgets.items():
      value = getattr(self._parameterNode, name)
      if widget.value != value:
        wasBlocked = widget.blockSignals(True)
        widget.value = value
        widget.blockSignals(wasBlocked)
    if (self.depthRangeWidget.minimumValue, self.depthRangeWidget.maximumValue) != (self._parameterNode.minimumDepth, self._parameterNode.maximumDepth):
      wasBlocked = self.depthRangeWidget.blockSignals(True)
      self.depthRangeWidget.setValues(self._parameterNode.minimumDepth, self._parameterNode.maximumDepth)
      self.depthRangeWidget.blockSignals(wasBlocked)
    for widget in self.pointCloudSettingWidgets:
      widget.enabled = self._parameterNode.enablePointCloud
    self.principalPointXSpinBox.enabled = not self._parameterNode.principalPointAtCenter
    self.principalPointYSpinBox.enabled = not self._parameterNode.principalPointAtCenter

    self.observeInputVolume(self._parameterNode.inputVolume)
    self.updateInputInfo()
    self.updateButtons()
    self.schedulePointCloudUpdate()

  def onValueWidgetChanged(self, name, value):
    if self._parameterNode:
      setattr(self._parameterNode, name, type(getattr(self._parameterNode, name))(value))

  def onDepthRangeChanged(self, minimumValue, maximumValue):
    if self._parameterNode:
      with slicer.util.NodeModify(self._parameterNode):
        self._parameterNode.minimumDepth = minimumValue
        self._parameterNode.maximumDepth = maximumValue

  def onOpenModelsFileButtonClicked(self):
    modelsFilePath = BronchoscopicDepthMapGenerationLogic.getModelsFilePath()
    if not qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(modelsFilePath)):
      slicer.util.errorDisplay(f"Could not open the models file. It is located at:\n{modelsFilePath}")

  def onModelSelected(self, itemIndex):
    if self._parameterNode:
      self._parameterNode.modelId = self.modelComboBox.itemData(itemIndex)

  # ------------------------------------ Input ---------------------------------------
  def observeInputVolume(self, inputVolume):
    """Observes the input volume so the input information follows changes in its image, e.g. when
    Video Input writes its first frame or the frame size changes."""
    if inputVolume is self._observedInputVolume:
      return
    if self._observedInputVolume:
      self.removeObserver(self._observedInputVolume, slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onInputImageModified)
    self._observedInputVolume = inputVolume
    if inputVolume:
      self.addObserver(inputVolume, slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onInputImageModified)

  def onInputImageModified(self, caller=None, event=None):
    self.updateInputInfo()
    self.updateButtons()

  def updateInputInfo(self):
    inputVolume = self._parameterNode.inputVolume if self._parameterNode else None
    if not inputVolume:
      self.inputInfoLabel.setText("")
      return
    error = BronchoscopicDepthMapGenerationLogic.getInputError(inputVolume, self._parameterNode.outputVolume)
    if error:
      text = f'<span style="color: #E53935;">{error}</span>'
    else:
      imageData = inputVolume.GetImageData()
      width, height, _ = imageData.GetDimensions()
      colorText = "grayscale" if imageData.GetNumberOfScalarComponents() == 1 else "color, converted to grayscale"
      text = f"{width} x {height} px, {colorText}"
    if self.inputInfoLabel.text != text:
      self.inputInfoLabel.setText(text)

  # ------------------------------------ Run ---------------------------------------
  def onApplyButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to generate the depth map.", waitCursor=True):
      self.logic.generateDepthMap()
    self.updateButtons()

  def onContinuousButtonClicked(self):
    if self.logic.isRunning:
      self.logic.stopContinuous()
      return
    with slicer.util.tryWithErrorDisplay("Failed to start depth map generation.", waitCursor=True):
      self.logic.startContinuous()
    self.updateButtons()

  def onLogicStatusChanged(self, errorMessage=None):
    self.deviceLabel.text = self.logic.deviceName or "Not loaded yet"
    inferenceTimeS = self.logic.lastInferenceTimeS
    self.inferenceTimeLabel.text = f"{inferenceTimeS * 1000:.1f} ms" if inferenceTimeS is not None else "-"
    frameRate = self.logic.measuredFrameRate if self.logic.isRunning else None
    self.frameRateLabel.text = f"{frameRate:.1f} fps" if frameRate else "-"
    self.updatePointCloudInfo()
    self.updateButtons()
    if errorMessage:
      slicer.util.errorDisplay(errorMessage)

  # ------------------------------------ Point Cloud ---------------------------------------
  def schedulePointCloudUpdate(self):
    """Regenerates the point cloud when it is enabled or its settings change (e.g. while adjusting the intrinsics),
    so the effect is visible without generating a new depth map. Rapid changes are combined into one update."""
    settings = self.logic.getPointCloudSettings()
    if settings == self._lastPointCloudSettings:
      return
    self._lastPointCloudSettings = settings
    if self._parameterNode.enablePointCloud and not self._pointCloudUpdatePending:
      self._pointCloudUpdatePending = True
      qt.QTimer.singleShot(0, self.updatePointCloudFromSettings)

  def updatePointCloudFromSettings(self):
    self._pointCloudUpdatePending = False
    if not self._parameterNode or not self._parameterNode.enablePointCloud:
      return
    if self.logic.getDepthMapError(self._parameterNode.outputVolume):
      return  # Nothing to update until there is a depth map
    try:
      self.logic.generatePointCloud()
    except Exception as e:
      logging.error(f"Failed to update the point cloud: {e}")
    # Generating can create the point cloud model, which changes the settings
    self._lastPointCloudSettings = self.logic.getPointCloudSettings()
    self.updatePointCloudInfo()

  def updatePointCloudInfo(self):
    if not self._parameterNode or not self._parameterNode.enablePointCloud:
      text = "-"
    elif self.logic.pointCount is None:
      text = "Waiting for a depth map"
    else:
      text = f"{self.logic.pointCount:,} ({self.logic.lastPointCloudTimeS * 1000:.1f} ms)"
    if self.pointCloudInfoLabel.text != text:
      self.pointCloudInfoLabel.text = text

  def updateButtons(self):
    isRunning = self.logic.isRunning
    inputVolume = self._parameterNode.inputVolume if self._parameterNode else None
    inputError = BronchoscopicDepthMapGenerationLogic.getInputError(inputVolume, self._parameterNode.outputVolume) if self._parameterNode else "No parameter node"

    self.applyButton.enabled = not isRunning and not inputError
    self.applyButton.toolTip = inputError or "Generate the depth map of the current input image once"

    wasBlocked = self.continuousButton.blockSignals(True)
    self.continuousButton.checked = isRunning
    self.continuousButton.blockSignals(wasBlocked)
    style = slicer.app.style()
    if isRunning:
      self.continuousButton.text = " Stop"
      self.continuousButton.icon = style.standardIcon(qt.QStyle.SP_MediaPause)
      self.continuousButton.styleSheet = self.STOP_BUTTON_STYLE
    else:
      self.continuousButton.text = " Run Continuously"
      self.continuousButton.icon = style.standardIcon(qt.QStyle.SP_MediaPlay)
      self.continuousButton.styleSheet = self.START_BUTTON_STYLE
    # Continuous mode can start before the input has an image (e.g. before Video Input plays) and waits for one
    canStart = inputVolume is not None and (not inputError or inputVolume.GetImageData() is None)
    self.continuousButton.enabled = isRunning or canStart
    if not self.continuousButton.enabled:
      self.continuousButton.styleSheet = ""  # Default (greyed out) look when disabled

# ------------------------------------ Logic ---------------------------------------
class BronchoscopicDepthMapGenerationLogic(ScriptedLoadableModuleLogic):
  # The model runs in the update timer on the main thread. A background thread would theoretically keep Slicer more responsive,
  # but I've found that it may have made each depth map take significantly longer to generate.
  UPDATE_INTERVAL_MS = 5
  STATUS_INTERVAL_S = 0.5
  FRAME_RATE_MEASUREMENT_WINDOW = 20

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self._parameterNode = None
    self.models = self.loadModels()
    self.estimator = None
    self._estimatorKey = None
    self.deviceName = None
    self.lastInferenceTimeS = None
    self.measuredFrameRate = None
    self.processedFrameCount = 0
    self.pointCount = None
    self.lastPointCloudTimeS = None
    self.statusCallback = None
    self._lastInputKey = None
    self._lastRunTime = 0
    self._lastStatusTime = 0
    self._frameTimes = collections.deque(maxlen=self.FRAME_RATE_MEASUREMENT_WINDOW)
    self.updateTimer = qt.QTimer()
    self.updateTimer.setInterval(self.UPDATE_INTERVAL_MS)
    self.updateTimer.connect('timeout()', self.onUpdateTimer)

  def getParameterNode(self):
    parameterNode = super().getParameterNode()
    if not self._parameterNode or self._parameterNode.parameterNode != parameterNode:
      self._parameterNode = BronchoscopicDepthMapGenerationParameterNode(parameterNode)
    return self._parameterNode

  def refreshParameterNode(self):
    """Makes the parameter node wrapper read the parameters again, e.g. after a scene is loaded. The wrapper keeps the
    selected nodes and reads them again when the parameter node is modified, but when a scene is loaded, the parameter
    node is modified before the nodes it refers to are loaded, so they would be read as None."""
    self.getParameterNode().parameterNode.Modified()

  # -- Models
  @staticmethod
  def getModelsFilePath():
    return os.path.join(os.path.dirname(__file__), 'Resources', 'Models.json')

  @staticmethod
  def loadModels():
    """Reads the list of available models from Resources/Models.json."""
    with open(BronchoscopicDepthMapGenerationLogic.getModelsFilePath(), encoding='utf-8') as modelsFile:
      return json.load(modelsFile)['models']

  def getModel(self, modelId):
    for model in self.models:
      if model['id'] == modelId:
        return model
    raise ValueError(f"Unknown model: {modelId}")

  def getSelectedModel(self):
    """Returns the model selected in the parameter node. If no model or an unknown model is selected (e.g. a model
    that was renamed or removed since the scene was saved), the first model is selected instead."""
    parameterNode = self.getParameterNode()
    for model in self.models:
      if model['id'] == parameterNode.modelId:
        return model
    parameterNode.modelId = self.models[0]['id']
    return self.models[0]

  @staticmethod
  def getModelDirectory():
    """Downloaded model weights are stored in Slicer's cache folder, outside of the extension."""
    return os.path.join(slicer.app.cachePath, 'BronchoscopicDepthMapGeneration', 'Models')

  def getModelFilePath(self, model):
    return os.path.join(self.getModelDirectory(), model['fileName'])

  def isModelDownloaded(self, model):
    return os.path.isfile(self.getModelFilePath(model))

  def downloadModel(self, model, askConfirmation=True):
    """Returns the path of the model weights, downloading them first if needed. Returns None if the user cancels."""
    filePath = self.getModelFilePath(model)
    if os.path.isfile(filePath):
      return filePath
    if not model['url']:
      raise RuntimeError(f"The {model['title']} model is not available for download yet. "
                         f"Copy {model['fileName']} into this folder:\n{self.getModelDirectory()}")
    if askConfirmation and not slicer.util.confirmOkCancelDisplay(
        f"The {model['title']} model needs to be downloaded before it can be used. Download it now?",
        "Download model"):
      return None
    os.makedirs(self.getModelDirectory(), exist_ok=True)
    import SampleData
    SampleData.SampleDataLogic().downloadFile(model['url'], self.getModelDirectory(), model['fileName'],
                                              checksum=f"SHA256:{model['sha256']}")
    if not os.path.isfile(filePath):
      raise RuntimeError(f"Failed to download the {model['title']} model. See the application log for details.")
    return filePath

  @staticmethod
  def computeSha256(filePath):
    sha256 = hashlib.sha256()
    with open(filePath, 'rb') as file:
      for block in iter(lambda: file.read(1024 * 1024), b''):
        sha256.update(block)
    return sha256.hexdigest()

  # -- PyTorch
  @staticmethod
  def isTorchInstalled():
    return importlib.util.find_spec('torch') is not None

  @staticmethod
  def checkTorchInstalled():
    """Raises an error explaining how to install PyTorch if it is not installed. PyTorch is not installed by this
    module: the user installs it with the PyTorch extension, which selects the build that matches their GPU."""
    if BronchoscopicDepthMapGenerationLogic.isTorchInstalled():
      return
    if importlib.util.find_spec('PyTorchUtils') is None:
      raise RuntimeError("Bronchoscopic Depth Map Generation requires PyTorch.\n\n"
                         "Install the PyTorch extension from the Extensions Manager and restart Slicer. Then open the "
                         "PyTorch Utils module (Utilities > PyTorch Utils) and install PyTorch.")
    raise RuntimeError("Bronchoscopic Depth Map Generation requires PyTorch.\n\n"
                       "Open the PyTorch Utils module (Utilities > PyTorch Utils) and install PyTorch.")

  @staticmethod
  def getDevice(useGPU):
    import torch
    if useGPU and torch.cuda.is_available():
      return 'cuda'
    if useGPU and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      return 'mps'
    return 'cpu'

  def loadModel(self, askConfirmation=True):
    """Downloads the selected model and loads it on the selected device, as needed. PyTorch must be installed.
    Returns False if the user cancels the download."""
    parameterNode = self.getParameterNode()
    model = self.getSelectedModel()
    key = (model['id'], parameterNode.useGPU)
    if self.estimator and self._estimatorKey == key:
      return True

    self.checkTorchInstalled()
    weightsPath = self.downloadModel(model, askConfirmation)
    if not weightsPath:
      return False
    if self.computeSha256(weightsPath) != model['sha256']:
      raise RuntimeError(f"The {model['title']} model file is not the expected version or is damaged. "
                         f"Delete it so it is downloaded again:\n{weightsPath}")

    import torch
    from BronchoscopicDepthMapGenerationLib.DepthEstimator import DepthEstimator
    device = self.getDevice(parameterNode.useGPU)
    self.estimator = DepthEstimator(model, weightsPath, device)
    self._estimatorKey = key
    if device == 'cuda':
      self.deviceName = f"GPU ({torch.cuda.get_device_name(0)})"
    elif device == 'mps':
      self.deviceName = "GPU (Apple)"
    else:
      self.deviceName = "CPU"
    logging.info(f"Bronchoscopic Depth Map Generation: loaded {model['title']} on {self.deviceName}")
    return True

  # -- Input and output
  @staticmethod
  def getInputError(inputVolume, outputVolume=None):
    """Returns a message describing why the volume cannot be used as input, or None if it can."""
    if inputVolume is None:
      return "Select an input volume."
    if outputVolume is not None and outputVolume is inputVolume:
      return "The output volume must be different from the input volume."
    imageData = inputVolume.GetImageData()
    if imageData is None:
      return "The input volume has no image yet."
    width, height, depth = imageData.GetDimensions()
    if depth != 1:
      return f"The input must be a 2D image (a single slice), but it has {depth} slices ({width} x {height} x {depth} px)."
    if width < 2 or height < 2:
      return f"The input image is too small ({width} x {height} px)."
    components = imageData.GetNumberOfScalarComponents()
    if components not in (1, 3, 4):
      return f"The input must be a grayscale or color (RGB) image, but it has {components} components per pixel."
    return None

  @staticmethod
  def getInputImage(inputVolume):
    """Returns the input image as a 2D float32 array (height, width) with values in [0, 1], as the models expect.
    Color images are converted to grayscale. 8-bit images are scaled from [0, 255]; other types are scaled from
    their minimum and maximum values."""
    image = slicer.util.arrayFromVolume(inputVolume)[0]
    isUint8 = image.dtype == np.uint8
    if image.ndim == 3:
      # Color (RGB or RGBA) to grayscale with the ITU-R BT.601 luma weights, as OpenCV and PIL do
      image = image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
      image = image.astype(np.float32)
    if isUint8:
      return image / 255.0
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
      return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)

  def getOrCreateOutputVolume(self):
    parameterNode = self.getParameterNode()
    if not parameterNode.outputVolume:
      outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName("DepthMap"))
      parameterNode.outputVolume = outputVolume
    return parameterNode.outputVolume

  @staticmethod
  def writeOutput(depthMap, inputVolume, outputVolume):
    """Writes the depth map into the output volume, with the same geometry as the input so they overlap."""
    volumeArray = depthMap[np.newaxis, ...]
    imageData = outputVolume.GetImageData()
    if imageData is not None and slicer.util.arrayFromVolume(outputVolume).shape == volumeArray.shape and imageData.GetScalarType() == vtk.VTK_UNSIGNED_CHAR:
      # Fast path: copy into the existing image buffer
      slicer.util.arrayFromVolume(outputVolume)[:] = volumeArray
      slicer.util.arrayFromVolumeModified(outputVolume)
    else:
      slicer.util.updateVolumeFromArray(outputVolume, volumeArray)
      if not outputVolume.GetDisplayNode():
        outputVolume.CreateDefaultDisplayNodes()
      # A fixed window keeps the brightness from changing with every frame
      displayNode = outputVolume.GetDisplayNode()
      displayNode.AutoWindowLevelOff()
      displayNode.SetWindowLevelMinMax(0, 255)

    inputIjkToRas = vtk.vtkMatrix4x4()
    inputVolume.GetIJKToRASMatrix(inputIjkToRas)
    outputIjkToRas = vtk.vtkMatrix4x4()
    outputVolume.GetIJKToRASMatrix(outputIjkToRas)
    if any(inputIjkToRas.GetElement(row, column) != outputIjkToRas.GetElement(row, column) for row in range(4) for column in range(4)):
      outputVolume.SetIJKToRASMatrix(inputIjkToRas)
    if outputVolume.GetTransformNodeID() != inputVolume.GetTransformNodeID():
      outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())

  # -- Single image
  def generateDepthMap(self, askConfirmation=True):
    """Generates the depth map of the current input image. Returns False if the user cancels the model download."""
    parameterNode = self.getParameterNode()
    inputError = self.getInputError(parameterNode.inputVolume, parameterNode.outputVolume)
    if inputError:
      raise ValueError(inputError)
    if not self.loadModel(askConfirmation):
      return False
    startTime = time.perf_counter()
    depthMap = self.estimator.predict(self.getInputImage(parameterNode.inputVolume))
    self.lastInferenceTimeS = time.perf_counter() - startTime
    self.writeOutput(depthMap, parameterNode.inputVolume, self.getOrCreateOutputVolume())
    self.processedFrameCount += 1
    if parameterNode.enablePointCloud:
      self.generatePointCloud()
    self._notifyStatus()
    return True

  # -- Continuous
  @property
  def isRunning(self):
    return self.updateTimer.isActive()

  def startContinuous(self, askConfirmation=True):
    """Starts generating depth maps whenever the input image changes. Returns False if the user cancels."""
    if self.isRunning:
      return True
    parameterNode = self.getParameterNode()
    inputVolume = parameterNode.inputVolume
    if inputVolume is None:
      raise ValueError("Select an input volume.")
    # An input without an image is allowed: frames are processed once they arrive
    inputError = self.getInputError(inputVolume, parameterNode.outputVolume)
    if inputError and inputVolume.GetImageData() is not None:
      raise ValueError(inputError)
    if not self.loadModel(askConfirmation):
      return False
    self.getOrCreateOutputVolume()
    self._lastInputKey = None
    self._lastRunTime = 0
    self._frameTimes.clear()
    self.measuredFrameRate = None
    self.updateTimer.start()
    self._notifyStatus()
    return True

  def stopContinuous(self, errorMessage=None):
    wasRunning = self.isRunning
    self.updateTimer.stop()
    self.measuredFrameRate = None
    if wasRunning or errorMessage:
      self._notifyStatus(errorMessage)

  @staticmethod
  def getImageModifiedTime(imageData):
    # arrayFromVolumeModified (used by Video Input) marks the scalar array as modified, not the image itself
    scalars = imageData.GetPointData().GetScalars()
    return max(imageData.GetMTime(), scalars.GetMTime() if scalars else 0)

  def onUpdateTimer(self):
    """Generates the depth map of the input image if it changed and the maximum frame rate allows it."""
    parameterNode = self.getParameterNode()
    inputVolume = parameterNode.inputVolume
    outputVolume = parameterNode.outputVolume
    if inputVolume is None or outputVolume is None:
      self.stopContinuous()
      return

    # Only process new images, and no more often than the maximum frame rate
    imageData = inputVolume.GetImageData()
    if imageData is None:
      return  # E.g. Video Input has not written its first frame yet
    inputKey = (inputVolume.GetID(), self.getImageModifiedTime(imageData))
    if inputKey == self._lastInputKey:
      return
    now = time.perf_counter()
    if now - self._lastRunTime < 1.0 / max(parameterNode.maxFrameRate, 0.01):
      return
    inputError = self.getInputError(inputVolume, outputVolume)
    if inputError:
      self.stopContinuous(inputError)
      return

    # The model or device can be changed while running
    if (parameterNode.modelId, parameterNode.useGPU) != self._estimatorKey:
      try:
        if not self.loadModel():
          self.stopContinuous()
          return
      except Exception as e:
        self.stopContinuous(f"Failed to load the model: {e}")
        return
      self._notifyStatus()

    self._lastInputKey = inputKey
    self._lastRunTime = now
    try:
      startTime = time.perf_counter()
      depthMap = self.estimator.predict(self.getInputImage(inputVolume))
      self.lastInferenceTimeS = time.perf_counter() - startTime
    except Exception as e:
      self.stopContinuous(f"Depth map generation failed: {e}")
      return
    self.writeOutput(depthMap, inputVolume, outputVolume)
    self.processedFrameCount += 1
    if parameterNode.enablePointCloud:
      try:
        self.generatePointCloud()
      except Exception as e:
        self.stopContinuous(f"Point cloud generation failed: {e}")
        return

    self._frameTimes.append(now)
    if len(self._frameTimes) >= 2 and self._frameTimes[-1] > self._frameTimes[0]:
      self.measuredFrameRate = (len(self._frameTimes) - 1) / (self._frameTimes[-1] - self._frameTimes[0])
    if now - self._lastStatusTime > self.STATUS_INTERVAL_S:
      self._notifyStatus()

  def _notifyStatus(self, errorMessage=None):
    self._lastStatusTime = time.perf_counter()
    if self.statusCallback:
      self.statusCallback(errorMessage)

  # -- Point cloud
  @staticmethod
  def getDepthMapError(depthMapVolume):
    """Returns a message describing why a point cloud cannot be generated from the volume, or None if it can."""
    if depthMapVolume is None:
      return "Generate a depth map first, or select a depth map as the output volume."
    imageData = depthMapVolume.GetImageData()
    if imageData is None:
      return "The output volume has no depth map yet. Generate a depth map first."
    width, height, depth = imageData.GetDimensions()
    if depth != 1:
      return f"The depth map must be a 2D image (a single slice), but it has {depth} slices ({width} x {height} x {depth} px)."
    if imageData.GetNumberOfScalarComponents() != 1:
      return "The depth map must be a grayscale image."
    return None

  def getPointCloudSettings(self):
    """Everything the point cloud depends on except the image contents, so changes to it can be detected."""
    p = self.getParameterNode()
    return (p.outputVolume, p.inputVolume, p.pointCloudModel, p.cameraTransform, p.enablePointCloud,
            p.focalLengthX, p.focalLengthY, p.principalPointAtCenter, p.principalPointX, p.principalPointY,
            p.scaleX, p.scaleY, p.scaleZ, p.minimumDepth, p.maximumDepth, p.pixelDensity, p.colorPointCloud)

  @staticmethod
  def getFullScaleDepth(depthMap):
    """Value of the farthest possible depth (100%): the largest value of the pixel type for integer depth maps
    (255 for the 8-bit depth maps from the models), and the largest value in the image for floating point ones."""
    if np.issubdtype(depthMap.dtype, np.integer):
      return float(np.iinfo(depthMap.dtype).max)
    maximum = float(depthMap.max()) if depthMap.size else 0.0
    return maximum if maximum > 0 else 1.0

  @staticmethod
  def getSampledIndices(size, density):
    """Evenly spread indices that keep about the given fraction of 0..size-1, including both ends."""
    count = min(max(int(round(size * density)), 1), size)
    return np.unique(np.round(np.linspace(0, size - 1, count)).astype(int))

  @staticmethod
  def depthMapToPoints(depthMap, focalLength, principalPoint, scale, depthRangePercent, pixelDensityPercent=100.0):
    """Back-projects the pixels of a depth map (rows, columns) into 3D points with the pinhole camera model.

    Depth is relative: the farthest possible depth (see getFullScaleDepth) is 100%. The points are first computed
    with 100% depth at a distance of 1, then each axis is multiplied by its value in scale (X, Y, Z), in mm. Equal
    scales keep the shape of the point cloud; different scales stretch it along an axis.
    Pixels with zero depth, or a depth outside depthRangePercent (minimum, maximum), are left out. Only about
    pixelDensityPercent of the pixels are used, spread evenly over the image. The points are in the camera
    coordinates used by Slicer-DepthMapTracking: +Z is the viewing direction, +X points to the bottom of the image,
    and +Y to the left of the image. focalLength is (fx, fy) and principalPoint is (cx, cy), in pixels.

    Returns the points as a float32 array (N, 3), and the row and column of the pixel of each point.
    """
    fx, fy = focalLength
    cx, cy = principalPoint
    height, width = depthMap.shape
    # The same fraction of rows and columns, so the fraction of pixels is the density
    axisDensity = math.sqrt(min(max(pixelDensityPercent, 0.0), 100.0) / 100.0)
    rows, columns = np.meshgrid(BronchoscopicDepthMapGenerationLogic.getSampledIndices(height, axisDensity),
                                BronchoscopicDepthMapGenerationLogic.getSampledIndices(width, axisDensity), indexing='ij')
    depthPercent = depthMap[rows, columns].astype(np.float32) * np.float32(100.0 / BronchoscopicDepthMapGenerationLogic.getFullScaleDepth(depthMap))
    keep = (depthPercent > 0) & (depthPercent >= depthRangePercent[0]) & (depthPercent <= depthRangePercent[1])
    rows, columns = rows[keep], columns[keep]
    depth = depthPercent[keep] / np.float32(100.0)
    scaleX, scaleY, scaleZ = scale
    points = np.empty((len(depth), 3), dtype=np.float32)
    # Image x (column) and y (row) in camera coordinates, then rotated about the viewing direction into the
    # orientation used by Slicer-DepthMapTracking
    points[:, 0] = (rows - cy) * depth / fy * scaleX
    points[:, 1] = -(columns - cx) * depth / fx * scaleY
    points[:, 2] = depth * scaleZ
    return points, rows, columns

  @staticmethod
  def getPointColors(inputVolume, depthMapShape, rows, columns):
    """Returns the color (N, 3) of each point's pixel in the input image as uint8 RGB, or None if the input image
    does not have the same size as the depth map."""
    if inputVolume is None or inputVolume.GetImageData() is None:
      return None
    image = slicer.util.arrayFromVolume(inputVolume)
    if image.shape[0] != 1 or image.shape[1:3] != depthMapShape:
      return None
    colors = image[0][rows, columns]
    if colors.ndim == 1:
      colors = np.repeat(colors[:, np.newaxis], 3, axis=1)
    elif colors.shape[1] >= 3:
      colors = colors[:, :3]
    else:
      return None
    if colors.dtype != np.uint8 and colors.size > 0:
      # Other pixel types have no fixed range, so they are scaled from their minimum to their maximum
      minimum, maximum = float(colors.min()), float(colors.max())
      scale = 255.0 / (maximum - minimum) if maximum > minimum else 0.0
      colors = ((colors.astype(np.float32) - minimum) * scale).round()
    return np.ascontiguousarray(colors, dtype=np.uint8)

  @staticmethod
  def createPointCloudPolyData(points, colors=None):
    """Creates poly data with a vertex for each point, so it is displayed as points."""
    pointCount = len(points)
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetData(numpy_support.numpy_to_vtk(points, deep=True))
    vertices = vtk.vtkCellArray()
    offsets = numpy_support.numpy_to_vtkIdTypeArray(np.arange(pointCount + 1, dtype=numpy_support.ID_TYPE_CODE), deep=True)
    connectivity = numpy_support.numpy_to_vtkIdTypeArray(np.arange(pointCount, dtype=numpy_support.ID_TYPE_CODE), deep=True)
    vertices.SetData(offsets, connectivity)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtkPoints)
    polyData.SetVerts(vertices)
    if colors is not None:
      colorArray = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
      colorArray.SetName("Color")
      polyData.GetPointData().SetScalars(colorArray)
    return polyData

  def getOrCreatePointCloudModel(self):
    parameterNode = self.getParameterNode()
    if not parameterNode.pointCloudModel:
      parameterNode.pointCloudModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", slicer.mrmlScene.GenerateUniqueName("PointCloud"))
    modelNode = parameterNode.pointCloudModel
    if not modelNode.GetDisplayNode():
      # Shown as points, as Slicer-DepthMapTracking did; the display can be changed in the Models module
      modelNode.CreateDefaultDisplayNodes()
      displayNode = modelNode.GetDisplayNode()
      displayNode.SetRepresentation(displayNode.PointsRepresentation)
      displayNode.SetPointSize(4)
      displayNode.SetColor(1, 0, 0)
    return modelNode

  def generatePointCloud(self):
    """Generates the point cloud of the depth map in the output volume. Returns the point cloud model."""
    parameterNode = self.getParameterNode()
    depthMapVolume = parameterNode.outputVolume
    depthMapError = self.getDepthMapError(depthMapVolume)
    if depthMapError:
      raise ValueError(depthMapError)

    startTime = time.perf_counter()
    depthMap = slicer.util.arrayFromVolume(depthMapVolume)[0]
    height, width = depthMap.shape
    if parameterNode.principalPointAtCenter:
      principalPoint = ((width - 1) / 2.0, (height - 1) / 2.0)
    else:
      principalPoint = (parameterNode.principalPointX, parameterNode.principalPointY)
    points, rows, columns = self.depthMapToPoints(
      depthMap, (parameterNode.focalLengthX, parameterNode.focalLengthY), principalPoint,
      (parameterNode.scaleX, parameterNode.scaleY, parameterNode.scaleZ),
      (parameterNode.minimumDepth, parameterNode.maximumDepth), parameterNode.pixelDensity)
    colors = self.getPointColors(parameterNode.inputVolume, depthMap.shape, rows, columns) if parameterNode.colorPointCloud else None

    modelNode = self.getOrCreatePointCloudModel()
    modelNode.SetAndObservePolyData(self.createPointCloudPolyData(points, colors))
    displayNode = modelNode.GetDisplayNode()
    if displayNode:
      if colors is not None:
        displayNode.SetActiveScalarName("Color")
        displayNode.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
      displayNode.SetScalarVisibility(colors is not None)
    cameraTransformId = parameterNode.cameraTransform.GetID() if parameterNode.cameraTransform else None
    if modelNode.GetTransformNodeID() != cameraTransformId:
      modelNode.SetAndObserveTransformNodeID(cameraTransformId)

    self.pointCount = len(points)
    self.lastPointCloudTimeS = time.perf_counter() - startTime
    return modelNode

# ------------------------------------ Tests ---------------------------------------
class BronchoscopicDepthMapGenerationTest(ScriptedLoadableModuleTest):
  """Tests for the Bronchoscopic Depth Map Generation module. Run them with the Reload and Test button (developer mode) or
  through CTest. Input images are generated by the tests. The tests that run the model need PyTorch (installed by the
  user through the PyTorch extension) and the model weights (downloaded, or copied into the model folder); if either
  is not available (e.g. on test machines without internet access), those tests are skipped and reported in the log.
  Without a GPU, the model runs on the CPU."""

  def setUp(self):
    slicer.mrmlScene.Clear()
    self.logic = BronchoscopicDepthMapGenerationLogic()
    # Reset the parameters, which keep their values when the scene is cleared
    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      parameterNode.modelId = self.logic.models[0]['id']
      parameterNode.inputVolume = None
      parameterNode.outputVolume = None
      parameterNode.maxFrameRate = 10.0
      parameterNode.useGPU = True
      parameterNode.pointCloudModel = None
      parameterNode.cameraTransform = None
      parameterNode.enablePointCloud = False
      parameterNode.focalLengthX = 200.0
      parameterNode.focalLengthY = 200.0
      parameterNode.principalPointAtCenter = True
      parameterNode.principalPointX = 100.0
      parameterNode.principalPointY = 100.0
      parameterNode.scaleX = 100.0
      parameterNode.scaleY = 100.0
      parameterNode.scaleZ = 100.0
      parameterNode.minimumDepth = 0.0
      parameterNode.maximumDepth = 60.0
      parameterNode.pixelDensity = 100.0
      parameterNode.colorPointCloud = True

  def runTest(self):
    tests = [
      self.test_Models,
      self.test_InputValidation,
      self.test_InputImage,
      self.test_PointCloud,
      self.test_GenerateDepthMap,
      self.test_ContinuousMode,
    ]
    self.skippedTests = []
    for test in tests:
      self.setUp()
      try:
        test()
      finally:
        self.logic.stopContinuous()
        self.doCleanups()
    if self.skippedTests:
      self.delayDisplay("Skipped tests: " + ", ".join(self.skippedTests))
    self.delayDisplay("All Bronchoscopic Depth Map Generation tests passed")

  # -- Helpers
  @staticmethod
  def waitFor(condition, timeoutS=5.0):
    """Processes events (which runs the logic's update timer) until the condition is true or the timeout passes."""
    endTime = time.perf_counter() + timeoutS
    while time.perf_counter() < endTime:
      slicer.app.processEvents()
      if condition():
        return True
      time.sleep(0.005)
    return condition()

  @staticmethod
  def createVolume(array, className="vtkMRMLScalarVolumeNode", name="TestImage"):
    """Creates a volume from an array of shape (slices, rows, columns) or (slices, rows, columns, components)."""
    volume = slicer.mrmlScene.AddNewNodeByClass(className, name)
    slicer.util.updateVolumeFromArray(volume, array)
    return volume

  @staticmethod
  def createTestImage(width, height):
    """Roughly like looking at an airway bifurcation: bright nearby walls with two dark holes side by side (the
    lumens of the two branches, which are farther away), one larger than the other. Each hole fades from black at
    its center to the wall brightness at its edge."""
    y, x = np.mgrid[0:height, 0:width]
    brightness = np.ones((height, width))
    # Center and radius of each hole, as fractions of the image width
    for holeCenterX, holeRadius in [(0.3, 0.22), (0.72, 0.13)]:
      distance = np.hypot(x / width - holeCenterX, (y - height / 2) / width)
      brightness = np.minimum(brightness, np.clip(distance / holeRadius, 0, 1))
    return (55 + 200 * brightness).astype(np.uint8)  # Walls are not pure white, as in real images

  def canRunModel(self, testName):
    """Loads the model if PyTorch is installed and the model is available. Returns False (and records the test
    as skipped) otherwise. PyTorch is not installed by the tests."""
    model = self.logic.getSelectedModel()
    reason = None
    if not self.logic.isTorchInstalled():
      reason = "PyTorch is not installed"
    elif not self.logic.isModelDownloaded(model) and not model['url']:
      reason = f"the {model['title']} model is not in {self.logic.getModelDirectory()} and has no download URL"
    elif not self.logic.isModelDownloaded(model):
      # E.g. on test machines without internet access
      try:
        self.logic.downloadModel(model, askConfirmation=False)
      except Exception as e:
        reason = f"the {model['title']} model could not be downloaded ({e})"
    if reason:
      logging.warning(f"Skipping {testName}: {reason}")
      self.skippedTests.append(testName)
      return False
    self.assertTrue(self.logic.loadModel(askConfirmation=False), "Failed to download the model")
    return True

  # -- Tests that do not need PyTorch
  def test_Models(self):
    """Checks that every model in Resources/Models.json is complete.

    Each model needs a unique id, a title and description for the model selector, a known network type with its
    settings, a 2-element input size, a weights file name, a download URL field (it may be empty until the weights
    are published), and a SHA256 checksum, which is used to verify downloaded and copied weights.
    """
    self.delayDisplay("Testing the model list")
    self.assertGreater(len(self.logic.models), 0)
    modelIds = [model['id'] for model in self.logic.models]
    self.assertEqual(len(modelIds), len(set(modelIds)), "Model ids must be unique")
    for model in self.logic.models:
      for field in ['id', 'title', 'description', 'network', 'inputSize', 'fileName', 'url', 'sha256']:
        self.assertIn(field, model, f"Model {model.get('id')} has no {field}")
      self.assertEqual(model['network'], 'GeneratorResNet', f"Model {model['id']} has an unknown network type")
      self.assertIsInstance(model['residualBlocks'], int)
      self.assertEqual(len(model['inputSize']), 2)
      self.assertRegex(model['sha256'], r'^[0-9a-f]{64}$')
    self.assertEqual(self.logic.getModel(modelIds[0])['id'], modelIds[0])
    with self.assertRaises(ValueError):
      self.logic.getModel('NoSuchModel')

  def test_InputValidation(self):
    """Checks which volumes are accepted as input, and that rejected volumes give a clear reason.

    The model works on single 2D images, so the input is checked before the model runs:
    - No volume, a volume without an image (e.g. a Video Input output before its first frame), and using the same
      volume as input and output are rejected.
    - A volume with more than one slice (a 3D volume) is rejected, and the message gives its size.
    - A volume that is too small, or that has 2 components per pixel (neither grayscale nor color), is rejected.
    - Grayscale images, RGB images, and RGBA images are accepted.
    """
    self.delayDisplay("Testing input validation")
    getInputError = BronchoscopicDepthMapGenerationLogic.getInputError
    self.assertIsNotNone(getInputError(None))
    emptyVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    self.assertIn("no image", getInputError(emptyVolume))

    grayscale = self.createVolume(np.zeros((1, 30, 40), dtype=np.uint8))
    self.assertIsNone(getInputError(grayscale))
    self.assertIn("different", getInputError(grayscale, grayscale))
    self.assertIsNone(getInputError(self.createVolume(np.zeros((1, 30, 40, 3), dtype=np.uint8), "vtkMRMLVectorVolumeNode")))
    self.assertIsNone(getInputError(self.createVolume(np.zeros((1, 30, 40, 4), dtype=np.uint8), "vtkMRMLVectorVolumeNode")))

    error = getInputError(self.createVolume(np.zeros((5, 30, 40), dtype=np.uint8)))
    self.assertIn("2D", error)
    self.assertIn("40 x 30 x 5", error)
    self.assertIn("too small", getInputError(self.createVolume(np.zeros((1, 1, 40), dtype=np.uint8))))
    self.assertIn("components", getInputError(self.createVolume(np.zeros((1, 30, 40, 2), dtype=np.uint8), "vtkMRMLVectorVolumeNode")))

  def test_InputImage(self):
    """Checks how input volumes are converted into the [0, 1] grayscale images the models expect.

    - 8-bit grayscale values are divided by 255, so 0 becomes 0, 255 becomes 1, and 51 becomes 0.2, which is how
      the models were trained.
    - Color images are converted to grayscale with the standard luma weights, so pure red (255, 0, 0) becomes
      0.299. An alpha channel is ignored.
    - Other pixel types (e.g. 16-bit or float images) have no fixed range, so they are scaled from their minimum
      to their maximum value. A constant image becomes all zeros instead of dividing by zero.
    - The result is a 2D (rows, columns) float32 array with the same size as the input.
    """
    self.delayDisplay("Testing input image conversion")
    getInputImage = BronchoscopicDepthMapGenerationLogic.getInputImage

    image = getInputImage(self.createVolume(np.array([[[0, 51, 255]]], dtype=np.uint8)))
    self.assertEqual(image.dtype, np.float32)
    self.assertEqual(image.shape, (1, 3))
    np.testing.assert_allclose(image[0], [0.0, 0.2, 1.0], atol=1e-6)

    colorArray = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    colorArray[0, 0, 0] = (255, 0, 0, 7)
    colorArray[0, 1, 2] = (255, 255, 255, 0)
    image = getInputImage(self.createVolume(colorArray, "vtkMRMLVectorVolumeNode"))
    self.assertEqual(image.shape, (2, 3))
    self.assertAlmostEqual(float(image[0, 0]), 0.299, places=5)
    self.assertAlmostEqual(float(image[1, 2]), 1.0, places=5)
    self.assertEqual(float(image[0, 1]), 0.0)

    image = getInputImage(self.createVolume(np.array([[[1000, 1500, 3000]]], dtype=np.uint16)))
    np.testing.assert_allclose(image[0], [0.0, 0.25, 1.0], atol=1e-6)
    image = getInputImage(self.createVolume(np.full((1, 4, 5), 7.5, dtype=np.float32)))
    np.testing.assert_array_equal(image, 0.0)

  def test_PointCloud(self):
    """Checks generating a point cloud from a depth map with the pinhole camera model.

    A small depth map (8 x 6 px) with known values is used, so each point can be checked:
    - A missing depth map, one without an image, a 3D volume, and a color image are rejected with a clear reason.
    - Depth is relative: 255 (the largest 8-bit value) is 100%. The scale of all axes is set to 102 mm, so each gray
      level is 0.4 mm of depth. For other integer types 100% is the largest value of the type, and for floating point
      depth maps it is the largest value in the image.
    - Each pixel becomes a point at ((row - cy) x depth / fy, -(column - cx) x depth / fx, depth), the camera
      coordinates used by Slicer-DepthMapTracking. With the principal point at the image center, (cx, cy) is the center
      of the pixel grid.
    - Different scales for each axis stretch the point cloud along that axis only.
    - Pixels with zero depth, or a depth outside the depth range (in percent), are left out.
    - A pixel density of 25% uses half of the rows and half of the columns, spread evenly and including the edges.
    - The points are displayed as points, and colored with the input image when it has the same size as the depth
      map (not when it has a different size, or when coloring is turned off).
    - The point cloud is placed under the camera transform.
    """
    self.delayDisplay("Testing point cloud generation")
    getDepthMapError = BronchoscopicDepthMapGenerationLogic.getDepthMapError
    self.assertIn("Generate a depth map", getDepthMapError(None))
    self.assertIn("no depth map", getDepthMapError(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")))
    self.assertIn("2D", getDepthMapError(self.createVolume(np.zeros((3, 6, 8), dtype=np.uint8))))
    self.assertIn("grayscale", getDepthMapError(self.createVolume(np.zeros((1, 6, 8, 3), dtype=np.uint8), "vtkMRMLVectorVolumeNode")))
    with self.assertRaises(ValueError):
      self.logic.generatePointCloud()

    getFullScaleDepth = BronchoscopicDepthMapGenerationLogic.getFullScaleDepth
    self.assertEqual(getFullScaleDepth(np.zeros((2, 2), dtype=np.uint8)), 255.0)
    self.assertEqual(getFullScaleDepth(np.zeros((2, 2), dtype=np.uint16)), 65535.0)
    self.assertEqual(getFullScaleDepth(np.array([[0.5, 2.0]], dtype=np.float32)), 2.0)
    self.assertEqual(getFullScaleDepth(np.zeros((2, 2), dtype=np.float32)), 1.0)

    # 8 x 6 px depth map: 40 mm (39%) everywhere, except for no depth, too far (80 mm, 78%), and 20 mm (20%) pixels
    depthMap = np.full((6, 8), 100, dtype=np.uint8)
    depthMap[0, 0] = 0
    depthMap[0, 1] = 200
    depthMap[2, 5] = 50
    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      parameterNode.outputVolume = self.createVolume(depthMap[np.newaxis], name="DepthMap")
      parameterNode.focalLengthX = 4.0
      parameterNode.focalLengthY = 5.0
      parameterNode.scaleX = 102.0
      parameterNode.scaleY = 102.0
      parameterNode.scaleZ = 102.0
      parameterNode.colorPointCloud = False

    def pointOf(row, column):
      """Returns the index of the point generated from the pixel, or None if it was left out."""
      points = slicer.util.arrayFromModelPoints(parameterNode.pointCloudModel)
      for index, point in enumerate(points):
        if np.allclose(point, expected(row, column)):
          return index
      return None

    def expected(row, column, cx=3.5, cy=2.5):
      depth = depthMap[row, column] * 0.4
      return [(row - cy) * depth / 5.0, -(column - cx) * depth / 4.0, depth]

    modelNode = self.logic.generatePointCloud()
    self.assertIs(modelNode, parameterNode.pointCloudModel)
    points = slicer.util.arrayFromModelPoints(modelNode)
    self.assertEqual(len(points), 6 * 8 - 2)
    self.assertEqual(self.logic.pointCount, len(points))
    self.assertEqual(modelNode.GetPolyData().GetNumberOfVerts(), len(points))
    np.testing.assert_allclose(points[0], expected(0, 2), atol=1e-5)  # Pixels are in row order; (0, 0) and (0, 1) are left out
    self.assertIsNotNone(pointOf(2, 5))
    np.testing.assert_allclose(sorted(set(np.round(points[:, 2], 4))), [20.0, 40.0], atol=1e-4)
    displayNode = modelNode.GetDisplayNode()
    self.assertEqual(displayNode.GetRepresentation(), displayNode.PointsRepresentation)
    self.assertFalse(displayNode.GetScalarVisibility())

    # Explicit principal point: the pixel at the principal point lies on the viewing direction
    with slicer.util.NodeModify(parameterNode):
      parameterNode.principalPointAtCenter = False
      parameterNode.principalPointX = 5.0
      parameterNode.principalPointY = 2.0
    points = slicer.util.arrayFromModelPoints(self.logic.generatePointCloud())
    self.assertTrue(np.any(np.all(np.isclose(points, [0.0, 0.0, 20.0]), axis=1)))
    parameterNode.principalPointAtCenter = True

    # Separate scale for each axis: X doubled and Y halved, so pixel (2, 5) at (-2, -7.5, 20) moves to (-4, -3.75, 20)
    with slicer.util.NodeModify(parameterNode):
      parameterNode.scaleX = 204.0
      parameterNode.scaleY = 51.0
    points = slicer.util.arrayFromModelPoints(self.logic.generatePointCloud())
    self.assertTrue(np.any(np.all(np.isclose(points, [-4.0, -3.75, 20.0]), axis=1)))
    with slicer.util.NodeModify(parameterNode):
      parameterNode.scaleX = 102.0
      parameterNode.scaleY = 102.0

    # Depth range (percent): only the 40 mm (39%) pixels
    parameterNode.minimumDepth = 30.0
    points = slicer.util.arrayFromModelPoints(self.logic.generatePointCloud())
    self.assertEqual(len(points), 6 * 8 - 3)
    np.testing.assert_allclose(points[:, 2], 40.0, atol=1e-4)
    parameterNode.minimumDepth = 0.0

    # 25% density: rows 0, 2, 5 and columns 0, 2, 5, 7, without (0, 0)
    parameterNode.pixelDensity = 25.0
    self.assertEqual(len(slicer.util.arrayFromModelPoints(self.logic.generatePointCloud())), 3 * 4 - 1)
    self.assertIsNotNone(pointOf(2, 5))
    self.assertIsNotNone(pointOf(5, 7))
    parameterNode.pixelDensity = 100.0

    # Colors from an input image of the same size, and not from one of a different size
    colorImage = np.zeros((1, 6, 8, 3), dtype=np.uint8)
    colorImage[0, 2, 5] = (10, 20, 30)
    with slicer.util.NodeModify(parameterNode):
      parameterNode.inputVolume = self.createVolume(colorImage, "vtkMRMLVectorVolumeNode", "ColorImage")
      parameterNode.colorPointCloud = True
    modelNode = self.logic.generatePointCloud()
    colors = slicer.util.arrayFromModelPointData(modelNode, "Color")
    self.assertEqual(colors.shape, (6 * 8 - 2, 3))
    np.testing.assert_array_equal(colors[pointOf(2, 5)], (10, 20, 30))
    self.assertTrue(modelNode.GetDisplayNode().GetScalarVisibility())
    parameterNode.inputVolume = self.createVolume(np.zeros((1, 7, 8), dtype=np.uint8), name="OtherSize")
    modelNode = self.logic.generatePointCloud()
    self.assertIsNone(modelNode.GetPolyData().GetPointData().GetScalars())
    self.assertFalse(modelNode.GetDisplayNode().GetScalarVisibility())

    # Camera transform
    cameraTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "CameraPose")
    parameterNode.cameraTransform = cameraTransform
    self.assertEqual(self.logic.generatePointCloud().GetTransformNodeID(), cameraTransform.GetID())
    parameterNode.cameraTransform = None
    self.assertIsNone(self.logic.generatePointCloud().GetTransformNodeID())

  # -- Tests that run the model
  def test_GenerateDepthMap(self):
    """Checks generating the depth map of a single image.

    A generated grayscale image is used as input, with a size (160 x 120) different from the model's input size and
    a custom spacing and origin. The test checks that:
    - An output volume is created when none is selected.
    - The depth map is an 8-bit image with the same size, spacing, and origin as the input, so it overlays it.
    - The model actually produced an image (it is not constant), and running it again gives the same result.
    - The depth map makes sense: the centers of both holes in the test image (the airway lumens) are farther away
      than the walls, by at least 40 gray levels. A black, noisy, or inverted depth map fails this.
    - A color image gives the same depth map as its grayscale equivalent.
    - The inference time and device are reported.
    - With the point cloud enabled, a point cloud is generated along with the depth map.
    """
    self.delayDisplay("Testing single depth map generation")
    if not self.canRunModel("test_GenerateDepthMap"):
      return
    parameterNode = self.logic.getParameterNode()
    image = self.createTestImage(160, 120)
    inputVolume = self.createVolume(image[np.newaxis])
    inputVolume.SetSpacing(0.5, 0.5, 1.0)
    inputVolume.SetOrigin(10.0, -20.0, 5.0)
    parameterNode.inputVolume = inputVolume

    self.assertTrue(self.logic.generateDepthMap(askConfirmation=False))
    outputVolume = parameterNode.outputVolume
    self.assertIsNotNone(outputVolume)
    depthMap = slicer.util.arrayFromVolume(outputVolume).copy()
    self.assertEqual(depthMap.shape, (1, 120, 160))
    self.assertEqual(depthMap.dtype, np.uint8)
    np.testing.assert_allclose(outputVolume.GetSpacing(), inputVolume.GetSpacing())
    np.testing.assert_allclose(outputVolume.GetOrigin(), inputVolume.GetOrigin())
    self.assertGreater(int(depthMap.max()) - int(depthMap.min()), 0, "The depth map is constant")
    # The holes (airway lumens) must be farther away than the walls. This fails for depth maps that are black,
    # noise, or inverted. The model gives about 145 and 124 at the hole centers and 28 to 38 on the walls.
    def depthAt(row, column):
      return float(np.median(depthMap[0, row - 2:row + 3, column - 2:column + 3]))
    wallDepth = max(depthAt(8, 80), depthAt(111, 80))
    for holeName, holeDepth in [("large hole", depthAt(60, 48)), ("small hole", depthAt(60, 115))]:
      self.assertGreater(holeDepth, wallDepth + 40, f"The {holeName} ({holeDepth}) is not farther away than the wall ({wallDepth})")
    self.assertGreater(self.logic.lastInferenceTimeS, 0)
    self.assertIsNotNone(self.logic.deviceName)

    self.assertTrue(self.logic.generateDepthMap(askConfirmation=False))
    np.testing.assert_array_equal(slicer.util.arrayFromVolume(outputVolume), depthMap)

    colorVolume = self.createVolume(np.stack([image] * 3, axis=-1)[np.newaxis], "vtkMRMLVectorVolumeNode", "ColorImage")
    colorVolume.SetSpacing(0.5, 0.5, 1.0)
    colorVolume.SetOrigin(10.0, -20.0, 5.0)
    parameterNode.inputVolume = colorVolume
    self.assertIsNone(parameterNode.pointCloudModel)
    parameterNode.enablePointCloud = True
    self.assertTrue(self.logic.generateDepthMap(askConfirmation=False))
    np.testing.assert_allclose(slicer.util.arrayFromVolume(outputVolume), depthMap, atol=1)
    self.assertIsNotNone(parameterNode.pointCloudModel)
    self.assertGreater(parameterNode.pointCloudModel.GetPolyData().GetNumberOfPoints(), 0)

  def test_ContinuousMode(self):
    """Checks running continuously on an input that changes, as with a live video.

    The input image is changed every 10 ms (about 100 frames per second) for 2 seconds while the maximum frame rate
    is 4 frames per second. The test checks that:
    - Depth maps are generated while the input changes, but no more often than the maximum frame rate (allowing
      for timing variation; fewer are acceptable on a slow computer).
    - No depth maps are generated while the input stays the same.
    - Stopping stops the updates.
    - Starting on an input without an image waits for the first image instead of failing, and a 3D input stops
      continuous mode with an error.
    """
    self.delayDisplay("Testing continuous depth map generation")
    if not self.canRunModel("test_ContinuousMode"):
      return
    parameterNode = self.logic.getParameterNode()
    inputVolume = self.createVolume(self.createTestImage(200, 200)[np.newaxis])
    with slicer.util.NodeModify(parameterNode):
      parameterNode.inputVolume = inputVolume
      parameterNode.maxFrameRate = 4.0
    errors = []
    self.logic.statusCallback = lambda errorMessage=None: errors.append(errorMessage) if errorMessage else None

    self.assertTrue(self.logic.startContinuous(askConfirmation=False))
    self.assertTrue(self.logic.isRunning)
    self.assertTrue(self.waitFor(lambda: self.logic.processedFrameCount >= 1), "No depth map was generated")

    # Change the input continuously for 2 seconds
    startCount = self.logic.processedFrameCount
    endTime = time.perf_counter() + 2.0
    frameIndex = 0
    while time.perf_counter() < endTime:
      frameIndex += 1
      slicer.util.arrayFromVolume(inputVolume)[0, 0, 0] = frameIndex % 256
      slicer.util.arrayFromVolumeModified(inputVolume)
      self.waitFor(lambda: False, timeoutS=0.01)
    generatedCount = self.logic.processedFrameCount - startCount
    self.assertGreaterEqual(generatedCount, 1, "No depth maps were generated while the input changed")
    self.assertLessEqual(generatedCount, 10, f"{generatedCount} depth maps in 2 s exceeds the 4 fps limit")

    # No new depth maps while the input is unchanged
    self.waitFor(lambda: False, timeoutS=0.5)
    idleCount = self.logic.processedFrameCount
    self.waitFor(lambda: False, timeoutS=0.6)
    self.assertEqual(self.logic.processedFrameCount, idleCount, "Depth maps were generated for an unchanged input")

    self.logic.stopContinuous()
    self.assertFalse(self.logic.isRunning)
    stoppedCount = self.logic.processedFrameCount
    slicer.util.arrayFromVolumeModified(inputVolume)
    self.waitFor(lambda: False, timeoutS=0.3)
    self.assertEqual(self.logic.processedFrameCount, stoppedCount, "Depth maps were generated after stopping")

    # An input without an image is waited for
    emptyVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "EmptyInput")
    parameterNode.inputVolume = emptyVolume
    self.assertTrue(self.logic.startContinuous(askConfirmation=False))
    self.waitFor(lambda: False, timeoutS=0.2)
    self.assertTrue(self.logic.isRunning)
    # A 3D image stops continuous mode with an error
    slicer.util.updateVolumeFromArray(emptyVolume, np.zeros((3, 20, 20), dtype=np.uint8))
    self.assertTrue(self.waitFor(lambda: not self.logic.isRunning), "Continuous mode did not stop for a 3D input")
    self.assertTrue(any("2D" in error for error in errors))
