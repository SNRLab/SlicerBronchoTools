##=========================================================================

#  Program:   Video Input - Slicer Module

#  Copyright (c) Brigham and Women's Hospital. All rights reserved.

#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.

#=========================================================================
import os
import re
import enum
import fnmatch
import time
import logging
import platform
import tempfile
import threading
import importlib
import collections
import numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLVectorVolumeNode

class VideoInput(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Video Input"
    self.parent.categories = ["Utilities"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King (Brigham and Women's Hospital)"]
    self.parent.helpText = """
    Streams video from a webcam, video file, or image sequence into a volume node.
"""
    self.parent.acknowledgementText = """This module was developed by Franklin King (Brigham and Women's Hospital)."""
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

# ------------------------------------ Parameter Node ---------------------------------------
class InputType(enum.Enum):
  WEBCAM = 'Webcam'
  VIDEO = 'Video'
  IMAGES = 'Images'

  def label(self):
    return self.value

@parameterNodeWrapper
class VideoInputParameterNode:
  # Input / Output
  inputType: InputType = InputType.WEBCAM
  outputVolume: vtkMRMLVectorVolumeNode | vtkMRMLScalarVolumeNode | None

  # Webcam
  webcamDeviceIndex: int = 0
  webcamDeviceName: str = ''

  # Video
  videoFilePath: str = ''
  videoPlaybackSpeed: float = 1.0
  videoLoop: bool = True

  # Images
  imagesDirectoryPath: str = ''
  imagesFileFilter: str = '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'
  imagesFrameRate: int = 30
  imagesLoop: bool = True

  # Global (applied in this order: flip, crop, rotation and zoom about the center of the cropped image, then position)
  flipHorizontal: bool = False
  flipVertical: bool = False
  # Crop as pixel margins removed from each edge, so 0 means no crop regardless of the input resolution
  cropLeft: int = 0
  cropRight: int = 0
  cropTop: int = 0
  cropBottom: int = 0
  # Degrees, positive is clockwise
  rotation: float = 0.0
  zoom: float = 1.0
  # Pixel offset of the image in the output frame, positive moves the image right / up
  positionHorizontal: int = 0
  positionVertical: int = 0

# Global parameters restored by the Reset button
GLOBAL_PARAMETER_DEFAULTS = {
  'flipHorizontal': False,
  'flipVertical': False,
  'cropLeft': 0,
  'cropRight': 0,
  'cropTop': 0,
  'cropBottom': 0,
  'rotation': 0.0,
  'zoom': 1.0,
  'positionHorizontal': 0,
  'positionVertical': 0,
}
# -------------------------------------- ----------  --------------------------------------

class VideoInputWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  PLAY_BUTTON_STYLE = "QPushButton {background-color: #2E7D32; color: white;} QPushButton:hover {background-color: #388E3C;}"
  PAUSE_BUTTON_STYLE = "QPushButton {background-color: #F9A825; color: black;} QPushButton:hover {background-color: #FBC02D;}"

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._parameterNodeGuiTag = None
    self.inputGroupBoxes = {}
    self.inputTypeRadioButtons = {}

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = VideoInputLogic()
    self.logic.statusCallback = self.onLogicStatusChanged

    # ------------------------------------ IO ---------------------------------------
    self.inputOutputCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputOutputCollapsibleButton.text = "Input / Output"
    self.inputOutputCollapsibleButton.collapsed = False
    self.layout.addWidget(self.inputOutputCollapsibleButton)
    inputOutputLayout = qt.QFormLayout(self.inputOutputCollapsibleButton)

    inputTypeWidget = qt.QWidget()
    inputTypeLayout = qt.QHBoxLayout(inputTypeWidget)
    inputTypeLayout.setContentsMargins(0, 0, 0, 0)
    self.inputTypeButtonGroup = qt.QButtonGroup(inputTypeWidget)
    for inputType in InputType:
      radioButton = qt.QRadioButton(inputType.label())
      self.inputTypeButtonGroup.addButton(radioButton)
      inputTypeLayout.addWidget(radioButton)
      self.inputTypeRadioButtons[inputType] = radioButton
    self.inputTypeButtonGroup.connect('buttonClicked(QAbstractButton*)', self.onInputTypeClicked)
    inputOutputLayout.addRow("Input Type:", inputTypeWidget)

    # -- Webcam input parameters --
    self.webcamGroupBox = qt.QGroupBox()
    webcamLayout = qt.QFormLayout(self.webcamGroupBox)
    inputOutputLayout.addRow(self.webcamGroupBox)
    self.inputGroupBoxes[InputType.WEBCAM] = self.webcamGroupBox

    self.webcamComboBox = qt.QComboBox()
    self.webcamComboBox.setToolTip("Camera to stream from")
    self.webcamComboBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    self.webcamComboBox.connect('activated(int)', self.onWebcamSelected)
    self.webcamRefreshButton = qt.QPushButton("Refresh")
    self.webcamRefreshButton.setToolTip("Detect cameras connected to this computer")
    self.webcamRefreshButton.connect('clicked()', lambda: self.populateWebcamComboBox(installIfMissing=True))
    webcamSelectionLayout = qt.QHBoxLayout()
    webcamSelectionLayout.addWidget(self.webcamComboBox)
    webcamSelectionLayout.addWidget(self.webcamRefreshButton)
    webcamLayout.addRow("Camera:", webcamSelectionLayout)

    self.webcamResolutionLabel = qt.QLabel()
    self.webcamResolutionLabel.setToolTip("Resolution of the frames coming from the camera")
    webcamLayout.addRow("Resolution:", self.webcamResolutionLabel)

    self.webcamFrameRateLabel = qt.QLabel()
    self.webcamFrameRateLabel.setToolTip("Frame rate coming from the camera")
    webcamLayout.addRow("Frame Rate:", self.webcamFrameRateLabel)

    # -- Video input parameters --
    self.videoGroupBox = qt.QGroupBox()
    videoLayout = qt.QFormLayout(self.videoGroupBox)
    inputOutputLayout.addRow(self.videoGroupBox)
    self.inputGroupBoxes[InputType.VIDEO] = self.videoGroupBox

    self.videoPathLineEdit = ctk.ctkPathLineEdit()
    self.videoPathLineEdit.filters = ctk.ctkPathLineEdit.Files
    self.videoPathLineEdit.nameFilters = ["Videos (*.mp4 *.avi *.mov *.mkv *.wmv)", "All files (*)"]
    self.videoPathLineEdit.settingKey = "VideoInput/VideoPathHistory"
    self.videoPathLineEdit.connect('currentPathChanged(QString)', self.onVideoPathChanged)
    videoLayout.addRow("Video File:", self.videoPathLineEdit)

    self.videoResolutionLabel = qt.QLabel()
    self.videoResolutionLabel.setToolTip("Resolution of the frames in the video file")
    videoLayout.addRow("Resolution:", self.videoResolutionLabel)

    self.videoFrameRateLabel = qt.QLabel()
    self.videoFrameRateLabel.setToolTip("Native frame rate of the video file, and the measured rate while playing")
    videoLayout.addRow("Frame Rate:", self.videoFrameRateLabel)

    self.videoSeekSlider = qt.QSlider(qt.Qt.Horizontal)
    self.videoSeekSlider.setToolTip("Current position in the video. Drag to seek, also while paused.")
    self.videoSeekSlider.connect('valueChanged(int)', self.onSeek)
    self.videoTimeLabel = qt.QLabel()
    videoSeekLayout = qt.QHBoxLayout()
    videoSeekLayout.addWidget(self.videoSeekSlider)
    videoSeekLayout.addWidget(self.videoTimeLabel)
    videoLayout.addRow("Position:", videoSeekLayout)

    self.videoPlaybackSpeedSliderWidget = ctk.ctkSliderWidget()
    self.videoPlaybackSpeedSliderWidget.setToolTip("Playback speed relative to the video's native frame rate")
    self.videoPlaybackSpeedSliderWidget.setDecimals(2)
    self.videoPlaybackSpeedSliderWidget.minimum = 0.1
    self.videoPlaybackSpeedSliderWidget.maximum = 4.0
    self.videoPlaybackSpeedSliderWidget.singleStep = 0.05
    self.videoPlaybackSpeedSliderWidget.suffix = "x"
    videoLayout.addRow("Playback Speed:", self.videoPlaybackSpeedSliderWidget)

    self.videoLoopCheckBox = qt.QCheckBox("Loop video")
    videoLayout.addRow(self.videoLoopCheckBox)

    # -- Image input parameters --
    self.imagesGroupBox = qt.QGroupBox()
    imagesLayout = qt.QFormLayout(self.imagesGroupBox)
    inputOutputLayout.addRow(self.imagesGroupBox)
    self.inputGroupBoxes[InputType.IMAGES] = self.imagesGroupBox

    self.imagesDirectoryButton = ctk.ctkDirectoryButton()
    self.imagesDirectoryButton.setToolTip("Directory containing the image sequence")
    self.imagesDirectoryButton.connect('directoryChanged(QString)', self.onImagesDirectoryChanged)
    imagesLayout.addRow("Image Directory:", self.imagesDirectoryButton)

    self.imagesFileFilterLineEdit = qt.QLineEdit()
    self.imagesFileFilterLineEdit.setToolTip("Space-separated file patterns used to select images in the directory. "
                                             "Images are played in file name order, with numbers sorted by value.")
    self.imagesFileFilterLineEdit.connect('editingFinished()', self.onPlaybackInputChanged)
    imagesLayout.addRow("File Filter:", self.imagesFileFilterLineEdit)

    self.imagesResolutionLabel = qt.QLabel()
    self.imagesResolutionLabel.setToolTip("Resolution of the first image")
    imagesLayout.addRow("Resolution:", self.imagesResolutionLabel)

    self.imagesSeekSlider = qt.QSlider(qt.Qt.Horizontal)
    self.imagesSeekSlider.setToolTip("Current image in the sequence. Drag to seek, also while paused.")
    self.imagesSeekSlider.connect('valueChanged(int)', self.onSeek)
    self.imagesPositionLabel = qt.QLabel()
    imagesSeekLayout = qt.QHBoxLayout()
    imagesSeekLayout.addWidget(self.imagesSeekSlider)
    imagesSeekLayout.addWidget(self.imagesPositionLabel)
    imagesLayout.addRow("Position:", imagesSeekLayout)

    self.imagesCurrentFileLabel = qt.QLabel()
    self.imagesCurrentFileLabel.setToolTip("File name of the image currently shown")
    # Long file names must not widen the module panel
    self.imagesCurrentFileLabel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    imagesLayout.addRow("Current Image:", self.imagesCurrentFileLabel)

    self.imagesFrameRateSpinBox = qt.QSpinBox()
    self.imagesFrameRateSpinBox.setToolTip("Frame rate for image playback. Can be changed while playing.")
    self.imagesFrameRateSpinBox.setRange(1, 240)
    self.imagesFrameRateSpinBox.suffix = " fps"
    self.imagesMeasuredFrameRateLabel = qt.QLabel()
    self.imagesMeasuredFrameRateLabel.setToolTip("Frame rate measured while playing")
    imagesFrameRateLayout = qt.QHBoxLayout()
    imagesFrameRateLayout.addWidget(self.imagesFrameRateSpinBox)
    imagesFrameRateLayout.addWidget(self.imagesMeasuredFrameRateLabel, 1)
    imagesLayout.addRow("Frame Rate:", imagesFrameRateLayout)

    self.imagesLoopCheckBox = qt.QCheckBox("Loop image sequence")
    imagesLayout.addRow(self.imagesLoopCheckBox)

    # Output volume
    self.outputVolumeSelector = slicer.qMRMLNodeComboBox()
    self.outputVolumeSelector.nodeTypes = ["vtkMRMLVectorVolumeNode", "vtkMRMLScalarVolumeNode"]
    self.outputVolumeSelector.selectNodeUponCreation = True
    self.outputVolumeSelector.addEnabled = True
    self.outputVolumeSelector.removeEnabled = True
    self.outputVolumeSelector.renameEnabled = True
    self.outputVolumeSelector.noneEnabled = True
    self.outputVolumeSelector.showHidden = False
    self.outputVolumeSelector.showChildNodeTypes = False
    self.outputVolumeSelector.baseName = "VideoInput"
    self.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.outputVolumeSelector.setToolTip("Volume that incoming frames are written to. A vector volume receives RGB frames "
                                         "and a scalar volume receives grayscale frames. If none is selected, a vector volume is created on Play.")
    inputOutputLayout.addRow("Output Volume:", self.outputVolumeSelector)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Play / Pause UI ---------------------------------------
    playFont = qt.QFont()
    playFont.setPointSize(14)
    self.playButton = qt.QPushButton()
    self.playButton.setCheckable(True)
    self.playButton.setFont(playFont)
    self.playButton.setMinimumHeight(40)
    self.playButton.setToolTip("Start or pause streaming into the output volume")
    self.playButton.connect('clicked()', self.onPlayButtonClicked)
    self.layout.addWidget(self.playButton)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Global Parameters UI ---------------------------------------
    self.globalParametersCollapsibleButton = ctk.ctkCollapsibleButton()
    self.globalParametersCollapsibleButton.text = "Parameters"
    self.globalParametersCollapsibleButton.collapsed = False
    self.layout.addWidget(self.globalParametersCollapsibleButton)
    globalParametersLayout = qt.QFormLayout(self.globalParametersCollapsibleButton)

    self.flipHorizontalCheckBox = qt.QCheckBox("Flip horizontally")
    self.flipVerticalCheckBox = qt.QCheckBox("Flip vertically")
    flipLayout = qt.QHBoxLayout()
    flipLayout.addWidget(self.flipHorizontalCheckBox)
    flipLayout.addWidget(self.flipVerticalCheckBox)
    globalParametersLayout.addRow(flipLayout)

    # Crop and position widgets are synced manually: their limits follow the frame size, which the
    # parameter node connectors cannot do (they fix the limits when they connect)
    self.cropHorizontalRangeWidget = ctk.ctkRangeWidget()
    self.cropHorizontalRangeWidget.setToolTip("Left and right pixel columns of the region to keep")
    self.cropHorizontalRangeWidget.decimals = 0
    self.cropHorizontalRangeWidget.singleStep = 1
    self.cropHorizontalRangeWidget.suffix = " px"
    self.cropHorizontalRangeWidget.connect('valuesChanged(double,double)', self.onCropChanged)
    globalParametersLayout.addRow("Crop Horizontal:", self.cropHorizontalRangeWidget)

    self.cropVerticalRangeWidget = ctk.ctkRangeWidget()
    self.cropVerticalRangeWidget.setToolTip("Top and bottom pixel rows of the region to keep")
    self.cropVerticalRangeWidget.decimals = 0
    self.cropVerticalRangeWidget.singleStep = 1
    self.cropVerticalRangeWidget.suffix = " px"
    self.cropVerticalRangeWidget.connect('valuesChanged(double,double)', self.onCropChanged)
    globalParametersLayout.addRow("Crop Vertical:", self.cropVerticalRangeWidget)

    self.rotationSliderWidget = ctk.ctkSliderWidget()
    self.rotationSliderWidget.setToolTip("Rotation about the center of the (cropped) image. Positive is clockwise.")
    self.rotationSliderWidget.setDecimals(1)
    self.rotationSliderWidget.minimum = -180.0
    self.rotationSliderWidget.maximum = 180.0
    self.rotationSliderWidget.singleStep = 0.5
    self.rotationSliderWidget.suffix = "°"
    globalParametersLayout.addRow("Rotation:", self.rotationSliderWidget)

    self.zoomSliderWidget = ctk.ctkSliderWidget()
    self.zoomSliderWidget.setToolTip("Magnification about the center of the (cropped) image")
    self.zoomSliderWidget.setDecimals(2)
    self.zoomSliderWidget.minimum = 1.0
    self.zoomSliderWidget.maximum = 10.0
    self.zoomSliderWidget.singleStep = 0.05
    self.zoomSliderWidget.suffix = "x"
    globalParametersLayout.addRow("Zoom:", self.zoomSliderWidget)

    self.positionHorizontalSliderWidget = ctk.ctkSliderWidget()
    self.positionHorizontalSliderWidget.setToolTip("Moves the image horizontally in the output frame. Positive is right.")
    self.positionHorizontalSliderWidget.decimals = 0
    self.positionHorizontalSliderWidget.singleStep = 1
    self.positionHorizontalSliderWidget.suffix = " px"
    self.positionHorizontalSliderWidget.connect('valueChanged(double)', self.onPositionChanged)
    globalParametersLayout.addRow("Horizontal Position:", self.positionHorizontalSliderWidget)

    self.positionVerticalSliderWidget = ctk.ctkSliderWidget()
    self.positionVerticalSliderWidget.setToolTip("Moves the image vertically in the output frame. Positive is up.")
    self.positionVerticalSliderWidget.decimals = 0
    self.positionVerticalSliderWidget.singleStep = 1
    self.positionVerticalSliderWidget.suffix = " px"
    self.positionVerticalSliderWidget.connect('valueChanged(double)', self.onPositionChanged)
    globalParametersLayout.addRow("Vertical Position:", self.positionVerticalSliderWidget)

    self.outputSizeLabel = qt.QLabel()
    self.outputSizeLabel.setToolTip("Size of the frames written to the output volume")
    globalParametersLayout.addRow("Output Size:", self.outputSizeLabel)

    self.resetGlobalParametersButton = qt.QPushButton("Reset to Default")
    self.resetGlobalParametersButton.setToolTip("Reset flips, crop, rotation, zoom, and position")
    self.resetGlobalParametersButton.connect('clicked()', self.onResetGlobalParameters)
    globalParametersLayout.addRow(self.resetGlobalParametersButton)
    # -------------------------------------- ----------  --------------------------------------

    self.layout.addStretch(1)

    # Widgets bound directly to parameter node values
    self.parameterWidgets = {
      'outputVolume': self.outputVolumeSelector,
      'videoLoop': self.videoLoopCheckBox,
      'imagesFileFilter': self.imagesFileFilterLineEdit,
      'imagesLoop': self.imagesLoopCheckBox,
      'flipHorizontal': self.flipHorizontalCheckBox,
      'flipVertical': self.flipVerticalCheckBox,
    }
    # Widgets with ranges set above are synced manually, since binding them would replace their ranges
    self.valueWidgets = {
      'videoPlaybackSpeed': self.videoPlaybackSpeedSliderWidget,
      'imagesFrameRate': self.imagesFrameRateSpinBox,
      'rotation': self.rotationSliderWidget,
      'zoom': self.zoomSliderWidget,
    }
    for name, widget in self.valueWidgets.items():
      widget.connect('valueChanged(int)' if isinstance(widget, qt.QSpinBox) else 'valueChanged(double)',
                     lambda value, name=name: self.onValueWidgetChanged(name, value))

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    self.initializeParameterNode()
    self.populateWebcamComboBox(installIfMissing=False)
    self.loadPlaybackInfo()

  def cleanup(self):
    if self.logic:
      self.logic.stop()
      self.logic.statusCallback = None
    self.removeObservers()

  def enter(self):
    self.initializeParameterNode()

  def exit(self):
    self.setParameterNode(None)

  def onSceneStartClose(self, caller, event):
    self.logic.stop()
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    self.setParameterNode(self.logic.getParameterNode())

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
      return

    selectedInputType = self._parameterNode.inputType
    for inputType, radioButton in self.inputTypeRadioButtons.items():
      radioButton.setChecked(inputType == selectedInputType)
    for inputType, groupBox in self.inputGroupBoxes.items():
      groupBox.setVisible(inputType == selectedInputType)

    self.selectWebcamComboBoxItem(self._parameterNode.webcamDeviceName, self._parameterNode.webcamDeviceIndex)

    if self.videoPathLineEdit.currentPath != self._parameterNode.videoFilePath:
      wasBlocked = self.videoPathLineEdit.blockSignals(True)
      self.videoPathLineEdit.currentPath = self._parameterNode.videoFilePath
      self.videoPathLineEdit.blockSignals(wasBlocked)

    if self._parameterNode.imagesDirectoryPath and self.imagesDirectoryButton.directory != self._parameterNode.imagesDirectoryPath:
      wasBlocked = self.imagesDirectoryButton.blockSignals(True)
      self.imagesDirectoryButton.directory = self._parameterNode.imagesDirectoryPath
      self.imagesDirectoryButton.blockSignals(wasBlocked)

    for name, widget in self.valueWidgets.items():
      value = getattr(self._parameterNode, name)
      if widget.value != value:
        wasBlocked = widget.blockSignals(True)
        widget.value = value
        widget.blockSignals(wasBlocked)

    self.updateCropWidgets()
    self.updatePlayButton()

  def onValueWidgetChanged(self, name, value):
    if self._parameterNode:
      setattr(self._parameterNode, name, type(getattr(self._parameterNode, name))(value))

  def onInputTypeClicked(self, button):
    if not self._parameterNode:
      return
    for inputType, radioButton in self.inputTypeRadioButtons.items():
      if radioButton is button and self._parameterNode.inputType != inputType:
        self.logic.clearInput()
        self._parameterNode.inputType = inputType
        self.loadPlaybackInfo()
        break

  def onVideoPathChanged(self, path):
    if self._parameterNode and self._parameterNode.videoFilePath != path:
      self._parameterNode.videoFilePath = path
      self.onPlaybackInputChanged()

  def onImagesDirectoryChanged(self, path):
    if self._parameterNode and self._parameterNode.imagesDirectoryPath != path:
      self._parameterNode.imagesDirectoryPath = path
      self.onPlaybackInputChanged()

  def onPlaybackInputChanged(self):
    """Called when the video file, image directory, or image file filter may have changed."""
    if self._parameterNode and self.logic.playbackInfoKey != self.logic.getPlaybackKey():
      self.logic.clearInput()
      self.loadPlaybackInfo()

  # ------------------------------------ Webcam ---------------------------------------
  def populateWebcamComboBox(self, installIfMissing):
    wasBlocked = self.webcamComboBox.blockSignals(True)
    self.webcamComboBox.clear()
    self.webcamComboBox.blockSignals(wasBlocked)

    if not self.logic.areDependenciesInstalled():
      if not installIfMissing or not self.logic.installDependencies():
        self.setWebcamComboBoxPlaceholder("Press Refresh to detect cameras")
        return

    with slicer.util.tryWithErrorDisplay("Failed to detect cameras.", waitCursor=True):
      cameras = self.logic.getAvailableWebcams()
      if not cameras:
        self.setWebcamComboBoxPlaceholder("No cameras detected")
        return
      wasBlocked = self.webcamComboBox.blockSignals(True)
      self.webcamComboBox.enabled = True
      for deviceIndex, deviceName in cameras:
        self.webcamComboBox.addItem(deviceName, deviceIndex)
      self.webcamComboBox.blockSignals(wasBlocked)

      if self._parameterNode:
        self.selectWebcamComboBoxItem(self._parameterNode.webcamDeviceName, self._parameterNode.webcamDeviceIndex)
        # Store the actual selection in case the previously used camera is no longer available
        self.onWebcamSelected(self.webcamComboBox.currentIndex)

  def setWebcamComboBoxPlaceholder(self, text):
    wasBlocked = self.webcamComboBox.blockSignals(True)
    self.webcamComboBox.clear()
    self.webcamComboBox.addItem(text, None)
    self.webcamComboBox.enabled = False
    self.webcamComboBox.blockSignals(wasBlocked)

  def selectWebcamComboBoxItem(self, deviceName, deviceIndex):
    if not self.webcamComboBox.enabled:
      return
    itemIndex = self.webcamComboBox.findText(deviceName) if deviceName else -1
    if itemIndex < 0:
      itemIndex = self.webcamComboBox.findData(deviceIndex)
    if itemIndex < 0:
      itemIndex = 0
    if self.webcamComboBox.currentIndex != itemIndex:
      wasBlocked = self.webcamComboBox.blockSignals(True)
      self.webcamComboBox.currentIndex = itemIndex
      self.webcamComboBox.blockSignals(wasBlocked)

  def onWebcamSelected(self, itemIndex):
    deviceIndex = self.webcamComboBox.itemData(itemIndex)
    if not self._parameterNode or deviceIndex is None:
      return
    deviceName = self.webcamComboBox.itemText(itemIndex)
    if self._parameterNode.webcamDeviceIndex == deviceIndex and self._parameterNode.webcamDeviceName == deviceName:
      return
    wasPlaying = self.logic.isPlaying
    self.logic.stop()
    with slicer.util.NodeModify(self._parameterNode):
      self._parameterNode.webcamDeviceIndex = deviceIndex
      self._parameterNode.webcamDeviceName = deviceName
    if wasPlaying:
      self.onPlayButtonClicked()

  # ------------------------------------ Video / Images ---------------------------------------
  def loadPlaybackInfo(self):
    """Reads the size and length of the selected video or image sequence so they can be shown before playing."""
    if self._parameterNode and self.logic.getPlaybackKey() is not None and self.logic.areDependenciesInstalled():
      try:
        self.logic.loadPlaybackInfo()
      except Exception as e:
        logging.warning(f"Could not read input information: {e}")
    self.onLogicStatusChanged()

  def onSeek(self, frameIndex):
    if not self._parameterNode or not self.logic.playbackInfo:
      return
    with slicer.util.tryWithErrorDisplay("Failed to seek.", waitCursor=False):
      self.logic.seek(frameIndex)

  def updatePlaybackWidgets(self):
    """Updates the information and position widgets of the video or image sequence."""
    inputType = self._parameterNode.inputType if self._parameterNode else None
    playbackInfo = self.logic.playbackInfo
    isPlaying = self.logic.isPlaying
    measuredFrameRate = self.logic.measuredFrameRate if isPlaying else None
    frameIndex = self.logic.playbackFrameIndex

    # Video
    videoInfo = playbackInfo if inputType == InputType.VIDEO else None
    if videoInfo:
      self.videoResolutionLabel.text = f"{videoInfo['width']} x {videoInfo['height']} px"
      frameRateText = f"{videoInfo['frameRate']:.2f} fps"
      if measuredFrameRate:
        frameRateText += f" (playing at {measuredFrameRate:.1f} fps)"
      self.videoFrameRateLabel.text = frameRateText
      frameRate = videoInfo['frameRate']
      self.videoTimeLabel.text = (f"{self.formatTime(frameIndex / frameRate)} / "
                                  f"{self.formatTime(videoInfo['frameCount'] / frameRate)}")
    else:
      self.videoResolutionLabel.text = "-"
      self.videoFrameRateLabel.text = "-"
      self.videoTimeLabel.text = "-"
    self.setSeekSlider(self.videoSeekSlider, videoInfo, frameIndex)

    # Images
    imagesInfo = playbackInfo if inputType == InputType.IMAGES else None
    if imagesInfo:
      self.imagesResolutionLabel.text = f"{imagesInfo['width']} x {imagesInfo['height']} px"
      self.imagesPositionLabel.text = f"{frameIndex + 1} / {imagesInfo['frameCount']}"
    else:
      self.imagesResolutionLabel.text = "-"
      self.imagesPositionLabel.text = "-"
    currentImagePath = self.logic.currentImagePath
    self.imagesCurrentFileLabel.text = os.path.basename(currentImagePath) if currentImagePath else "-"
    self.imagesMeasuredFrameRateLabel.text = f"(playing at {measuredFrameRate:.1f} fps)" if imagesInfo and measuredFrameRate else ""
    self.setSeekSlider(self.imagesSeekSlider, imagesInfo, frameIndex)

  @staticmethod
  def setSeekSlider(slider, playbackInfo, frameIndex):
    frameCount = playbackInfo['frameCount'] if playbackInfo else 0
    slider.enabled = frameCount > 1
    if slider.isSliderDown():
      return  # Do not move the slider while the user drags it
    wasBlocked = slider.blockSignals(True)
    slider.maximum = max(frameCount - 1, 0)
    slider.value = frameIndex if playbackInfo else 0
    slider.blockSignals(wasBlocked)

  @staticmethod
  def formatTime(seconds):
    minutes, seconds = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"

  # ------------------------------------ Play / Pause ---------------------------------------
  def onPlayButtonClicked(self):
    if self.logic.isPlaying:
      self.logic.pause()
      return
    if not self.logic.areDependenciesInstalled():
      if not self.logic.installDependencies():
        self.updatePlayButton()
        return
      self.populateWebcamComboBox(installIfMissing=False)
    with slicer.util.tryWithErrorDisplay("Failed to start streaming.", waitCursor=True):
      self.logic.start()
    self.updatePlayButton()

  def onLogicStatusChanged(self, errorMessage=None):
    isWebcam = self._parameterNode is not None and self._parameterNode.inputType == InputType.WEBCAM
    if isWebcam and self.logic.frameSize:
      self.webcamResolutionLabel.text = f"{self.logic.frameSize[0]} x {self.logic.frameSize[1]} px"
    else:
      self.webcamResolutionLabel.text = "-"
    if isWebcam and self.logic.isPlaying and self.logic.measuredFrameRate:
      self.webcamFrameRateLabel.text = f"{self.logic.measuredFrameRate:.1f} fps"
    else:
      self.webcamFrameRateLabel.text = "-"
    self.updatePlaybackWidgets()
    if self.logic.outputFrameSize:
      self.outputSizeLabel.text = f"{self.logic.outputFrameSize[0]} x {self.logic.outputFrameSize[1]} px"
    else:
      self.outputSizeLabel.text = "-"
    self.updateCropWidgets()
    self.updatePlayButton()
    if errorMessage:
      slicer.util.errorDisplay(errorMessage)

  # ------------------------------------ Crop / Position ---------------------------------------
  def updateCropWidgets(self):
    """Sets the crop and position widget limits from the frame size and their values from the parameter node."""
    frameSize = self.logic.frameSize if self.logic else None
    hasFrameSize = frameSize is not None and self._parameterNode is not None
    frameSizeWidgets = (self.cropHorizontalRangeWidget, self.cropVerticalRangeWidget,
                        self.positionHorizontalSliderWidget, self.positionVerticalSliderWidget)
    for widget in frameSizeWidgets:
      widget.enabled = hasFrameSize
      widget.toolTip = widget.toolTip.split("\n")[0] + ("" if hasFrameSize else "\nAvailable once the input has been played so its frame size is known")
    if not hasFrameSize:
      return

    width, height = frameSize
    self.setCropWidgetValues(self.cropHorizontalRangeWidget, width,
                             self._parameterNode.cropLeft, width - self._parameterNode.cropRight)
    self.setCropWidgetValues(self.cropVerticalRangeWidget, height,
                             self._parameterNode.cropTop, height - self._parameterNode.cropBottom)

    # The image can be moved by up to the size of the cropped frame in each direction
    left, right, top, bottom = VideoInputLogic.getCropBounds(width, height, self._parameterNode)
    self.setPositionWidgetValue(self.positionHorizontalSliderWidget, right - left, self._parameterNode.positionHorizontal)
    self.setPositionWidgetValue(self.positionVerticalSliderWidget, bottom - top, self._parameterNode.positionVertical)

  @staticmethod
  def setCropWidgetValues(widget, size, start, end):
    if widget.maximum == size and widget.minimumValue == start and widget.maximumValue == end:
      return
    wasBlocked = widget.blockSignals(True)
    widget.setRange(0, size)
    widget.setValues(start, end)
    widget.blockSignals(wasBlocked)

  @staticmethod
  def setPositionWidgetValue(widget, size, value):
    if widget.maximum == size and widget.value == value:
      return
    wasBlocked = widget.blockSignals(True)
    widget.minimum = -size
    widget.maximum = size
    widget.value = value
    widget.blockSignals(wasBlocked)

  def onPositionChanged(self, value=None):
    if not self._parameterNode or not self.logic.frameSize:
      return
    with slicer.util.NodeModify(self._parameterNode):
      self._parameterNode.positionHorizontal = int(round(self.positionHorizontalSliderWidget.value))
      self._parameterNode.positionVertical = int(round(self.positionVerticalSliderWidget.value))

  def onCropChanged(self, minimumValue=None, maximumValue=None):
    if not self._parameterNode or not self.logic.frameSize:
      return
    width, height = self.logic.frameSize
    with slicer.util.NodeModify(self._parameterNode):
      self._parameterNode.cropLeft = int(round(self.cropHorizontalRangeWidget.minimumValue))
      self._parameterNode.cropRight = width - int(round(self.cropHorizontalRangeWidget.maximumValue))
      self._parameterNode.cropTop = int(round(self.cropVerticalRangeWidget.minimumValue))
      self._parameterNode.cropBottom = height - int(round(self.cropVerticalRangeWidget.maximumValue))

  def onResetGlobalParameters(self):
    if not self._parameterNode:
      return
    with slicer.util.NodeModify(self._parameterNode):
      for name, value in GLOBAL_PARAMETER_DEFAULTS.items():
        setattr(self._parameterNode, name, value)

  def updatePlayButton(self):
    isPlaying = self.logic.isPlaying
    wasBlocked = self.playButton.blockSignals(True)
    self.playButton.checked = isPlaying
    self.playButton.blockSignals(wasBlocked)
    style = slicer.app.style()
    if isPlaying:
      self.playButton.text = " Pause"
      self.playButton.icon = style.standardIcon(qt.QStyle.SP_MediaPause)
    else:
      self.playButton.text = " Play"
      self.playButton.icon = style.standardIcon(qt.QStyle.SP_MediaPlay)

    self.playButton.styleSheet = self.PAUSE_BUTTON_STYLE if isPlaying else self.PLAY_BUTTON_STYLE

# ------------------------------------ Frame Sources ---------------------------------------
class FrameSource:
  """Base class for sources that read frames on a background thread so the GUI stays responsive.
  Only the most recent frame is kept; older frames are dropped if Slicer cannot keep up."""

  FRAME_RATE_MEASUREMENT_WINDOW = 30

  def __init__(self):
    self._lock = threading.Lock()
    self._latestFrame = None
    self._frameTimes = collections.deque(maxlen=self.FRAME_RATE_MEASUREMENT_WINDOW)
    self.failed = False
    self.failureMessage = ""
    self._running = True
    self._thread = None

  def _startThread(self, name):
    self._thread = threading.Thread(target=self._run, name=name, daemon=True)
    self._thread.start()

  def _run(self):
    raise NotImplementedError

  def _release(self):
    """Releases the camera or file once the thread has stopped."""
    pass

  def _publishFrame(self, frame, countForFrameRate=True):
    with self._lock:
      self._latestFrame = frame
      if countForFrameRate:
        self._frameTimes.append(time.perf_counter())

  def _fail(self, message):
    self.failureMessage = message
    self.failed = True

  @property
  def isPlaying(self):
    return self._running and not self.failed

  def getLatestFrame(self):
    """Returns the newest BGR frame since the last call, or None if there is no new frame."""
    with self._lock:
      frame = self._latestFrame
      self._latestFrame = None
    return frame

  def getMeasuredFrameRate(self):
    with self._lock:
      if len(self._frameTimes) < 2:
        return None
      elapsed = self._frameTimes[-1] - self._frameTimes[0]
      return (len(self._frameTimes) - 1) / elapsed if elapsed > 0 else None

  def _clearFrameRate(self):
    with self._lock:
      self._frameTimes.clear()

  def stop(self):
    self._running = False
    if self._thread:
      self._thread.join(timeout=2.0)
    self._release()

class WebcamSource(FrameSource):
  """Streams frames from a camera as fast as the camera delivers them."""

  MAX_CONSECUTIVE_FAILURES = 50

  def __init__(self, deviceIndex, backend):
    import cv2
    FrameSource.__init__(self)
    self.capture = cv2.VideoCapture(deviceIndex, backend)
    if not self.capture.isOpened():
      self.capture.release()
      raise RuntimeError(f"Unable to open camera {deviceIndex}. It may be in use by another application.")
    self._startThread("VideoInputWebcam")

  def _release(self):
    self.capture.release()

  def _run(self):
    consecutiveFailures = 0
    while self._running:
      ok, frame = self.capture.read()
      if not ok:
        consecutiveFailures += 1
        if consecutiveFailures >= self.MAX_CONSECUTIVE_FAILURES:
          self._fail("Lost connection to the camera.")
          return
        time.sleep(0.01)
        continue
      consecutiveFailures = 0
      self._publishFrame(frame)

class PlaybackSource(FrameSource):
  """Base class for recorded inputs (video files and image sequences) that play at a set frame rate.
  The source starts paused; it can be paused, resumed, and seeked without reopening the input.

  Subclasses set self.info ({'width', 'height', 'frameCount'}) and self.frameRate, then call
  _startThread(). They implement _setPosition() and _readNextFrame(), which run on the playback thread."""

  # If reading falls behind by more than this, restart the timing instead of trying to catch up
  MAX_LAG_S = 0.25
  READ_FAILURE_MESSAGE = "Unable to read frames from the input."

  def __init__(self, key):
    FrameSource.__init__(self)
    # Identifies the input (e.g. the file path) so the source can be reused while the input is unchanged
    self.key = key
    self.info = None
    self.frameRate = 30.0
    self.playbackSpeed = 1.0
    self.loop = True
    # Index of the frame that _readNextFrame() returns next
    self._nextFrameIndex = 0
    self._hasReadFrame = False
    self.reachedEnd = False
    self._playing = False
    self._seekRequest = None
    self._wakeEvent = threading.Event()

  def _setPosition(self, frameIndex):
    """Makes frameIndex the next frame read."""
    raise NotImplementedError

  def _readNextFrame(self):
    """Returns the next BGR frame and advances _nextFrameIndex, or returns None at the end of the input."""
    raise NotImplementedError

  @property
  def isPlaying(self):
    return self._playing and not self.failed

  @property
  def currentFrameIndex(self):
    """Index of the most recently read (displayed) frame."""
    return max(self._nextFrameIndex - 1, 0)

  def play(self):
    if self.reachedEnd:
      self.seek(0)
    self._clearFrameRate()
    self._playing = True
    self._wakeEvent.set()

  def pause(self):
    self._playing = False
    self._wakeEvent.set()

  def seek(self, frameIndex):
    """Moves to the given frame. The frame is shown even while paused."""
    lastFrameIndex = max(self.info['frameCount'] - 1, 0)
    with self._lock:
      self._seekRequest = min(max(int(frameIndex), 0), lastFrameIndex)
    self.reachedEnd = False
    self._wakeEvent.set()

  def stop(self):
    self._running = False
    self._wakeEvent.set()
    FrameSource.stop(self)

  def _readFrame(self):
    """Reads the next frame and publishes it. Returns False at the end of the input."""
    frame = self._readNextFrame()
    if frame is None:
      return False
    self._hasReadFrame = True
    self._publishFrame(frame, countForFrameRate=self._playing)
    return True

  def _run(self):
    nextFrameTime = None
    while self._running:
      with self._lock:
        seekRequest = self._seekRequest
        self._seekRequest = None
      if seekRequest is not None:
        # Read and show the requested frame; playing then continues from the frame after it
        self._setPosition(seekRequest)
        self._readFrame()
        nextFrameTime = None
        continue

      if not self._playing:
        self._wakeEvent.wait(0.1)
        self._wakeEvent.clear()
        nextFrameTime = None
        continue

      # Wait until the next frame is due, waking early on pause or seek
      now = time.perf_counter()
      if nextFrameTime is not None and nextFrameTime > now:
        self._wakeEvent.wait(nextFrameTime - now)
        self._wakeEvent.clear()
        continue

      if not self._readFrame():
        if not self._hasReadFrame:
          self._fail(self.READ_FAILURE_MESSAGE)
          return
        if self.loop:
          self._setPosition(0)
          continue
        self._playing = False
        self.reachedEnd = True
        continue

      frameInterval = 1.0 / (max(self.frameRate, 0.01) * max(self.playbackSpeed, 0.01))
      now = time.perf_counter()
      if nextFrameTime is None or now - nextFrameTime > self.MAX_LAG_S:
        nextFrameTime = now
      nextFrameTime += frameInterval

class VideoFileSource(PlaybackSource):
  """Plays a video file at its native frame rate multiplied by the playback speed."""

  READ_FAILURE_MESSAGE = "Unable to read frames from the video file. The video format may not be supported."

  def __init__(self, filePath):
    import cv2
    PlaybackSource.__init__(self, key=(InputType.VIDEO, filePath))
    self.capture = cv2.VideoCapture(filePath)
    if not self.capture.isOpened():
      self.capture.release()
      raise RuntimeError(f"Unable to open video file:\n{filePath}")
    self.info = self.getCaptureInfo(self.capture)
    self.frameRate = self.info['frameRate']
    self._startThread("VideoInputVideoFile")

  @staticmethod
  def getCaptureInfo(capture):
    import cv2
    frameRate = capture.get(cv2.CAP_PROP_FPS)
    if not frameRate or frameRate != frameRate or frameRate <= 0:  # Missing or NaN
      frameRate = 30.0
    return {
      'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
      'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      'frameRate': frameRate,
      'frameCount': max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
    }

  @staticmethod
  def readInfo(filePath):
    """Reads the video information without starting playback."""
    import cv2
    capture = cv2.VideoCapture(filePath)
    try:
      if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file:\n{filePath}")
      return VideoFileSource.getCaptureInfo(capture)
    finally:
      capture.release()

  def _release(self):
    self.capture.release()

  def _setPosition(self, frameIndex):
    import cv2
    self.capture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    self._nextFrameIndex = frameIndex

  def _readNextFrame(self):
    ok, frame = self.capture.read()
    if not ok:
      return None
    self._nextFrameIndex += 1
    return frame

class ImageSequenceSource(PlaybackSource):
  """Plays the images in a directory, in natural file name order (frame2 before frame10), at a set frame rate."""

  READ_FAILURE_MESSAGE = "Unable to read any of the images. The image format may not be supported."

  def __init__(self, directoryPath, fileFilter):
    PlaybackSource.__init__(self, key=(InputType.IMAGES, directoryPath, fileFilter))
    self.info = self.readInfo(directoryPath, fileFilter)
    self.filePaths = self.info['filePaths']
    self._startThread("VideoInputImageSequence")

  @staticmethod
  def findImageFiles(directoryPath, fileFilter):
    """Returns the paths of files in the directory (not subdirectories) matching any of the space-separated
    patterns, case-insensitively, in natural order."""
    if not directoryPath or not os.path.isdir(directoryPath):
      raise RuntimeError(f"Image directory not found:\n{directoryPath}")
    patterns = [pattern.lower() for pattern in fileFilter.split()] or ['*']
    fileNames = [entry.name for entry in os.scandir(directoryPath)
                 if entry.is_file() and any(fnmatch.fnmatchcase(entry.name.lower(), pattern) for pattern in patterns)]
    naturalSortKey = lambda name: [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]
    return [os.path.join(directoryPath, fileName) for fileName in sorted(fileNames, key=naturalSortKey)]

  @staticmethod
  def readImage(filePath):
    """Reads an image as an 8-bit BGR frame, or returns None if it cannot be read.
    Decoding from memory also works for paths that cv2.imread cannot open (non-ASCII characters on Windows)."""
    import cv2
    try:
      data = np.fromfile(filePath, dtype=np.uint8)
    except OSError:
      return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

  @staticmethod
  def readInfo(directoryPath, fileFilter):
    """Finds the images and reads the size of the first one without starting playback."""
    filePaths = ImageSequenceSource.findImageFiles(directoryPath, fileFilter)
    if not filePaths:
      raise RuntimeError(f"No images matching \"{fileFilter}\" found in:\n{directoryPath}")
    firstImage = None
    for filePath in filePaths[:10]:
      firstImage = ImageSequenceSource.readImage(filePath)
      if firstImage is not None:
        break
    if firstImage is None:
      raise RuntimeError(f"Unable to read the images in:\n{directoryPath}")
    return {
      'width': firstImage.shape[1],
      'height': firstImage.shape[0],
      'frameCount': len(filePaths),
      'filePaths': filePaths,
    }

  @property
  def currentFilePath(self):
    return self.filePaths[self.currentFrameIndex] if self._hasReadFrame else None

  def _setPosition(self, frameIndex):
    self._nextFrameIndex = frameIndex

  def _readNextFrame(self):
    # Images that cannot be read are skipped
    while self._nextFrameIndex < len(self.filePaths):
      filePath = self.filePaths[self._nextFrameIndex]
      self._nextFrameIndex += 1
      frame = self.readImage(filePath)
      if frame is not None:
        return frame
      logging.warning(f"Skipping unreadable image: {filePath}")
    return None

# ------------------------------------ Logic ---------------------------------------
class VideoInputLogic(ScriptedLoadableModuleLogic):
  REQUIRED_PACKAGES = {
    'cv2': 'opencv-python-headless',
    'cv2_enumerate_cameras': 'cv2_enumerate_cameras',
  }
  UPDATE_INTERVAL_MS = 5
  # Often enough for the video position slider to move smoothly
  STATUS_INTERVAL_S = 0.2

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self._parameterNode = None
    self.source = None
    self.frameSize = None
    self.outputFrameSize = None
    self.measuredFrameRate = None
    # Information about the selected video or image sequence, and the input it belongs to
    self.playbackInfo = None
    self.playbackInfoKey = None
    self._lastReportedPlaying = None
    self.statusCallback = None
    self._lastStatusTime = 0
    self._configuredVolume = None
    self._configuredFrameShape = None
    self.updateTimer = qt.QTimer()
    self.updateTimer.setInterval(self.UPDATE_INTERVAL_MS)
    self.updateTimer.connect('timeout()', self.onUpdateTimer)

  def getParameterNode(self):
    parameterNode = super().getParameterNode()
    if not self._parameterNode or self._parameterNode.parameterNode != parameterNode:
      self._parameterNode = VideoInputParameterNode(parameterNode)
    return self._parameterNode

  # -- Dependencies
  def areDependenciesInstalled(self):
    return all(importlib.util.find_spec(moduleName) is not None for moduleName in self.REQUIRED_PACKAGES)

  def installDependencies(self, askConfirmation=True):
    missingPackages = [packageName for moduleName, packageName in self.REQUIRED_PACKAGES.items()
                       if importlib.util.find_spec(moduleName) is None]
    if not missingPackages:
      return True
    if askConfirmation and not slicer.util.confirmOkCancelDisplay(
        "Video Input requires the following Python packages to read cameras, video files, and images:\n\n"
        + "\n".join(missingPackages) + "\n\nInstall them now?",
        "Install Python packages"):
      return False
    with slicer.util.tryWithErrorDisplay("Failed to install Python packages.", waitCursor=True):
      slicer.util.pip_install(missingPackages)
      importlib.invalidate_caches()
    return self.areDependenciesInstalled()

  # -- Webcams
  @staticmethod
  def getWebcamBackend():
    import cv2
    system = platform.system()
    if system == 'Windows':
      return cv2.CAP_DSHOW
    if system == 'Linux':
      return cv2.CAP_V4L2
    if system == 'Darwin':
      return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY

  def getAvailableWebcams(self):
    """Returns a list of (deviceIndex, deviceName) for cameras usable with getWebcamBackend()."""
    import cv2
    backend = self.getWebcamBackend()
    try:
      from cv2_enumerate_cameras import enumerate_cameras
      return [(camera.index, camera.name) for camera in enumerate_cameras(backend)]
    except Exception as e:
      # Fall back to probing indices, which works anywhere OpenCV does but cannot report camera names
      logging.warning(f"Camera enumeration failed ({e}), probing device indices instead")
      cameras = []
      for deviceIndex in range(5):
        capture = cv2.VideoCapture(deviceIndex, backend)
        if capture.isOpened():
          cameras.append((deviceIndex, f"Camera {deviceIndex}"))
        capture.release()
      return cameras

  # -- Video files and image sequences
  def getPlaybackKey(self):
    """Identifies the selected video file or image sequence, or returns None for a webcam or if nothing is selected."""
    parameterNode = self.getParameterNode()
    if parameterNode.inputType == InputType.VIDEO and parameterNode.videoFilePath:
      return (InputType.VIDEO, parameterNode.videoFilePath)
    if parameterNode.inputType == InputType.IMAGES and parameterNode.imagesDirectoryPath:
      return (InputType.IMAGES, parameterNode.imagesDirectoryPath, parameterNode.imagesFileFilter)
    return None

  def loadPlaybackInfo(self):
    """Reads the size and length of the selected video or image sequence without playing it,
    e.g. to set up cropping beforehand."""
    key = self.getPlaybackKey()
    if key is None:
      return
    if key[0] == InputType.VIDEO:
      self.playbackInfo = VideoFileSource.readInfo(key[1])
    else:
      self.playbackInfo = ImageSequenceSource.readInfo(key[1], key[2])
    self.playbackInfoKey = key
    self.frameSize = (self.playbackInfo['width'], self.playbackInfo['height'])

  @property
  def playbackFrameIndex(self):
    return self.source.currentFrameIndex if isinstance(self.source, PlaybackSource) else 0

  @property
  def currentImagePath(self):
    return self.source.currentFilePath if isinstance(self.source, ImageSequenceSource) else None

  def seek(self, frameIndex):
    """Shows the given frame of the video or image sequence. Opens the input (paused) if it is not open yet."""
    self._openPlaybackSource()
    self.source.seek(frameIndex)

  def _openPlaybackSource(self):
    key = self.getPlaybackKey()
    if key is None:
      raise ValueError("Select a video file first." if self.getParameterNode().inputType == InputType.VIDEO
                       else "Select an image directory first.")
    if isinstance(self.source, PlaybackSource) and self.source.key == key:
      return
    self.stop()
    self.source = VideoFileSource(key[1]) if key[0] == InputType.VIDEO else ImageSequenceSource(key[1], key[2])
    self.playbackInfo = self.source.info
    self.playbackInfoKey = key
    self._startUpdating()

  # -- Playback
  @property
  def isPlaying(self):
    return self.source is not None and self.source.isPlaying

  def start(self):
    if self.isPlaying:
      return
    parameterNode = self.getParameterNode()
    if parameterNode.inputType == InputType.WEBCAM:
      self.stop()
      self.source = WebcamSource(parameterNode.webcamDeviceIndex, self.getWebcamBackend())
      self._startUpdating()
    else:
      self._openPlaybackSource()
      self.source.play()
    self._notifyStatus()

  def pause(self):
    """Pauses playback. Videos and image sequences keep their position; cameras are released so other
    applications can use them."""
    if isinstance(self.source, PlaybackSource):
      self.source.pause()
      self.measuredFrameRate = None
      self._notifyStatus()
    else:
      self.stop()

  def stop(self, errorMessage=None):
    """Stops playback and releases the camera or video file."""
    self.updateTimer.stop()
    if self.source:
      self.source.stop()
      self.source = None
    self.measuredFrameRate = None
    self._notifyStatus(errorMessage)

  def clearInput(self):
    """Stops playback and forgets the input frame size, e.g. when a different input is selected."""
    self.stop()
    self.frameSize = None
    self.outputFrameSize = None
    self.playbackInfo = None
    self.playbackInfoKey = None
    self._notifyStatus()

  def _startUpdating(self):
    """Makes sure there is an output volume and starts copying frames from the source into it."""
    parameterNode = self.getParameterNode()
    if not parameterNode.outputVolume:
      outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode", slicer.mrmlScene.GenerateUniqueName("VideoInput"))
      outputVolume.CreateDefaultDisplayNodes()
      parameterNode.outputVolume = outputVolume
    self._configuredVolume = None
    self._configuredFrameShape = None
    self.measuredFrameRate = None
    self._lastReportedPlaying = None
    self.updateTimer.start()

  def onUpdateTimer(self):
    if not self.source:
      return
    if self.source.failed:
      self.stop(self.source.failureMessage)
      return

    parameterNode = self.getParameterNode()
    outputVolume = parameterNode.outputVolume
    if not outputVolume:
      self.stop()
      return

    # Playback speed, frame rate, and looping can be changed while playing
    if isinstance(self.source, VideoFileSource):
      self.source.playbackSpeed = parameterNode.videoPlaybackSpeed
      self.source.loop = parameterNode.videoLoop
    elif isinstance(self.source, ImageSequenceSource):
      self.source.frameRate = parameterNode.imagesFrameRate
      self.source.loop = parameterNode.imagesLoop

    frame = self.source.getLatestFrame()
    if frame is not None:
      self.frameSize = (frame.shape[1], frame.shape[0])
      outputFrame = self.processFrame(frame, parameterNode, outputVolume)
      self.outputFrameSize = (outputFrame.shape[1], outputFrame.shape[0])
      self.writeFrameToVolume(outputFrame, outputVolume)

    # Report immediately when playback stops by itself (end of the input) or a frame is shown while paused
    # (seeking), otherwise periodically
    playingChanged = self._lastReportedPlaying is not None and self._lastReportedPlaying != self.isPlaying
    statusDue = time.perf_counter() - self._lastStatusTime > self.STATUS_INTERVAL_S
    if playingChanged or frame is not None and (statusDue or not self.isPlaying):
      self.measuredFrameRate = self.source.getMeasuredFrameRate() if self.isPlaying else None
      self._notifyStatus()
    self._lastReportedPlaying = self.isPlaying

  def processFrame(self, frame, parameterNode, outputVolume):
    """Converts a BGR camera frame into the array that is written to the output volume."""
    import cv2
    # The output volume type determines the color: vector volumes receive RGB, scalar volumes receive grayscale
    if outputVolume.IsA("vtkMRMLVectorVolumeNode"):
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if parameterNode.flipHorizontal and parameterNode.flipVertical:
      frame = cv2.flip(frame, -1)
    elif parameterNode.flipHorizontal:
      frame = cv2.flip(frame, 1)
    elif parameterNode.flipVertical:
      frame = cv2.flip(frame, 0)

    # Crop (after flipping so the crop ranges match what is displayed)
    left, right, top, bottom = self.getCropBounds(frame.shape[1], frame.shape[0], parameterNode)
    if (left, right, top, bottom) != (0, frame.shape[1], 0, frame.shape[0]):
      frame = frame[top:bottom, left:right]

    # Rotation and zoom about the image center, then position, in a single warp. The output size stays the
    # same, so changing them does not reallocate the volume; areas moved in from outside the image are black.
    rotation = parameterNode.rotation
    zoom = parameterNode.zoom
    positionHorizontal = parameterNode.positionHorizontal
    positionVertical = parameterNode.positionVertical
    if rotation != 0.0 or zoom != 1.0 or positionHorizontal != 0 or positionVertical != 0:
      height, width = frame.shape[:2]
      center = ((width - 1) / 2.0, (height - 1) / 2.0)
      # OpenCV angles are counter-clockwise for a top-down image, so negate for clockwise-positive
      transform = cv2.getRotationMatrix2D(center, -rotation, zoom)
      # Position is applied in output pixels so it moves the image as displayed (image rows increase downward)
      transform[0, 2] += positionHorizontal
      transform[1, 2] -= positionVertical
      frame = cv2.warpAffine(frame, transform, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return frame

  @staticmethod
  def getCropBounds(width, height, parameterNode):
    """Converts the crop margins into pixel bounds (left, right, top, bottom), keeping at least 1 pixel.
    Margins larger than the frame (e.g. after switching to a lower resolution input) are clamped."""
    def toBounds(startMargin, endMargin, size):
      start = min(max(startMargin, 0), size - 1)
      end = min(max(size - endMargin, start + 1), size)
      return start, end
    left, right = toBounds(parameterNode.cropLeft, parameterNode.cropRight, width)
    top, bottom = toBounds(parameterNode.cropTop, parameterNode.cropBottom, height)
    return left, right, top, bottom

  def writeFrameToVolume(self, frame, outputVolume):
    volumeArray = frame[np.newaxis, ...]
    if outputVolume is self._configuredVolume and volumeArray.shape == self._configuredFrameShape:
      # Fast path: copy into the existing image buffer
      slicer.util.arrayFromVolume(outputVolume)[:] = volumeArray
      slicer.util.arrayFromVolumeModified(outputVolume)
      return

    # First frame, new volume, or frame size changed: (re)allocate the image and show it
    slicer.util.updateVolumeFromArray(outputVolume, volumeArray)
    # Row 0 of the image is the top; orient IJK so the frame appears upright in the Red (axial) slice view
    outputVolume.SetIJKToRASDirections(-1, 0, 0, 0, -1, 0, 0, 0, 1)
    outputVolume.SetOrigin(0, 0, 0)
    outputVolume.SetSpacing(1, 1, 1)
    slicer.util.setSliceViewerLayers(background=outputVolume, fit=True)
    self._configuredVolume = outputVolume
    self._configuredFrameShape = volumeArray.shape

  def _notifyStatus(self, errorMessage=None):
    self._lastStatusTime = time.perf_counter()
    if self.statusCallback:
      self.statusCallback(errorMessage)

# ------------------------------------ Tests ---------------------------------------
class VideoInputTest(ScriptedLoadableModuleTest):
  """Tests for the Video Input module. Run them with the Reload and Test button (developer mode) or through CTest.
  Video and image files are generated by the tests, with each frame filled with a known gray value, so the tests
  can check exactly which frame reached the output volume."""

  def setUp(self):
    slicer.mrmlScene.Clear()
    self.logic = VideoInputLogic()
    if not self.logic.areDependenciesInstalled():
      self.assertTrue(self.logic.installDependencies(askConfirmation=False), "Failed to install Python packages")
    # Reset the parameters, which keep their values when the scene is cleared
    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      for name, value in GLOBAL_PARAMETER_DEFAULTS.items():
        setattr(parameterNode, name, value)
      parameterNode.outputVolume = None
      parameterNode.videoPlaybackSpeed = 1.0
      parameterNode.videoLoop = True
      parameterNode.imagesFileFilter = '*.png *.jpg *.jpeg *.bmp *.tif *.tiff'
      parameterNode.imagesFrameRate = 30
      parameterNode.imagesLoop = True

  def runTest(self):
    tests = [
      self.test_ProcessFrameColor,
      self.test_ProcessFrameTransforms,
      self.test_CropBounds,
      self.test_ImageFileOrder,
      self.test_ImageSequencePlayback,
      self.test_VideoFilePlayback,
      self.test_Webcam,
    ]
    for test in tests:
      self.setUp()
      try:
        test()
      finally:
        # Removes temporary files (runTest is called directly, so unittest does not do this)
        self.doCleanups()
    self.delayDisplay("All Video Input tests passed")

  # -- Helpers
  def createTemporaryDirectory(self):
    temporaryDirectory = tempfile.TemporaryDirectory(dir=slicer.app.temporaryPath, ignore_cleanup_errors=True)
    self.addCleanup(temporaryDirectory.cleanup)
    return temporaryDirectory.name

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

  def outputMean(self):
    """Mean value of the output volume, or None until a frame has been written to it."""
    outputVolume = self.logic.getParameterNode().outputVolume
    if outputVolume is None or outputVolume.GetImageData() is None:
      return None
    return float(slicer.util.arrayFromVolume(outputVolume).mean())

  def assertOutputValue(self, expectedValue, tolerance=0):
    """Waits until the output volume shows a frame filled with the expected gray value."""
    def reached():
      mean = self.outputMean()
      return mean is not None and abs(mean - expectedValue) <= tolerance
    self.assertTrue(self.waitFor(reached), f"Expected output value {expectedValue}, got {self.outputMean()}")

  @staticmethod
  def grayFrame(width, height, value):
    return np.full((height, width, 3), value, dtype=np.uint8)

  # -- Frame processing (generated arrays)
  def test_ProcessFrameColor(self):
    """Checks that the output volume type selects the color mode.

    OpenCV delivers frames in BGR channel order. A frame filled with one known BGR color is processed for a vector
    volume and for a scalar volume:
    - Vector volume: the frame keeps its 3 channels, reordered to RGB, so (B, G, R) = (10, 20, 30) becomes (30, 20, 10).
    - Scalar volume: the frame becomes a single-channel grayscale image, matching OpenCV's own BGR-to-gray conversion.
    """
    self.delayDisplay("Testing color conversion")
    import cv2
    parameterNode = self.logic.getParameterNode()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # OpenCV frames are BGR

    vectorVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode")
    output = self.logic.processFrame(frame, parameterNode, vectorVolume)
    self.assertEqual(output.shape, (4, 6, 3))
    self.assertEqual(list(output[0, 0]), [30, 20, 10])

    scalarVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    output = self.logic.processFrame(frame, parameterNode, scalarVolume)
    self.assertEqual(output.shape, (4, 6))
    self.assertEqual(output[0, 0], cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[0, 0])

  def test_ProcessFrameTransforms(self):
    """Checks the global parameters (flip, crop, rotation, zoom, and position) pixel by pixel.

    A small 7 x 7 grayscale frame with a different value in every pixel is processed with one parameter changed at a
    time, and the result is compared with the same operation done in NumPy:
    - Defaults leave the frame unchanged.
    - Horizontal, vertical, and combined flips mirror the frame.
    - Crop margins remove the given number of pixels from each edge. Cropping happens after flipping, so the margins
      apply to the image as displayed (cropping the left after a horizontal flip removes the original right side).
    - Rotation by +90 degrees turns the frame clockwise and -90 degrees counter-clockwise. A small tolerance allows
      for interpolation rounding.
    - Zoom keeps the frame size and the center pixel, and the pixel next to the center is halfway between two input
      pixels at 2x zoom.
    - Positive horizontal and vertical positions move the image right and up, and the uncovered area is black.
    """
    self.delayDisplay("Testing flip, crop, rotation, zoom, and position")
    parameterNode = self.logic.getParameterNode()
    scalarVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    # Square image with an odd size so the rotation center is a pixel center, and a distinct value in every pixel.
    # Equal color channels make the grayscale conversion exact.
    image = (np.arange(7 * 7).reshape(7, 7) * 5).astype(np.uint8)
    frame = np.dstack([image] * 3)

    def process(**parameters):
      with slicer.util.NodeModify(parameterNode):
        for name, value in GLOBAL_PARAMETER_DEFAULTS.items():
          setattr(parameterNode, name, parameters.get(name, value))
      return self.logic.processFrame(frame, parameterNode, scalarVolume).astype(int)

    np.testing.assert_array_equal(process(), image)
    np.testing.assert_array_equal(process(flipHorizontal=True), np.fliplr(image))
    np.testing.assert_array_equal(process(flipVertical=True), np.flipud(image))
    np.testing.assert_array_equal(process(flipHorizontal=True, flipVertical=True), np.flipud(np.fliplr(image)))

    # Crop margins, applied after flipping so they match what is displayed
    np.testing.assert_array_equal(process(cropLeft=1, cropRight=2, cropTop=3, cropBottom=0), image[3:, 1:5])
    np.testing.assert_array_equal(process(flipHorizontal=True, cropLeft=2), np.fliplr(image)[:, 2:])

    # Positive rotation is clockwise
    np.testing.assert_allclose(process(rotation=90.0), np.rot90(image, k=-1), atol=1)
    np.testing.assert_allclose(process(rotation=-90.0), np.rot90(image, k=1), atol=1)

    # Zoom magnifies about the center and keeps the frame size
    zoomed = process(zoom=2.0)
    self.assertEqual(zoomed.shape, image.shape)
    self.assertEqual(zoomed[3, 3], image[3, 3])
    self.assertAlmostEqual(zoomed[3, 4], (int(image[3, 3]) + int(image[3, 4])) / 2, delta=1)

    # Positive positions move the image right and up; uncovered areas are black
    moved = process(positionHorizontal=2)
    np.testing.assert_array_equal(moved[:, 2:], image[:, :-2])
    np.testing.assert_array_equal(moved[:, :2], 0)
    moved = process(positionVertical=1)
    np.testing.assert_array_equal(moved[:-1, :], image[1:, :])
    np.testing.assert_array_equal(moved[-1, :], 0)

  def test_CropBounds(self):
    """Checks how crop margins are converted into pixel bounds (left, right, top, bottom).

    Margins are stored in pixels, so they can be larger than the frame, e.g. after cropping a 1920 x 1080 video and
    then switching to a 640 x 480 camera. Normal margins give the expected bounds, and margins larger than the frame,
    or top and bottom margins that overlap, are clamped so at least one pixel row and column are kept instead of
    producing an empty frame.
    """
    self.delayDisplay("Testing crop bounds")
    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      parameterNode.cropLeft, parameterNode.cropRight = 2, 3
      parameterNode.cropTop, parameterNode.cropBottom = 0, 0
    self.assertEqual(VideoInputLogic.getCropBounds(10, 8, parameterNode), (2, 7, 0, 8))
    with slicer.util.NodeModify(parameterNode):
      parameterNode.cropLeft, parameterNode.cropRight = 100, 100
      parameterNode.cropTop, parameterNode.cropBottom = 5, 5
    self.assertEqual(VideoInputLogic.getCropBounds(10, 8, parameterNode), (9, 10, 5, 6))

  # -- Image sequences (generated files)
  def test_ImageFileOrder(self):
    """Checks which files in a directory are used as an image sequence, and in what order.

    Empty files with a mix of names are created in a temporary directory, then listed with different file filters:
    - Numbers in file names are sorted by value, so frame_2 comes before frame_10 (a plain text sort would put
      frame_10 first). Files do not need zero-padded numbers.
    - Patterns match regardless of case on every operating system, so frame_1.PNG matches *.png.
    - Several space-separated patterns can be combined, and their matches are sorted together.
    - Files that match no pattern (notes.txt) and folders (even one named like an image) are ignored.
    - A filter that matches nothing gives an empty list, and reading the sequence information then gives an error
      instead of failing later during playback.
    """
    self.delayDisplay("Testing image file matching and order")
    directoryPath = self.createTemporaryDirectory()
    for fileName in ['frame_10.png', 'frame_2.png', 'frame_1.PNG', 'frame_3.jpg', 'notes.txt']:
      open(os.path.join(directoryPath, fileName), 'wb').close()
    os.mkdir(os.path.join(directoryPath, 'subfolder.png'))

    def fileNames(fileFilter):
      return [os.path.basename(path) for path in ImageSequenceSource.findImageFiles(directoryPath, fileFilter)]
    self.assertEqual(fileNames('*.png'), ['frame_1.PNG', 'frame_2.png', 'frame_10.png'])
    self.assertEqual(fileNames('*.png *.jpg'), ['frame_1.PNG', 'frame_2.png', 'frame_3.jpg', 'frame_10.png'])
    self.assertEqual(fileNames('*.bmp'), [])
    with self.assertRaises(RuntimeError):
      ImageSequenceSource.readInfo(directoryPath, '*.bmp')

  def test_ImageSequencePlayback(self):
    """Checks playing an image sequence into a grayscale (scalar) output volume.

    Twelve 40 x 30 PNG images are generated, where image i is filled with the gray value i * 20, so the value in the
    output volume shows exactly which image is displayed. Image 7 is replaced by a corrupt file. The test checks that:
    - The image count and size are read when the directory is selected, before playing, without opening the sequence.
    - Seeking while stopped opens the sequence paused and writes the selected image to the output volume, and the
      current file name is reported.
    - Without looping, playback stops by itself after the last image and leaves it displayed. The corrupt image is
      skipped instead of stopping playback.
    - Seeking to the corrupt image shows the next readable image.
    - With looping, playback continues from the first image after the last one.
    - Clearing the input (as when another input type or directory is selected) releases the sequence and forgets
      its information.
    """
    self.delayDisplay("Testing image sequence playback")
    import cv2
    directoryPath = self.createTemporaryDirectory()
    # Image i is filled with the value i * 20. Image 7 is corrupt and must be skipped.
    imageCount = 12
    for i in range(imageCount):
      filePath = os.path.join(directoryPath, f'frame_{i}.png')
      if i == 7:
        with open(filePath, 'wb') as file:
          file.write(b'not an image')
      else:
        cv2.imwrite(filePath, self.grayFrame(40, 30, i * 20))

    parameterNode = self.logic.getParameterNode()
    scalarVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ImagesOutput")
    with slicer.util.NodeModify(parameterNode):
      parameterNode.inputType = InputType.IMAGES
      parameterNode.imagesDirectoryPath = directoryPath
      parameterNode.imagesFileFilter = '*.png'
      parameterNode.outputVolume = scalarVolume
    try:
      # Information is available before playing
      self.logic.loadPlaybackInfo()
      self.assertEqual(self.logic.playbackInfo['frameCount'], imageCount)
      self.assertEqual(self.logic.frameSize, (40, 30))
      self.assertIsNone(self.logic.source)

      # Seeking opens the sequence paused and shows that image
      self.logic.seek(5)
      self.assertOutputValue(100)
      self.assertFalse(self.logic.isPlaying)
      self.assertEqual(self.logic.playbackFrameIndex, 5)
      self.assertEqual(os.path.basename(self.logic.currentImagePath), 'frame_5.png')
      self.assertEqual(slicer.util.arrayFromVolume(scalarVolume).shape, (1, 30, 40))

      # Without looping, playback stops at the last image, skipping the corrupt one
      with slicer.util.NodeModify(parameterNode):
        parameterNode.imagesLoop = False
        parameterNode.imagesFrameRate = 100
      self.logic.start()
      self.assertTrue(self.logic.isPlaying)
      self.assertTrue(self.waitFor(lambda: not self.logic.isPlaying), "Playback did not stop at the end")
      self.assertTrue(self.logic.source.reachedEnd)
      self.assertEqual(self.logic.playbackFrameIndex, imageCount - 1)
      self.assertOutputValue((imageCount - 1) * 20)

      # Seeking to the corrupt image shows the next readable one
      self.logic.seek(7)
      self.assertOutputValue(160)

      # With looping, playback wraps around to the start
      parameterNode.imagesLoop = True
      self.logic.seek(imageCount - 2)
      self.logic.start()
      self.assertTrue(self.waitFor(lambda: self.logic.playbackFrameIndex < 5), "Playback did not loop")
      self.assertTrue(self.logic.isPlaying)

      # Selecting a different input releases the sequence
      self.logic.clearInput()
      self.assertIsNone(self.logic.source)
      self.assertIsNone(self.logic.playbackInfo)
    finally:
      self.logic.stop()

  # -- Video files (generated files)
  def test_VideoFilePlayback(self):
    """Checks playing a video file into an output volume that is created automatically.

    A 30-frame, 64 x 48, 30 fps MJPG video is generated, where frame i is filled with the gray value i * 8, so the
    value in the output volume shows exactly which frame is displayed (within a small tolerance for JPEG compression).
    The test checks that:
    - The size, frame count, and frame rate are read when the file is selected, before playing, without opening it.
    - Seeking while stopped opens the video paused, creates an RGB (vector) output volume because none is selected,
      and writes the selected frame to it.
    - Playback advances, and pausing keeps the video open at the same frame instead of releasing it.
    - Without looping, playback stops by itself on the last frame, and pressing Play again restarts from the
      beginning.
    - With looping, playback continues from the first frame after the last one.
    """
    self.delayDisplay("Testing video file playback")
    import cv2
    # Frame i is filled with the value i * 8. MJPG in AVI can be written by every OpenCV build, and every
    # frame is a key frame, so seeking is exact.
    frameCount, frameRate, width, height = 30, 30.0, 64, 48
    filePath = os.path.join(self.createTemporaryDirectory(), 'test.avi')
    writer = cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'MJPG'), frameRate, (width, height))
    self.assertTrue(writer.isOpened(), "Unable to write the test video")
    for i in range(frameCount):
      writer.write(self.grayFrame(width, height, i * 8))
    writer.release()

    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      parameterNode.inputType = InputType.VIDEO
      parameterNode.videoFilePath = filePath
    # JPEG compression changes uniform values slightly
    tolerance = 3
    try:
      # Information is available before playing
      self.logic.loadPlaybackInfo()
      info = self.logic.playbackInfo
      self.assertEqual((info['width'], info['height'], info['frameCount']), (width, height, frameCount))
      self.assertAlmostEqual(info['frameRate'], frameRate, places=1)
      self.assertIsNone(self.logic.source)

      # Seeking opens the video paused, creates a vector output volume, and shows that frame
      self.logic.seek(20)
      self.assertOutputValue(160, tolerance)
      self.assertFalse(self.logic.isPlaying)
      outputVolume = parameterNode.outputVolume
      self.assertTrue(outputVolume.IsA("vtkMRMLVectorVolumeNode"))
      self.assertEqual(slicer.util.arrayFromVolume(outputVolume).shape, (1, height, width, 3))

      # Pausing keeps the video open at the same frame
      parameterNode.videoPlaybackSpeed = 0.5
      self.logic.seek(0)
      # Seeking happens on the playback thread, so wait for it before checking that playback advances
      self.assertOutputValue(0, tolerance)
      self.logic.start()
      self.assertTrue(self.waitFor(lambda: self.logic.playbackFrameIndex >= 2), "Playback did not advance")
      self.logic.pause()
      pausedFrameIndex = self.logic.playbackFrameIndex
      self.waitFor(lambda: False, timeoutS=0.3)
      self.assertFalse(self.logic.isPlaying)
      self.assertIsNotNone(self.logic.source)
      self.assertEqual(self.logic.playbackFrameIndex, pausedFrameIndex)

      # Without looping, playback stops at the last frame; playing again restarts from the beginning
      with slicer.util.NodeModify(parameterNode):
        parameterNode.videoLoop = False
        parameterNode.videoPlaybackSpeed = 4.0
      self.logic.seek(frameCount - 5)
      self.logic.start()
      self.assertTrue(self.waitFor(lambda: not self.logic.isPlaying), "Playback did not stop at the end")
      self.assertEqual(self.logic.playbackFrameIndex, frameCount - 1)
      self.assertOutputValue((frameCount - 1) * 8, tolerance)
      self.logic.start()
      self.assertTrue(self.waitFor(lambda: self.logic.playbackFrameIndex < 10), "Playback did not restart")
      self.logic.pause()

      # With looping, playback wraps around to the start
      parameterNode.videoLoop = True
      self.logic.seek(frameCount - 2)
      self.logic.start()
      self.assertTrue(self.waitFor(lambda: self.logic.playbackFrameIndex < 10), "Playback did not loop")
      self.assertTrue(self.logic.isPlaying)
    finally:
      self.logic.stop()

  # -- Webcams (test machines usually have no camera, so no camera is opened)
  def test_Webcam(self):
    """Checks camera detection and the error for a camera that cannot be opened.

    Test machines, such as the Slicer extension build machines, usually have no camera, so this test does not open
    or stream from a real camera. It checks that:
    - Listing cameras works whether or not cameras are connected, and returns (device index, device name) pairs.
    - Opening a device index that does not exist raises a RuntimeError with a readable message, which the module
      shows to the user, instead of failing silently or hanging.
    """
    self.delayDisplay("Testing webcam detection")
    cameras = self.logic.getAvailableWebcams()
    self.assertIsInstance(cameras, list)
    for deviceIndex, deviceName in cameras:
      self.assertIsInstance(deviceIndex, int)
      self.assertIsInstance(deviceName, str)
    with self.assertRaises(RuntimeError):
      WebcamSource(1000, self.logic.getWebcamBackend())
