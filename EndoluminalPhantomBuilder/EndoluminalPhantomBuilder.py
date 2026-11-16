##=========================================================================

#  Program:   Endoluminal Phantom Builder - Slicer Module

#  Copyright (c) Brigham and Women's Hospital. All rights reserved.

#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.

#=========================================================================
import os
import math
import logging
import numpy as np
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer import vtkMRMLSegmentationNode, vtkMRMLModelNode, vtkMRMLMarkupsFiducialNode
from vtk.util import numpy_support

class EndoluminalPhantomBuilder(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Endoluminal Phantom Builder"
    self.parent.categories = ["Endoscopy"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King (Brigham and Women's Hospital)"]
    self.parent.helpText = """
    Helps create a physical phantom of an endoluminal structure (such as lung airways) starting from
    a segmentation of the lumen. Can be used to create a 3D-printed rigid phantom or to create a mold to cast a flexible phantom from"""
    self.parent.acknowledgementText = """This module was developed by Franklin King (Brigham and Women's Hospital)."""
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

# ------------------------------------ Parameter Node ---------------------------------------
@parameterNodeWrapper
class EndoluminalPhantomBuilderParameterNode:
  # Inputs
  lumenSegmentation: vtkMRMLSegmentationNode | None
  lumenSegmentId: str = ''
  centerlineModel: vtkMRMLModelNode | None
  centerlineEndpoints: vtkMRMLMarkupsFiducialNode | None

  # Segmentation that receives the wall. If none is selected, "[lumen segmentation name]_wall" is created.
  wallSegmentation: vtkMRMLSegmentationNode | None
  # Model that the wall is exported to. If there is none, "[wall segmentation name]_model" is created.
  wallModel: vtkMRMLModelNode | None
  wallThickness: float = 4.0
  openingExtraDiameter: float = 3.0
  cutTracheaFlat: bool = True
  tracheaCutExtraDiameter: float = 10.0
  # Smoothing factor of the surface made from the wall (0 to 1), which is also the surface of the exported model
  surfaceSmoothingFactor: float = 0.5

  # Mold
  # Segmentation that receives the mold. If none is selected, "[lumen segmentation name]_mold" is created.
  moldSegmentation: vtkMRMLSegmentationNode | None
  moldThickness: float = 1.5
  # Cut the end of the mold at the trachea flat across the centerline, like the wall
  cutMoldTracheaFlat: bool = True
  moldTracheaCutExtraDiameter: float = 10.0
  # Smoothing factor of the surface made from the mold (0 to 1). It is low, as more smoothing makes the surface of a
  # mold only a voxel or two thick collapse.
  moldSurfaceSmoothingFactor: float = 0.1
  # Model that the mold is exported to. If there is none, "[mold segmentation name]_model" is created.
  moldModel: vtkMRMLModelNode | None
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

class EndoluminalPhantomBuilderWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  ERROR_COLOR = "#E53935"
  # The main action of a step, like the start buttons of the other modules. Muted while disabled.
  STEP_BUTTON_STYLE = ("QPushButton {background-color: #1565C0; color: white;} "
                       "QPushButton:hover {background-color: #1976D2;} "
                       "QPushButton:disabled {background-color: #3B4F66; color: #9AA5B1;}")
  EXPORT_BUTTON_STYLE = ("QPushButton {background-color: #9C5218; color: white;} "
                         "QPushButton:hover {background-color: #AD5E21;} "
                         "QPushButton:disabled {background-color: #5C4535; color: #A89A8F;}")
  MODEL_DIRECTORY_SETTING = "EndoluminalPhantomBuilder/ModelDirectory"

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)
    self.logic = None
    self._parameterNode = None
    self._parameterNodeGuiTag = None
    self._observedInputNodes = []
    self._inputsComplete = None
    self._wallCreated = None

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = EndoluminalPhantomBuilderLogic()

    stepFont = qt.QFont()
    stepFont.setPointSize(12)

    # ------------------------------------ Inputs UI ---------------------------------------
    self.inputsCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputsCollapsibleButton.text = "1. Initial Segmentation"
    self.inputsCollapsibleButton.collapsed = False
    self.layout.addWidget(self.inputsCollapsibleButton)
    inputsLayout = qt.QFormLayout(self.inputsCollapsibleButton)

    self.inputsNoteLabel = WrappingLabel(
      "Select a segmentation of the lumen, a model of its centerline, and the endpoints of the centerline. "
      "<a href=\"https://github.com/vmtk/SlicerExtension-VMTK\">SlicerVMTK</a> may be used to do this.")
    self.inputsNoteLabel.setOpenExternalLinks(True)
    inputsLayout.addRow(self.inputsNoteLabel)

    self.lumenSegmentSelector = slicer.qMRMLSegmentSelectorWidget()
    self.lumenSegmentSelector.noneEnabled = True
    self.lumenSegmentSelector.setMRMLScene(slicer.mrmlScene)
    self.lumenSegmentSelector.setToolTip("Segmentation and segment of the lumen (the open space inside, e.g. the airways)")
    self.lumenSegmentSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onLumenSegmentationChanged)
    self.lumenSegmentSelector.connect('currentSegmentChanged(QString)', self.onLumenSegmentChanged)
    inputsLayout.addRow("Lumen Segmentation:", self.lumenSegmentSelector)

    self.centerlineModelSelector = slicer.qMRMLNodeComboBox()
    self.centerlineModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.centerlineModelSelector.showChildNodeTypes = False
    self.centerlineModelSelector.selectNodeUponCreation = True
    self.centerlineModelSelector.addEnabled = False
    self.centerlineModelSelector.removeEnabled = False
    self.centerlineModelSelector.noneEnabled = True
    self.centerlineModelSelector.showHidden = False
    self.centerlineModelSelector.setMRMLScene(slicer.mrmlScene)
    self.centerlineModelSelector.setToolTip("Model of the centerline of the lumen, made of lines (a model, not a markups curve)")
    inputsLayout.addRow("Centerline Model:", self.centerlineModelSelector)

    self.centerlineEndpointsSelector = slicer.qMRMLNodeComboBox()
    self.centerlineEndpointsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.centerlineEndpointsSelector.selectNodeUponCreation = True
    self.centerlineEndpointsSelector.addEnabled = False
    self.centerlineEndpointsSelector.removeEnabled = False
    self.centerlineEndpointsSelector.noneEnabled = True
    self.centerlineEndpointsSelector.showHidden = False
    self.centerlineEndpointsSelector.setMRMLScene(slicer.mrmlScene)
    self.centerlineEndpointsSelector.setToolTip("Point list with the endpoints of the centerline: the opening of the lumen "
                                                "and the end of each branch")
    inputsLayout.addRow("Centerline Endpoints:", self.centerlineEndpointsSelector)

    self.extractCenterlineButton = qt.QPushButton("Auto-generate Centerline and Endpoints (Requires SlicerVMTK)")
    self.extractCenterlineButton.connect('clicked()', self.onExtractCenterlineButtonClicked)
    inputsLayout.addRow("", self.extractCenterlineButton)

    self.inputsStatusLabel = WrappingLabel()
    inputsLayout.addRow("", self.inputsStatusLabel)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Wall UI ---------------------------------------
    self.wallCollapsibleButton = ctk.ctkCollapsibleButton()
    self.wallCollapsibleButton.text = "2. Wall"
    self.wallCollapsibleButton.collapsed = True
    self.layout.addWidget(self.wallCollapsibleButton)
    wallLayout = qt.QFormLayout(self.wallCollapsibleButton)

    self.wallNoteLabel = WrappingLabel(
      "Creates a wall of the set thickness around the lumen. <b>This is what the finished phantom will look like.</b><br>"
      "A sphere is removed around each endpoint in order to create an opening. The original segment is left unchanged."
      "If you are only creating a rigid model, then you can export the model at this stage and 3D print it as-is."
      "If you are creating a mold for a flexible phantom, then move on to the next step.")
    wallLayout.addRow(self.wallNoteLabel)

    self.wallThicknessSpinBox = ctk.ctkDoubleSpinBox()
    self.wallThicknessSpinBox.setToolTip("Thickness of the wall around the lumen")
    self.wallThicknessSpinBox.decimals = 2
    self.wallThicknessSpinBox.minimum = 0.1
    self.wallThicknessSpinBox.maximum = 50.0
    self.wallThicknessSpinBox.singleStep = 0.5
    self.wallThicknessSpinBox.suffix = " mm"
    wallLayout.addRow("Wall Thickness:", self.wallThicknessSpinBox)

    self.wallSegmentationSelector = slicer.qMRMLNodeComboBox()
    self.wallSegmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.wallSegmentationSelector.selectNodeUponCreation = True
    self.wallSegmentationSelector.addEnabled = False
    self.wallSegmentationSelector.removeEnabled = True
    self.wallSegmentationSelector.renameEnabled = True
    self.wallSegmentationSelector.noneEnabled = True
    self.wallSegmentationSelector.noneDisplay = "Create new segmentation"
    self.wallSegmentationSelector.showHidden = False
    self.wallSegmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.wallSegmentationSelector.setToolTip("Segmentation that receives the wall. Its segments are replaced. If none is "
                                             "selected, \"[lumen segmentation name]_wall\" is created.")
    wallLayout.addRow("Wall Segmentation:", self.wallSegmentationSelector)

    self.openingExtraDiameterSpinBox = ctk.ctkDoubleSpinBox()
    self.openingExtraDiameterSpinBox.setToolTip(
      "The wall is opened at each centerline endpoint by erasing a sphere as wide as the lumen there plus the wall on "
      "each side, so the hole goes all the way through. This adds to the diameter of the sphere, to make the openings "
      "larger. The tube is trimmed back around each opening by about the radius of the sphere minus the lumen radius.")
    self.openingExtraDiameterSpinBox.decimals = 1
    self.openingExtraDiameterSpinBox.minimum = 0.0
    self.openingExtraDiameterSpinBox.maximum = 50.0
    self.openingExtraDiameterSpinBox.singleStep = 0.5
    self.openingExtraDiameterSpinBox.suffix = " mm"
    wallLayout.addRow("Opening Extra Diameter:", self.openingExtraDiameterSpinBox)

    self.cutTracheaFlatCheckBox = qt.QCheckBox("Cut trachea opening flat")
    self.cutTracheaFlatCheckBox.setToolTip(
      "Open the end with the widest lumen (the trachea) with a flat cut across the centerline instead of a sphere.")
    wallLayout.addRow(self.cutTracheaFlatCheckBox)

    self.tracheaCutExtraDiameterSpinBox = ctk.ctkDoubleSpinBox()
    self.tracheaCutExtraDiameterSpinBox.setToolTip(
      "Added to the diameter of the flat cut across the end of the trachea.")
    self.tracheaCutExtraDiameterSpinBox.decimals = 1
    self.tracheaCutExtraDiameterSpinBox.minimum = 0.0
    self.tracheaCutExtraDiameterSpinBox.maximum = 200.0
    self.tracheaCutExtraDiameterSpinBox.singleStep = 1.0
    self.tracheaCutExtraDiameterSpinBox.suffix = " mm"
    wallLayout.addRow("Trachea Cut Extra Diameter:", self.tracheaCutExtraDiameterSpinBox)

    self.surfaceSmoothingSliderWidget = ctk.ctkSliderWidget()
    self.surfaceSmoothingSliderWidget.setToolTip(
      "Smoothing of the 3D surface of the wall, which is also the surface of the exported model: 0 follows the voxels "
      "exactly, and higher values are smoother but can make thin parts thinner. Changing it updates the surface of an "
      "existing wall.")
    self.surfaceSmoothingSliderWidget.decimals = 2
    self.surfaceSmoothingSliderWidget.minimum = 0.0
    self.surfaceSmoothingSliderWidget.maximum = 1.0
    self.surfaceSmoothingSliderWidget.singleStep = 0.05
    # Updates when the slider is released, as making the surface can take a moment
    self.surfaceSmoothingSliderWidget.tracking = False
    wallLayout.addRow("Surface Smoothing:", self.surfaceSmoothingSliderWidget)

    self.createWallButton = qt.QPushButton("Create Wall")
    self.createWallButton.setFont(stepFont)
    self.createWallButton.setMinimumHeight(40)
    self.createWallButton.styleSheet = self.STEP_BUTTON_STYLE
    self.createWallButton.connect('clicked()', self.onCreateWallButtonClicked)
    wallLayout.addRow(self.createWallButton)

    self.editInSegmentEditorButton = qt.QPushButton("Edit in Segment Editor")
    self.editInSegmentEditorButton.connect('clicked()', self.onEditInSegmentEditorButtonClicked)
    wallLayout.addRow(self.editInSegmentEditorButton)

    self.exportModelButton = qt.QPushButton("Export and Save Model")
    # Smaller than the main action of the step, but larger than the other buttons
    exportFont = qt.QFont()
    exportFont.setPointSize(10)
    self.exportModelButton.setFont(exportFont)
    self.exportModelButton.setMinimumHeight(32)
    self.exportModelButton.styleSheet = self.EXPORT_BUTTON_STYLE
    self.exportModelButton.connect('clicked()', self.onExportModelButtonClicked)
    wallLayout.addRow(self.exportModelButton)
    # -------------------------------------- ----------  --------------------------------------

    # ------------------------------------ Mold UI ---------------------------------------
    self.moldCollapsibleButton = ctk.ctkCollapsibleButton()
    self.moldCollapsibleButton.text = "3. Mold"
    self.moldCollapsibleButton.collapsed = True
    self.layout.addWidget(self.moldCollapsibleButton)
    moldLayout = qt.QFormLayout(self.moldCollapsibleButton)

    self.moldNoteLabel = WrappingLabel(
      "Creates a mold around the wall for casting a flexible phantom: a shell of the set thickness around the "
      "outside of the wall. An internal section is suspended within the "
      "outer wall. The thickness of the wall should be kept thin enough to easily remove after casting. "
      "3D printing this mold will likely require a dissolvable support material such as PVA or SR-30.")
    moldLayout.addRow(self.moldNoteLabel)

    self.moldThicknessSpinBox = ctk.ctkDoubleSpinBox()
    self.moldThicknessSpinBox.setToolTip("Thickness of the mold around the wall")
    self.moldThicknessSpinBox.decimals = 2
    self.moldThicknessSpinBox.minimum = 0.1
    self.moldThicknessSpinBox.maximum = 50.0
    self.moldThicknessSpinBox.singleStep = 0.5
    self.moldThicknessSpinBox.suffix = " mm"
    moldLayout.addRow("Mold Thickness:", self.moldThicknessSpinBox)

    self.moldSegmentationSelector = slicer.qMRMLNodeComboBox()
    self.moldSegmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.moldSegmentationSelector.selectNodeUponCreation = True
    self.moldSegmentationSelector.addEnabled = False
    self.moldSegmentationSelector.removeEnabled = True
    self.moldSegmentationSelector.renameEnabled = True
    self.moldSegmentationSelector.noneEnabled = True
    self.moldSegmentationSelector.noneDisplay = "Create new segmentation"
    self.moldSegmentationSelector.showHidden = False
    self.moldSegmentationSelector.setMRMLScene(slicer.mrmlScene)
    self.moldSegmentationSelector.setToolTip("Segmentation that receives the mold. Its segments are replaced. If none is "
                                             "selected, \"[lumen segmentation name]_mold\" is created.")
    moldLayout.addRow("Mold Segmentation:", self.moldSegmentationSelector)

    self.cutMoldTracheaFlatCheckBox = qt.QCheckBox("Cut trachea opening flat")
    self.cutMoldTracheaFlatCheckBox.setToolTip(
      "Open the end of the mold at the trachea (the end with the widest lumen) with a flat cut across the centerline.")
    moldLayout.addRow(self.cutMoldTracheaFlatCheckBox)

    self.moldTracheaCutExtraDiameterSpinBox = ctk.ctkDoubleSpinBox()
    self.moldTracheaCutExtraDiameterSpinBox.setToolTip(
      "Added to the diameter of the flat cut across the end of the trachea, which is otherwise the diameter of the lumen "
      "plus the wall and the mold on each side.")
    self.moldTracheaCutExtraDiameterSpinBox.decimals = 1
    self.moldTracheaCutExtraDiameterSpinBox.minimum = 0.0
    self.moldTracheaCutExtraDiameterSpinBox.maximum = 200.0
    self.moldTracheaCutExtraDiameterSpinBox.singleStep = 1.0
    self.moldTracheaCutExtraDiameterSpinBox.suffix = " mm"
    moldLayout.addRow("Trachea Cut Extra Diameter:", self.moldTracheaCutExtraDiameterSpinBox)

    self.moldSurfaceSmoothingSliderWidget = ctk.ctkSliderWidget()
    self.moldSurfaceSmoothingSliderWidget.setToolTip(
      "Smoothing of the 3D surface of the mold: 0 follows the voxels exactly, and higher values are smoother. Keep it "
      "low for a thin mold (a voxel or two thick), whose surface more smoothing makes collapse. Changing it updates the "
      "surface of an existing mold.")
    self.moldSurfaceSmoothingSliderWidget.decimals = 2
    self.moldSurfaceSmoothingSliderWidget.minimum = 0.0
    self.moldSurfaceSmoothingSliderWidget.maximum = 1.0
    self.moldSurfaceSmoothingSliderWidget.singleStep = 0.05
    # Updates when the slider is released, as making the surface can take a moment
    self.moldSurfaceSmoothingSliderWidget.tracking = False
    moldLayout.addRow("Surface Smoothing:", self.moldSurfaceSmoothingSliderWidget)

    self.createMoldButton = qt.QPushButton("Create Mold")
    self.createMoldButton.setFont(stepFont)
    self.createMoldButton.setMinimumHeight(40)
    self.createMoldButton.styleSheet = self.STEP_BUTTON_STYLE
    self.createMoldButton.connect('clicked()', self.onCreateMoldButtonClicked)
    moldLayout.addRow(self.createMoldButton)

    self.editMoldInSegmentEditorButton = qt.QPushButton("Edit in Segment Editor")
    self.editMoldInSegmentEditorButton.connect('clicked()', self.onEditMoldInSegmentEditorButtonClicked)
    moldLayout.addRow(self.editMoldInSegmentEditorButton)

    self.exportMoldModelButton = qt.QPushButton("Export and Save Model")
    self.exportMoldModelButton.setFont(exportFont)
    self.exportMoldModelButton.setMinimumHeight(32)
    self.exportMoldModelButton.styleSheet = self.EXPORT_BUTTON_STYLE
    self.exportMoldModelButton.connect('clicked()', self.onExportMoldModelButtonClicked)
    moldLayout.addRow(self.exportMoldModelButton)
    # -------------------------------------- ----------  --------------------------------------

    self.layout.addStretch(1)

    # Widgets bound directly to parameter node values
    self.parameterWidgets = {
      'centerlineModel': self.centerlineModelSelector,
      'centerlineEndpoints': self.centerlineEndpointsSelector,
      'wallSegmentation': self.wallSegmentationSelector,
      'cutTracheaFlat': self.cutTracheaFlatCheckBox,
      'moldSegmentation': self.moldSegmentationSelector,
      'cutMoldTracheaFlat': self.cutMoldTracheaFlatCheckBox,
    }
    # Widgets with ranges set above are synced manually, since binding them would replace their ranges
    self.valueWidgets = {
      'wallThickness': self.wallThicknessSpinBox,
      'openingExtraDiameter': self.openingExtraDiameterSpinBox,
      'tracheaCutExtraDiameter': self.tracheaCutExtraDiameterSpinBox,
      'surfaceSmoothingFactor': self.surfaceSmoothingSliderWidget,
      'moldThickness': self.moldThicknessSpinBox,
      'moldTracheaCutExtraDiameter': self.moldTracheaCutExtraDiameterSpinBox,
      'moldSurfaceSmoothingFactor': self.moldSurfaceSmoothingSliderWidget,
    }
    for name, widget in self.valueWidgets.items():
      widget.connect('valueChanged(double)', lambda value, name=name: self.onValueWidgetChanged(name, value))
    # Connected after the parameter is set from the slider
    self.surfaceSmoothingSliderWidget.connect('valueChanged(double)', self.onSurfaceSmoothingChanged)
    self.moldSurfaceSmoothingSliderWidget.connect('valueChanged(double)', self.onSurfaceSmoothingChanged)

    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneEndImport)

    self.initializeParameterNode()

  def cleanup(self):
    self.removeObservers()

  def enter(self):
    self.initializeParameterNode()

  def exit(self):
    self.setParameterNode(None)

  def onSceneStartClose(self, caller, event):
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
    self._inputsComplete = None
    self._wallCreated = None
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    if not self._parameterNode:
      self.observeInputNodes([])
      return

    if self.lumenSegmentSelector.currentNode() != self._parameterNode.lumenSegmentation:
      wasBlocked = self.lumenSegmentSelector.blockSignals(True)
      self.lumenSegmentSelector.setCurrentNode(self._parameterNode.lumenSegmentation)
      self.lumenSegmentSelector.blockSignals(wasBlocked)
    if self._parameterNode.lumenSegmentation and self.lumenSegmentSelector.currentSegmentID() != self._parameterNode.lumenSegmentId:
      wasBlocked = self.lumenSegmentSelector.blockSignals(True)
      self.lumenSegmentSelector.setCurrentSegmentID(self._parameterNode.lumenSegmentId)
      self.lumenSegmentSelector.blockSignals(wasBlocked)

    for name, widget in self.valueWidgets.items():
      value = getattr(self._parameterNode, name)
      if widget.value != value:
        wasBlocked = widget.blockSignals(True)
        widget.value = value
        widget.blockSignals(wasBlocked)
    self.tracheaCutExtraDiameterSpinBox.enabled = self._parameterNode.cutTracheaFlat
    self.moldTracheaCutExtraDiameterSpinBox.enabled = self._parameterNode.cutMoldTracheaFlat

    self.observeInputNodes([self._parameterNode.lumenSegmentation, self._parameterNode.centerlineModel,
                            self._parameterNode.centerlineEndpoints, self._parameterNode.wallSegmentation,
                            self._parameterNode.moldSegmentation])
    self.updateSteps()

  def onValueWidgetChanged(self, name, value):
    if self._parameterNode:
      setattr(self._parameterNode, name, type(getattr(self._parameterNode, name))(value))

  # ------------------------------------ Inputs ---------------------------------------
  def onLumenSegmentationChanged(self, node):
    if self._parameterNode:
      with slicer.util.NodeModify(self._parameterNode):
        self._parameterNode.lumenSegmentation = node
        self._parameterNode.lumenSegmentId = self.lumenSegmentSelector.currentSegmentID() or ''

  def onLumenSegmentChanged(self, segmentId):
    if self._parameterNode:
      self._parameterNode.lumenSegmentId = segmentId or ''

  def observeInputNodes(self, nodes):
    """Observes the input nodes so the steps follow changes in them, e.g. points added to the endpoints."""
    # Each node once, as the same node can be selected for more than one input (which is reported as an error)
    nodes = list({node.GetID(): node for node in nodes if node}.values())
    if nodes == self._observedInputNodes:
      return
    for node in self._observedInputNodes:
      for event in self.getInputNodeEvents(node):
        self.removeObserver(node, event, self.onInputNodeModified)
    self._observedInputNodes = nodes
    for node in nodes:
      for event in self.getInputNodeEvents(node):
        self.addObserver(node, event, self.onInputNodeModified)

  @staticmethod
  def getInputNodeEvents(node):
    """Returns the events sent when the content of the node changes. Not all changes send a ModifiedEvent, e.g. removing
    a point from a point list does not."""
    if node.IsA("vtkMRMLSegmentationNode"):
      return [slicer.vtkSegmentation.SegmentAdded, slicer.vtkSegmentation.SegmentRemoved, slicer.vtkSegmentation.SegmentModified]
    if node.IsA("vtkMRMLModelNode"):
      return [slicer.vtkMRMLModelNode.MeshModifiedEvent]
    if node.IsA("vtkMRMLMarkupsNode"):
      return [slicer.vtkMRMLMarkupsNode.PointAddedEvent, slicer.vtkMRMLMarkupsNode.PointRemovedEvent]
    return [vtk.vtkCommand.ModifiedEvent]

  def onInputNodeModified(self, caller=None, event=None):
    self.updateSteps()

  def onExtractCenterlineButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to extract the centerline.", waitCursor=True):
      self.logic.extractCenterline()

  # ------------------------------------ Steps ---------------------------------------
  def updateSteps(self):
    """Updates the status of each step. Like a wizard, each step opens once the steps before it are done, but earlier
    steps stay open so they can be revisited."""
    inputErrors = self.logic.getInputErrors()
    if inputErrors:
      inputsText = "<br>".join(f'<span style="color: {self.ERROR_COLOR};">{error}</span>' for error in inputErrors)
    else:
      inputsText = "Inputs are ready."
    if self.inputsStatusLabel.text != inputsText:
      self.inputsStatusLabel.setText(inputsText)

    if not self.logic.isExtractCenterlineAvailable():
      self.extractCenterlineButton.enabled = False
      self.extractCenterlineButton.setToolTip("Needs the Extract Centerline module of the SlicerVMTK extension. Install "
                                              "SlicerVMTK from the Extensions Manager and restart Slicer to use it.")
    else:
      lumenError = self.logic.getLumenSegmentError()
      self.extractCenterlineButton.enabled = not lumenError
      self.extractCenterlineButton.setToolTip(
        lumenError or "Make the centerline model and endpoints from the lumen segment with the Extract Centerline module "
        "of SlicerVMTK. The endpoints are detected automatically (like its Auto-detect button), with the widest one "
        "(the trachea) as the start point. Nodes made by an earlier extraction are replaced, others are kept.")

    inputsComplete = not inputErrors
    if inputsComplete and self._inputsComplete is False:
      self.wallCollapsibleButton.collapsed = False
    self._inputsComplete = inputsComplete

    wallErrors = self.logic.getWallErrors()
    self.createWallButton.enabled = not wallErrors
    self.createWallButton.setToolTip(" ".join(wallErrors) or "Create the wall around the lumen. An existing wall is replaced.")

    createdWallError = self.logic.getCreatedWallError()
    self.editInSegmentEditorButton.enabled = not createdWallError
    self.editInSegmentEditorButton.setToolTip(
      createdWallError or "Open the wall in the Segment Editor module to fix it by hand, e.g. with Paint and Erase. The "
      "CT that the lumen segmentation was made from is selected as the source volume, if it is known.")
    self.exportModelButton.enabled = not createdWallError
    self.exportModelButton.setToolTip(createdWallError or "Export the wall to a model and save it to a file (e.g. STL for "
                                      "3D printing). A model exported earlier is replaced.")

    wallCreated = not createdWallError
    if wallCreated and self._wallCreated is False:
      self.moldCollapsibleButton.collapsed = False
    self._wallCreated = wallCreated

    createdMoldError = self.logic.getCreatedMoldError()
    self.editMoldInSegmentEditorButton.enabled = not createdMoldError
    self.editMoldInSegmentEditorButton.setToolTip(
      createdMoldError or "Open the mold in the Segment Editor module to fix it by hand, e.g. with Paint and Erase. The "
      "CT that the lumen segmentation was made from is selected as the source volume, if it is known.")
    self.exportMoldModelButton.enabled = not createdMoldError
    self.exportMoldModelButton.setToolTip(createdMoldError or "Export the mold to a model and save it to a file (e.g. STL "
                                          "for 3D printing). A model exported earlier is replaced.")

    moldErrors = self.logic.getMoldErrors()
    self.createMoldButton.enabled = not moldErrors
    self.createMoldButton.setToolTip(" ".join(moldErrors) or "Create the mold around the wall. An existing mold is replaced.")

  # ------------------------------------ Wall ---------------------------------------
  def onCreateWallButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to create the wall.", waitCursor=True):
      self.logic.createWall()

  def onSurfaceSmoothingChanged(self, value):
    with slicer.util.tryWithErrorDisplay("Failed to update the surfaces.", waitCursor=True):
      self.logic.updateSurfaces()

  def onEditInSegmentEditorButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to open the wall in the Segment Editor."):
      createdWallError = self.logic.getCreatedWallError()
      if createdWallError:
        raise ValueError(createdWallError)
      self.openInSegmentEditor(self._parameterNode.wallSegmentation, self.logic.getWallSegmentId())

  def openInSegmentEditor(self, segmentationNode, segmentId):
    """Switches to the Segment Editor module with the segment selected, to fix it by hand."""
    lumenSegmentationNode = self._parameterNode.lumenSegmentation
    # Paint and Erase need a source volume. The segment has its own segmentation, so it does not have one yet.
    sourceVolume = self.logic.getSegmentationVolume(lumenSegmentationNode) if lumenSegmentationNode else None
    slicer.util.selectModule("SegmentEditor")
    segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    if sourceVolume:
      segmentEditorWidget.mrmlSegmentEditorNode().SetAndObserveSourceVolumeNode(sourceVolume)
    segmentEditorWidget.setCurrentSegmentID(segmentId)

  # ------------------------------------ Mold ---------------------------------------
  def onCreateMoldButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to create the mold.", waitCursor=True):
      self.logic.createMold()

  def onEditMoldInSegmentEditorButtonClicked(self):
    with slicer.util.tryWithErrorDisplay("Failed to open the mold in the Segment Editor."):
      createdMoldError = self.logic.getCreatedMoldError()
      if createdMoldError:
        raise ValueError(createdMoldError)
      self.openInSegmentEditor(self._parameterNode.moldSegmentation, self.logic.getMoldSegmentId())

  def onExportMoldModelButtonClicked(self):
    self.exportAndSaveModel(self.logic.exportMoldModel)

  # ------------------------------------ Export ---------------------------------------
  def onExportModelButtonClicked(self):
    self.exportAndSaveModel(self.logic.exportWallModel)

  def exportAndSaveModel(self, exportModel):
    """Exports a model with the export function, then asks where to save it and saves it there. If saving is
    cancelled, the model stays in the scene."""
    model = None
    with slicer.util.tryWithErrorDisplay("Failed to export the model.", waitCursor=True):
      model = exportModel()
    if not model:
      return
    # The file dialog starts in the folder that a model was last saved to
    settings = qt.QSettings()
    directory = settings.value(self.MODEL_DIRECTORY_SETTING) or slicer.app.defaultScenePath
    filePath = qt.QFileDialog.getSaveFileName(slicer.util.mainWindow(), "Save Model",
                                              os.path.join(directory, model.GetName() + ".stl"),
                                              "STL files (*.stl);;OBJ files (*.obj);;PLY files (*.ply)")
    if not filePath:
      return  # The model stays in the scene
    with slicer.util.tryWithErrorDisplay("Failed to save the model.", waitCursor=True):
      self.logic.saveModel(model, filePath)
      settings.setValue(self.MODEL_DIRECTORY_SETTING, os.path.dirname(filePath))

class EndoluminalPhantomBuilderLogic(ScriptedLoadableModuleLogic):
  WALL_COLOR = (0.89, 0.60, 0.55)
  MOLD_COLOR = (0.55, 0.70, 0.90)
  # The lumen is measured within this distance (mm) of each endpoint
  OPENING_SEARCH_DISTANCE = 50.0
  # The direction of the trachea at its endpoint is from the centerline this far (mm) into the trachea
  TRACHEA_DIRECTION_DISTANCE = 20.0
  # Marks the centerline model and endpoints made by extractCenterline, which extracting again replaces
  EXTRACTED_CENTERLINE_ATTRIBUTE = "EndoluminalPhantomBuilder.ExtractedCenterline"

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self._parameterNode = None
    self.lastOpenings = []

  def getParameterNode(self):
    parameterNode = super().getParameterNode()
    if not self._parameterNode or self._parameterNode.parameterNode != parameterNode:
      self._parameterNode = EndoluminalPhantomBuilderParameterNode(parameterNode)
    return self._parameterNode

  def refreshParameterNode(self):
    """Makes the parameter node wrapper read the parameters again, e.g. after a scene is loaded. The wrapper keeps the
    selected nodes and reads them again when the parameter node is modified, but when a scene is loaded, the parameter
    node is modified before the nodes it refers to are loaded, so they would be read as None."""
    self.getParameterNode().parameterNode.Modified()

  # -- Inputs
  def getLumenSegment(self):
    parameterNode = self.getParameterNode()
    segmentationNode = parameterNode.lumenSegmentation
    if not segmentationNode or not parameterNode.lumenSegmentId:
      return None
    return segmentationNode.GetSegmentation().GetSegment(parameterNode.lumenSegmentId)

  def getLumenSegmentError(self):
    """Returns why no lumen segment is selected, or None if one is."""
    if not self.getParameterNode().lumenSegmentation:
      return "Select the lumen segmentation."
    if not self.getLumenSegment():
      return "Select the lumen segment."
    return None

  def getLumenError(self):
    """Returns why the lumen segment cannot be used for the wall, or None if it can."""
    parameterNode = self.getParameterNode()
    lumenSegmentError = self.getLumenSegmentError()
    if lumenSegmentError:
      return lumenSegmentError
    if parameterNode.lumenSegmentation is parameterNode.wallSegmentation:
      return "The wall segmentation must be different from the lumen segmentation."
    return None

  def getInputErrors(self):
    """Returns a list of the problems with the inputs, which is empty when all inputs are ready."""
    lumenError = self.getLumenError()
    return ([lumenError] if lumenError else []) + self.getCenterlineErrors()

  def getWallErrors(self):
    """Returns a list of the reasons the wall cannot be created, which is empty when it can. It needs the lumen, the
    centerline endpoints to open the ends, and the centerline model to cut the trachea flat."""
    parameterNode = self.getParameterNode()
    lumenError = self.getLumenError()
    errors = [lumenError] if lumenError else []
    errors += self.getEndpointsErrors()
    if parameterNode.cutTracheaFlat:
      errors += self.getTracheaCutCenterlineErrors()
    return errors

  def getEndpointsErrors(self):
    centerlineEndpoints = self.getParameterNode().centerlineEndpoints
    if not centerlineEndpoints:
      return ["Select the centerline endpoints."]
    if centerlineEndpoints.GetNumberOfControlPoints() == 0:
      return ["The centerline endpoints have no points."]
    return []

  def getTracheaCutCenterlineErrors(self):
    centerlineModel = self.getParameterNode().centerlineModel
    if not centerlineModel:
      return ["Select the centerline model, which is needed to cut the trachea flat."]
    if not centerlineModel.GetPolyData() or centerlineModel.GetPolyData().GetNumberOfLines() == 0:
      return ["The centerline model has no lines."]
    return []

  def getCenterlineErrors(self):
    parameterNode = self.getParameterNode()
    errors = []
    centerlineModel = parameterNode.centerlineModel
    if not centerlineModel:
      errors.append("Select the centerline model.")
    elif not centerlineModel.GetPolyData() or centerlineModel.GetPolyData().GetNumberOfLines() == 0:
      errors.append("The centerline model has no lines.")
    centerlineEndpoints = parameterNode.centerlineEndpoints
    if not centerlineEndpoints:
      errors.append("Select the centerline endpoints.")
    elif centerlineEndpoints.GetNumberOfControlPoints() < 2:
      errors.append("The centerline endpoints need at least 2 points.")
    return errors

  @staticmethod
  def getSegmentationVolume(segmentationNode):
    """Returns the scalar volume that the segmentation was made from (the volume whose geometry it uses, which the
    Segment Editor sets), or None if there is none."""
    volume = segmentationNode.GetNodeReference(slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())
    if volume and volume.IsA("vtkMRMLScalarVolumeNode") and not volume.IsA("vtkMRMLLabelMapVolumeNode"):
      return volume
    return None

  # -- Extract Centerline (SlicerVMTK)
  @staticmethod
  def isExtractCenterlineAvailable():
    """Returns whether the Extract Centerline module of the SlicerVMTK extension is installed. It is only needed to make
    the centerline with the Extract Centerline button."""
    return hasattr(slicer.modules, "extractcenterline")

  def getOrCreateExtractedNode(self, currentNode, className, name):
    """Returns the current node if an earlier extraction made it, so extracting again replaces it, or a new node."""
    if currentNode and currentNode.GetAttribute(self.EXTRACTED_CENTERLINE_ATTRIBUTE) == "true":
      return currentNode
    node = slicer.mrmlScene.AddNewNodeByClass(className, slicer.mrmlScene.GetUniqueNameByString(name))
    node.SetAttribute(self.EXTRACTED_CENTERLINE_ATTRIBUTE, "true")
    node.CreateDefaultDisplayNodes()
    return node

  def extractCenterline(self):
    """Makes the centerline model and endpoints from the lumen segment with the Extract Centerline module of SlicerVMTK,
    the same way as its module does with the default settings: the endpoints are detected automatically (like its
    Auto-detect button) and the centerline is extracted between them. The widest endpoint (the trachea) is the start
    point, and is first in the endpoints. Selects the results as the inputs, and returns the centerline model and
    endpoints."""
    if not self.isExtractCenterlineAvailable():
      raise RuntimeError("Extracting the centerline needs the Extract Centerline module of the SlicerVMTK extension.")
    lumenSegmentError = self.getLumenSegmentError()
    if lumenSegmentError:
      raise ValueError(lumenSegmentError)
    import ExtractCenterline
    extractCenterlineLogic = ExtractCenterline.ExtractCenterlineLogic()
    parameterNode = self.getParameterNode()
    lumenSegmentationNode = parameterNode.lumenSegmentation

    lumenSurface = vtk.vtkPolyData()
    lumenSegmentationNode.CreateClosedSurfaceRepresentation()
    lumenSegmentationNode.GetClosedSurfaceRepresentation(parameterNode.lumenSegmentId, lumenSurface)
    if lumenSurface.GetNumberOfPoints() == 0:
      raise ValueError("The lumen segment is empty.")
    # The default preprocessing settings of Extract Centerline
    preprocessedSurface = extractCenterlineLogic.preprocess(lumenSurface, targetNumberOfPoints=5000,
                                                            decimationAggressiveness=4.0, subdivide=False)

    baseName = self.getLumenSegment().GetName()
    centerlineEndpoints = self.getOrCreateExtractedNode(parameterNode.centerlineEndpoints, "vtkMRMLMarkupsFiducialNode",
                                                        baseName + " centerline endpoints")
    centerlineModel = self.getOrCreateExtractedNode(parameterNode.centerlineModel, "vtkMRMLModelNode",
                                                    baseName + " centerline")
    # The surface is in the coordinates of the segmentation
    for node in [centerlineEndpoints, centerlineModel]:
      node.SetAndObserveTransformNodeID(lumenSegmentationNode.GetTransformNodeID())

    centerlineEndpoints.RemoveAllControlPoints()
    networkPolyData = extractCenterlineLogic.extractNetwork(preprocessedSurface, centerlineEndpoints)
    endpointPositions = extractCenterlineLogic.getEndPoints(networkPolyData, startPointPosition=None)
    if len(endpointPositions) < 2:
      raise ValueError(f"Found {len(endpointPositions)} centerline endpoints, but at least 2 are needed.")
    with slicer.util.NodeModify(centerlineEndpoints):
      for position in endpointPositions:
        centerlineEndpoints.AddControlPoint(vtk.vtkVector3d(position))
      # Extract Centerline uses the first unselected point as the start point
      centerlineEndpoints.SetNthControlPointSelected(0, False)
    centerlineEndpoints.GetDisplayNode().PointLabelsVisibilityOff()

    centerlinePolyData, _ = extractCenterlineLogic.extractCenterline(preprocessedSurface, centerlineEndpoints,
                                                                     curveSamplingDistance=1.0)
    centerlineModel.SetAndObserveMesh(centerlinePolyData)
    centerlineModel.GetDisplayNode().SetColor(0.0, 1.0, 0.0)
    centerlineModel.GetDisplayNode().SetLineWidth(3)

    with slicer.util.NodeModify(parameterNode):
      parameterNode.centerlineModel = centerlineModel
      parameterNode.centerlineEndpoints = centerlineEndpoints
    return centerlineModel, centerlineEndpoints

  # -- Wall
  @staticmethod
  def getPaddedGeometry(labelmap, paddingMm):
    """Returns an image with the geometry (spacing, origin, directions) of the labelmap and an extent that covers its
    nonzero voxels plus the padding on each side, or None if the labelmap is empty.

    Segment Editor effects only change voxels within the reference geometry of the segmentation, which is often
    just big enough for the lumen (e.g. a segmentation imported from a model) or ends at the edge of the image. A wall
    grown outward would then be cut off where it reaches that boundary, so the geometry is padded by the wall
    thickness first."""
    effectiveExtent = [0, -1, 0, -1, 0, -1]
    slicer.vtkOrientedImageDataResample.CalculateEffectiveExtent(labelmap, effectiveExtent)
    if effectiveExtent[0] > effectiveExtent[1] or effectiveExtent[2] > effectiveExtent[3] or effectiveExtent[4] > effectiveExtent[5]:
      return None
    spacing = labelmap.GetSpacing()
    # A couple of extra voxels, so the outer surface of the wall is not on the boundary
    padding = [math.ceil(paddingMm / spacing[axis]) + 2 for axis in range(3)]
    paddedExtent = []
    for axis in range(3):
      paddedExtent += [effectiveExtent[2 * axis] - padding[axis], effectiveExtent[2 * axis + 1] + padding[axis]]
    geometry = slicer.vtkOrientedImageData()
    imageToWorld = vtk.vtkMatrix4x4()
    labelmap.GetImageToWorldMatrix(imageToWorld)
    geometry.SetGeometryFromImageToWorldMatrix(imageToWorld)
    geometry.SetExtent(paddedExtent)
    return geometry

  def getOrCreateWallSegmentation(self):
    parameterNode = self.getParameterNode()
    if not parameterNode.wallSegmentation:
      name = parameterNode.lumenSegmentation.GetName() + "_wall"
      wallSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
      wallSegmentationNode.CreateDefaultDisplayNodes()
      parameterNode.wallSegmentation = wallSegmentationNode
    return parameterNode.wallSegmentation

  def getWallSegmentId(self):
    """Returns the id of the wall segment (the first segment of the wall segmentation), or None if there is none."""
    wallSegmentationNode = self.getParameterNode().wallSegmentation
    if not wallSegmentationNode or wallSegmentationNode.GetSegmentation().GetNumberOfSegments() == 0:
      return None
    return wallSegmentationNode.GetSegmentation().GetNthSegmentID(0)

  def getLumenLabelmap(self, wallSegmentationNode):
    """Copies the lumen segment into the wall segmentation and returns it as a labelmap that contains only the lumen.
    The copy is converted to a labelmap if the lumen segmentation stores closed surfaces."""
    parameterNode = self.getParameterNode()
    lumenSegmentation = parameterNode.lumenSegmentation.GetSegmentation()
    binaryLabelmapName = slicer.vtkSegmentationConverter.GetBinaryLabelmapRepresentationName()
    wallSegmentation = wallSegmentationNode.GetSegmentation()
    wallSegmentation.RemoveAllSegments()
    wallSegmentation.CopyConversionParameters(lumenSegmentation)
    wallSegmentation.SetSourceRepresentationName(lumenSegmentation.GetSourceRepresentationName())
    if not wallSegmentation.CopySegmentFromSegmentation(lumenSegmentation, parameterNode.lumenSegmentId):
      raise RuntimeError("Could not copy the lumen segment.")
    if wallSegmentation.GetSourceRepresentationName() != binaryLabelmapName:
      if not wallSegmentation.CreateRepresentation(binaryLabelmapName):
        raise RuntimeError("Could not convert the lumen segment to a binary labelmap.")
      wallSegmentation.SetSourceRepresentationName(binaryLabelmapName)
    # Extracts the lumen alone, since a copied segment can share its labelmap with the other segments it was stored with
    lumenLabelmap = slicer.vtkOrientedImageData()
    wallSegmentationNode.GetBinaryLabelmapRepresentation(parameterNode.lumenSegmentId, lumenLabelmap)
    wallSegmentation.RemoveAllSegments()
    return lumenLabelmap

  @staticmethod
  def applySegmentEditorEffect(segmentationNode, segmentId, effectName, parameters):
    """Applies an effect of the Segment Editor, with the parameters (a dictionary of names and values), to the
    segment. Other segments are not changed."""
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    try:
      segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
      segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
      segmentEditorWidget.setSegmentationNode(segmentationNode)
      segmentEditorNode.SetSelectedSegmentID(segmentId)
      segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)
      segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
      segmentEditorWidget.setActiveEffectByName(effectName)
      effect = segmentEditorWidget.activeEffect()
      if not effect:
        raise RuntimeError(f"The {effectName} effect of the Segment Editor is not available.")
      for name, value in parameters.items():
        effect.setParameter(name, value)
      effect.self().onApply()
    finally:
      segmentEditorWidget.setActiveEffectByName(None)
      segmentEditorWidget.setMRMLSegmentEditorNode(None)
      slicer.mrmlScene.RemoveNode(segmentEditorNode)

  @classmethod
  def hollowSegment(cls, segmentationNode, segmentId, thicknessMm):
    """Runs the Hollow effect of the Segment Editor on the segment, with the segment as the inside surface, so the
    shell grows outward from it."""
    cls.applySegmentEditorEffect(segmentationNode, segmentId, "Hollow", {
      "ApplyToAllVisibleSegments": 0,
      "ShellMode": "INSIDE_SURFACE",
      "ShellThicknessMm": thicknessMm,
    })

  @classmethod
  def keepLargestIsland(cls, segmentationNode, segmentId):
    """Runs the Islands effect of the Segment Editor on the segment, keeping only its largest connected piece (e.g.
    removing small pieces of wall left around the openings)."""
    # A minimum size of 0 keeps the largest piece however small it is
    cls.applySegmentEditorEffect(segmentationNode, segmentId, "Islands", {
      "Operation": "KEEP_LARGEST_ISLAND",
      "MinimumSize": 0,
    })

  # -- Opening the ends
  def getEndpointPositions(self, segmentationNode):
    """Returns the centerline endpoints (N x 3) in the coordinates of the segmentation (which differ from world
    coordinates if the segmentation is transformed), and their labels."""
    endpoints = self.getParameterNode().centerlineEndpoints
    worldToSegmentation = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, segmentationNode.GetParentTransformNode(), worldToSegmentation)
    positions = []
    labels = []
    for pointIndex in range(endpoints.GetNumberOfControlPoints()):
      positions.append(worldToSegmentation.TransformPoint(endpoints.GetNthControlPointPositionWorld(pointIndex)))
      labels.append(endpoints.GetNthControlPointLabel(pointIndex) or f"point {pointIndex + 1}")
    return np.array(positions).reshape(-1, 3), labels

  @staticmethod
  def getLabelmapArray(labelmap):
    """Returns the voxels of the labelmap as an array (k, j, i) that shares memory with it."""
    extent = labelmap.GetExtent()
    shape = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
    return numpy_support.vtk_to_numpy(labelmap.GetPointData().GetScalars()).reshape(shape)

  @classmethod
  def getRegionAroundPoint(cls, labelmap, center, radius):
    """Returns the index ranges (minimum and maximum i, j, k, within the extent of the labelmap) of the voxels within
    the radius of the center, and the positions of those voxels (k, j, i, 3). Returns None if there are none."""
    extent = labelmap.GetExtent()
    imageToWorld = vtk.vtkMatrix4x4()
    labelmap.GetImageToWorldMatrix(imageToWorld)
    imageToWorldArray = slicer.util.arrayFromVTKMatrix(imageToWorld)
    worldToIjk = np.linalg.inv(imageToWorldArray)
    centerIjk = worldToIjk[:3, :3] @ center + worldToIjk[:3, 3]
    radiusIjk = radius / np.array(labelmap.GetSpacing())
    minimumIjk = np.maximum(np.floor(centerIjk - radiusIjk).astype(int), [extent[0], extent[2], extent[4]])
    maximumIjk = np.minimum(np.ceil(centerIjk + radiusIjk).astype(int), [extent[1], extent[3], extent[5]])
    if np.any(minimumIjk > maximumIjk):
      return None
    k, j, i = np.mgrid[minimumIjk[2]:maximumIjk[2] + 1, minimumIjk[1]:maximumIjk[1] + 1, minimumIjk[0]:maximumIjk[0] + 1]
    positions = np.stack([i, j, k], axis=-1).astype(float) @ imageToWorldArray[:3, :3].T + imageToWorldArray[:3, 3]
    return minimumIjk, maximumIjk, positions

  @classmethod
  def getRegionArray(cls, labelmap, minimumIjk, maximumIjk):
    """Returns the voxels of the labelmap within the index ranges, as an array (k, j, i) that shares memory with it."""
    extent = labelmap.GetExtent()
    return cls.getLabelmapArray(labelmap)[minimumIjk[2] - extent[4]:maximumIjk[2] - extent[4] + 1,
                                          minimumIjk[1] - extent[2]:maximumIjk[1] - extent[2] + 1,
                                          minimumIjk[0] - extent[0]:maximumIjk[0] - extent[0] + 1]

  @classmethod
  def measureLumenRadius(cls, lumenLabelmap, endpoint):
    """Returns the radius of the largest sphere that fits in the lumen and reaches the endpoint, which is the radius of
    the lumen at the endpoint, or None if no sphere in the lumen reaches it (the endpoint is not at the lumen). The
    spheres are centered on voxels, so the radius is accurate to about a voxel."""
    import SimpleITK as sitk
    region = cls.getRegionAroundPoint(lumenLabelmap, endpoint, cls.OPENING_SEARCH_DISTANCE)
    if not region:
      return None
    minimumIjk, maximumIjk, positions = region
    lumen = cls.getRegionArray(lumenLabelmap, minimumIjk, maximumIjk) > 0
    if not np.any(lumen):
      return None
    # Distance from each lumen voxel to the nearest voxel outside the lumen. The region is padded with voxels outside
    # the lumen, so a lumen cut off by the edge of the labelmap is closed there.
    outside = sitk.GetImageFromArray(np.pad(~lumen, 1, constant_values=True).astype(np.uint8))
    outside.SetSpacing(lumenLabelmap.GetSpacing())
    distanceToOutside = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
      outside, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))[1:-1, 1:-1, 1:-1]
    # The surface of the lumen is halfway between the voxels inside and outside it
    spacing = lumenLabelmap.GetSpacing()
    sphereRadii = distanceToOutside[lumen] - min(spacing) / 2
    # The endpoint can be on the surface of the lumen, so spheres up to a voxel short of it also reach it
    distanceToEndpoint = np.linalg.norm(positions[lumen] - endpoint, axis=1)
    reaching = distanceToEndpoint <= sphereRadii + max(spacing)
    if not np.any(reaching):
      return None
    return float(np.max(sphereRadii[reaching]))

  @classmethod
  def eraseSphere(cls, labelmap, center, radius):
    """Clears the voxels of the labelmap within the sphere. Returns the number of voxels cleared."""
    region = cls.getRegionAroundPoint(labelmap, center, radius)
    if not region:
      return 0
    minimumIjk, maximumIjk, positions = region
    array = cls.getRegionArray(labelmap, minimumIjk, maximumIjk)
    clear = (np.linalg.norm(positions - center, axis=-1) <= radius) & (array > 0)
    array[clear] = 0
    labelmap.Modified()
    return int(clear.sum())

  @classmethod
  def eraseCylinder(cls, labelmap, base, direction, radius, length):
    """Clears the voxels of the labelmap within the cylinder that starts at the base and extends along the direction
    (a unit vector) for the length. Returns the number of voxels cleared."""
    region = cls.getRegionAroundPoint(labelmap, base + direction * length / 2, math.hypot(radius, length / 2))
    if not region:
      return 0
    minimumIjk, maximumIjk, positions = region
    array = cls.getRegionArray(labelmap, minimumIjk, maximumIjk)
    relative = positions - base
    axial = relative @ direction
    radial = np.linalg.norm(relative - axial[..., np.newaxis] * direction, axis=-1)
    clear = (axial >= 0) & (axial <= length) & (radial <= radius) & (array > 0)
    array[clear] = 0
    labelmap.Modified()
    return int(clear.sum())

  def getCenterlineLines(self, segmentationNode):
    """Returns the lines of the centerline model as arrays of points (N x 3), in the coordinates of the segmentation."""
    centerlineModel = self.getParameterNode().centerlineModel
    polyData = centerlineModel.GetPolyData()
    modelToSegmentation = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(centerlineModel.GetParentTransformNode(),
                                                          segmentationNode.GetParentTransformNode(), modelToSegmentation)
    points = np.array([modelToSegmentation.TransformPoint(polyData.GetPoint(pointIndex))
                       for pointIndex in range(polyData.GetNumberOfPoints())])
    lines = []
    for cellIndex in range(polyData.GetNumberOfCells()):
      if polyData.GetCellType(cellIndex) not in [vtk.VTK_LINE, vtk.VTK_POLY_LINE]:
        continue
      pointIds = polyData.GetCell(cellIndex).GetPointIds()
      if pointIds.GetNumberOfIds() >= 2:
        lines.append(points[[pointIds.GetId(index) for index in range(pointIds.GetNumberOfIds())]])
    return lines

  @staticmethod
  def getCenterlineDirection(lines, endpoint, directionDistance):
    """Finds the centerline point closest to the endpoint and follows its line, away from the endpoint, for the
    direction distance (or to the end of the line). Returns the distance from the endpoint to the closest point, and the
    unit vector from the point reached to the closest point, which points out of the lumen at the endpoint (None if
    the line has no length there)."""
    closestDistance = None
    for line in lines:
      distances = np.linalg.norm(line - endpoint, axis=1)
      index = int(np.argmin(distances))
      if closestDistance is None or distances[index] < closestDistance:
        closestDistance = float(distances[index])
        closestLine = line
        closestIndex = index
    if closestDistance is None:
      return None, None
    # Follow the line toward its farther end, which is into the lumen when the endpoint is at an end of the line
    step = -1 if closestIndex > len(closestLine) - 1 - closestIndex else 1
    length = 0.0
    index = closestIndex
    while length < directionDistance and 0 <= index + step < len(closestLine):
      length += np.linalg.norm(closestLine[index + step] - closestLine[index])
      index += step
    direction = closestLine[closestIndex] - closestLine[index]
    directionLength = np.linalg.norm(direction)
    return closestDistance, (direction / directionLength if directionLength > 0 else None)

  def getTracheaDirection(self, segmentationNode, endpoint, lumenRadius):
    """Returns the direction out of the trachea at its endpoint (from the centerline), and None, or None and why the
    direction is not known."""
    distanceToCenterline, direction = self.getCenterlineDirection(
      self.getCenterlineLines(segmentationNode), endpoint, self.TRACHEA_DIRECTION_DISTANCE)
    if distanceToCenterline is None or distanceToCenterline > lumenRadius:
      return None, "it is not at the centerline"
    if direction is None:
      return None, "the centerline there has no length"
    return direction, None

  def openWallEnds(self, wallSegmentationNode, wallSegmentId, lumenLabelmap):
    """Opens the wall at each centerline endpoint by erasing a sphere centered on the endpoint. The radius of the
    sphere is the radius of the lumen there plus the wall thickness, so the hole goes through the wall over the whole
    width of the lumen, plus half the extra opening diameter.
    If the trachea is cut flat, the endpoint with the widest lumen (the trachea) is instead opened by erasing a cylinder
    from the endpoint outward along the centerline, so the end of the tube is cut flat across the centerline instead of
    trimmed back. The radius of the cylinder is the radius of the lumen plus the wall thickness, plus half of both the
    opening extra diameter and the trachea cut extra diameter.
    Returns a list with the result for each endpoint: its label, the radius of the lumen there or why it was not opened,
    and whether it is the trachea."""
    parameterNode = self.getParameterNode()
    extraRadius = parameterNode.openingExtraDiameter / 2
    endpoints, labels = self.getEndpointPositions(wallSegmentationNode)
    wallLabelmap = slicer.vtkOrientedImageData()
    wallSegmentationNode.GetBinaryLabelmapRepresentation(wallSegmentId, wallLabelmap)
    openings = []
    for endpoint, label in zip(endpoints, labels):
      radius = self.measureLumenRadius(lumenLabelmap, endpoint)
      openings.append({'label': label, 'radius': radius, 'error': None if radius is not None else "it is not at the lumen",
                       'trachea': False})

    measuredIndices = [index for index, opening in enumerate(openings) if opening['radius'] is not None]
    tracheaIndex = None
    if parameterNode.cutTracheaFlat and measuredIndices:
      tracheaIndex = max(measuredIndices, key=lambda index: openings[index]['radius'])
      openings[tracheaIndex]['trachea'] = True

    for index in measuredIndices:
      opening = openings[index]
      endpoint = endpoints[index]
      cutRadius = opening['radius'] + parameterNode.wallThickness + extraRadius
      if index != tracheaIndex:
        self.eraseSphere(wallLabelmap, endpoint, cutRadius)
        continue
      direction, opening['error'] = self.getTracheaDirection(wallSegmentationNode, endpoint, opening['radius'])
      if opening['error']:
        opening['radius'] = None
        continue
      # Long enough to go through the wall even if the endpoint is up to a lumen radius inside the end of the lumen
      self.eraseCylinder(wallLabelmap, endpoint, direction, cutRadius + parameterNode.tracheaCutExtraDiameter / 2,
                         cutRadius + max(wallLabelmap.GetSpacing()))
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(wallLabelmap, wallSegmentationNode, wallSegmentId)
    return openings

  def createHollowWall(self):
    """Creates the wall segment as a copy of the lumen segment, hollowed outward by the wall thickness. This closes
    the ends of the lumen, which createWall then opens. Returns the wall segmentation node, the id of the wall segment,
    and the lumen labelmap."""
    parameterNode = self.getParameterNode()
    lumenError = self.getLumenError()
    if lumenError:
      raise ValueError(lumenError)
    lumenSegmentationNode = parameterNode.lumenSegmentation
    lumenSegment = self.getLumenSegment()
    thickness = parameterNode.wallThickness

    wallSegmentationNode = self.getOrCreateWallSegmentation()
    wallSegmentationNode.SetAndObserveTransformNodeID(lumenSegmentationNode.GetTransformNodeID())
    wallSegmentation = wallSegmentationNode.GetSegmentation()
    lumenLabelmap = self.getLumenLabelmap(wallSegmentationNode)
    geometry = self.getPaddedGeometry(lumenLabelmap, thickness)
    if not geometry:
      raise ValueError("The lumen segment is empty.")
    wallSegmentation.SetConversionParameter(slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName(),
                                            slicer.vtkSegmentationConverter.SerializeImageGeometry(geometry))

    wallSegmentId = wallSegmentation.AddEmptySegment("", lumenSegment.GetName() + "_wall", self.WALL_COLOR)
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(lumenLabelmap, wallSegmentationNode, wallSegmentId)
    self.hollowSegment(wallSegmentationNode, wallSegmentId, thickness)
    return wallSegmentationNode, wallSegmentId, lumenLabelmap

  def createWall(self):
    """Creates the wall segment: a copy of the lumen segment, hollowed outward by the wall thickness, opened at the
    centerline endpoints, and with only its largest piece kept. Returns the wall segmentation node and the id of the
    wall segment. The result of opening each end is in lastOpenings."""
    wallErrors = self.getWallErrors()
    if wallErrors:
      raise ValueError(" ".join(wallErrors))
    wallSegmentationNode, wallSegmentId, lumenLabelmap = self.createHollowWall()
    self.lastOpenings = self.openWallEnds(wallSegmentationNode, wallSegmentId, lumenLabelmap)
    self.keepLargestIsland(wallSegmentationNode, wallSegmentId)

    self.updateSurface(wallSegmentationNode, self.getParameterNode().surfaceSmoothingFactor)
    displayNode = wallSegmentationNode.GetDisplayNode()
    if displayNode:
      displayNode.SetSegmentVisibility(wallSegmentId, True)
    return wallSegmentationNode, wallSegmentId

  def updateSurfaces(self):
    """Makes the surfaces of the wall and the mold with their surface smoothing factors."""
    parameterNode = self.getParameterNode()
    if parameterNode.wallSegmentation:
      self.updateSurface(parameterNode.wallSegmentation, parameterNode.surfaceSmoothingFactor)
    if parameterNode.moldSegmentation:
      self.updateSurface(parameterNode.moldSegmentation, parameterNode.moldSurfaceSmoothingFactor)

  @staticmethod
  def updateSurface(segmentationNode, smoothingFactor):
    """Makes the surface (closed surface representation) of the segmentation with the smoothing factor, if it is not
    made with it already."""
    segmentation = segmentationNode.GetSegmentation()
    smoothingFactorName = slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName()
    smoothingFactor = str(smoothingFactor)
    closedSurfaceName = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
    if (segmentation.GetConversionParameter(smoothingFactorName) == smoothingFactor
        and segmentation.ContainsRepresentation(closedSurfaceName)):
      return
    segmentation.SetConversionParameter(smoothingFactorName, smoothingFactor)
    # The surface is made again with the new smoothing
    segmentationNode.RemoveClosedSurfaceRepresentation()
    segmentationNode.CreateClosedSurfaceRepresentation()

  def getCreatedWallError(self):
    """Returns why there is no wall to edit or export, or None if there is one."""
    if not self.getWallSegmentId():
      return "Create the wall first."
    return None

  # -- Mold
  def getMoldErrors(self):
    """Returns a list of the reasons the mold cannot be created, which is empty when it can."""
    parameterNode = self.getParameterNode()
    lumenSegmentError = self.getLumenSegmentError()
    errors = [lumenSegmentError] if lumenSegmentError else []
    createdWallError = self.getCreatedWallError()
    if createdWallError:
      errors.append(createdWallError)
    moldSegmentationNode = parameterNode.moldSegmentation
    if moldSegmentationNode and moldSegmentationNode in [parameterNode.lumenSegmentation, parameterNode.wallSegmentation]:
      errors.append("The mold segmentation must be different from the lumen and wall segmentations.")
    if parameterNode.cutMoldTracheaFlat:
      errors += self.getEndpointsErrors() + self.getTracheaCutCenterlineErrors()
    return errors

  def getMoldSegmentId(self):
    """Returns the id of the mold segment (the first segment of the mold segmentation), or None if there is none."""
    moldSegmentationNode = self.getParameterNode().moldSegmentation
    if not moldSegmentationNode or moldSegmentationNode.GetSegmentation().GetNumberOfSegments() == 0:
      return None
    return moldSegmentationNode.GetSegmentation().GetNthSegmentID(0)

  def getCreatedMoldError(self):
    """Returns why there is no mold to edit or export, or None if there is one."""
    if not self.getMoldSegmentId():
      return "Create the mold first."
    return None

  def getOrCreateMoldSegmentation(self):
    parameterNode = self.getParameterNode()
    if not parameterNode.moldSegmentation:
      name = parameterNode.lumenSegmentation.GetName() + "_mold"
      moldSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
      moldSegmentationNode.CreateDefaultDisplayNodes()
      parameterNode.moldSegmentation = moldSegmentationNode
    return parameterNode.moldSegmentation

  def cutMoldTrachea(self, moldSegmentationNode, moldSegmentId, lumenLabelmap):
    """Cuts the end of the mold at the trachea (the endpoint with the widest lumen) flat across the centerline, with a
    cylinder from the endpoint outward. Its radius is the radius of the lumen plus the wall and mold thicknesses, plus
    half the trachea cut extra diameter of the mold. Returns None, or why the end was not cut."""
    parameterNode = self.getParameterNode()
    endpoints, _ = self.getEndpointPositions(moldSegmentationNode)
    radii = [self.measureLumenRadius(lumenLabelmap, endpoint) for endpoint in endpoints]
    measuredIndices = [index for index, radius in enumerate(radii) if radius is not None]
    if not measuredIndices:
      return "no centerline endpoint is at the lumen"
    tracheaIndex = max(measuredIndices, key=lambda index: radii[index])
    endpoint = endpoints[tracheaIndex]
    direction, error = self.getTracheaDirection(moldSegmentationNode, endpoint, radii[tracheaIndex])
    if error:
      return f"its endpoint {error}"
    moldLabelmap = slicer.vtkOrientedImageData()
    moldSegmentationNode.GetBinaryLabelmapRepresentation(moldSegmentId, moldLabelmap)
    cutRadius = radii[tracheaIndex] + parameterNode.wallThickness + parameterNode.moldThickness
    # Long enough to go through the mold even if the endpoint is up to a lumen radius inside the end of the lumen
    self.eraseCylinder(moldLabelmap, endpoint, direction, cutRadius + parameterNode.moldTracheaCutExtraDiameter / 2,
                       cutRadius + max(moldLabelmap.GetSpacing()))
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(moldLabelmap, moldSegmentationNode, moldSegmentId)
    return None

  def createMold(self):
    """Creates the mold segment: a copy of the wall segment, hollowed outward by the mold thickness, so the mold
    encloses the wall (around its outside, along the lumen, and over its ends), with the trachea end cut flat if
    enabled. Returns the mold segmentation node and the id of the mold segment. If the trachea end could not be cut,
    the mold is created and an error is raised."""
    moldErrors = self.getMoldErrors()
    if moldErrors:
      raise ValueError(" ".join(moldErrors))
    parameterNode = self.getParameterNode()
    lumenSegmentationNode = parameterNode.lumenSegmentation
    wallSegmentationNode = parameterNode.wallSegmentation
    thickness = parameterNode.moldThickness

    moldSegmentationNode = self.getOrCreateMoldSegmentation()
    moldSegmentationNode.SetAndObserveTransformNodeID(lumenSegmentationNode.GetTransformNodeID())
    moldSegmentation = moldSegmentationNode.GetSegmentation()
    # The lumen is used to find the trachea
    lumenLabelmap = self.getLumenLabelmap(moldSegmentationNode)
    wallLabelmap = slicer.vtkOrientedImageData()
    wallSegmentationNode.GetBinaryLabelmapRepresentation(self.getWallSegmentId(), wallLabelmap)

    geometry = self.getPaddedGeometry(wallLabelmap, thickness)
    if not geometry:
      raise ValueError("The wall segment is empty.")
    moldSegmentation.SetConversionParameter(slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName(),
                                            slicer.vtkSegmentationConverter.SerializeImageGeometry(geometry))

    moldSegmentId = moldSegmentation.AddEmptySegment("", self.getLumenSegment().GetName() + "_mold", self.MOLD_COLOR)
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(wallLabelmap, moldSegmentationNode, moldSegmentId)
    self.hollowSegment(moldSegmentationNode, moldSegmentId, thickness)
    tracheaCutError = self.cutMoldTrachea(moldSegmentationNode, moldSegmentId, lumenLabelmap) if parameterNode.cutMoldTracheaFlat else None

    self.updateSurface(moldSegmentationNode, parameterNode.moldSurfaceSmoothingFactor)
    displayNode = moldSegmentationNode.GetDisplayNode()
    if displayNode:
      displayNode.SetSegmentVisibility(moldSegmentId, True)
    if tracheaCutError:
      raise ValueError(f"The mold was created, but its trachea end was not cut: {tracheaCutError}.")
    return moldSegmentationNode, moldSegmentId

  # -- Export
  def exportWallModel(self):
    """Exports the wall segment to the wall model, which is created if there is none. Returns the model."""
    createdWallError = self.getCreatedWallError()
    if createdWallError:
      raise ValueError(createdWallError)
    return self.exportSegmentToModel(self.getParameterNode().wallSegmentation, self.getWallSegmentId(), "wallModel")

  def exportMoldModel(self):
    """Exports the mold segment to the mold model, which is created if there is none. Returns the model."""
    createdMoldError = self.getCreatedMoldError()
    if createdMoldError:
      raise ValueError(createdMoldError)
    return self.exportSegmentToModel(self.getParameterNode().moldSegmentation, self.getMoldSegmentId(), "moldModel")

  def exportSegmentToModel(self, segmentationNode, segmentId, modelParameterName):
    """Exports the segment (its surface) to the model in the parameter of that name, which is created as
    "[segmentation name]_model" if there is none. Returns the model."""
    parameterNode = self.getParameterNode()
    segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
    segmentationNode.CreateClosedSurfaceRepresentation()
    model = getattr(parameterNode, modelParameterName)
    if not model:
      model = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLModelNode", slicer.mrmlScene.GetUniqueNameByString(segmentationNode.GetName() + "_model"))
      model.CreateDefaultDisplayNodes()
      setattr(parameterNode, modelParameterName, model)
    # Exporting names the model after the segment, so the name of the model is kept
    modelName = model.GetName()
    if not slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentToRepresentationNode(segment, model):
      raise RuntimeError(f"Could not export the segment {segment.GetName()} to a model.")
    model.SetName(modelName)
    model.SetAndObserveTransformNodeID(segmentationNode.GetTransformNodeID())
    return model

  @staticmethod
  def saveModel(modelNode, filePath):
    """Saves the model to the file. The format is set by the file extension (e.g. .stl)."""
    if not slicer.util.saveNode(modelNode, filePath):
      raise RuntimeError(f"Could not save the model to {filePath}")

class EndoluminalPhantomBuilderTest(ScriptedLoadableModuleTest):
  """Tests for the Endoluminal Phantom Builder module. Run them with the Reload and Test button (developer mode) or
  through CTest. The lumen segmentations are generated by the tests."""

  def setUp(self):
    slicer.mrmlScene.Clear()
    self.logic = EndoluminalPhantomBuilderLogic()
    # Reset the parameters, which keep their values when the scene is cleared
    parameterNode = self.logic.getParameterNode()
    with slicer.util.NodeModify(parameterNode):
      parameterNode.lumenSegmentation = None
      parameterNode.lumenSegmentId = ''
      parameterNode.centerlineModel = None
      parameterNode.centerlineEndpoints = None
      parameterNode.wallSegmentation = None
      parameterNode.wallModel = None
      parameterNode.wallThickness = 4.0
      parameterNode.openingExtraDiameter = 3.0
      parameterNode.cutTracheaFlat = True
      parameterNode.tracheaCutExtraDiameter = 10.0
      parameterNode.surfaceSmoothingFactor = 0.5
      parameterNode.moldSegmentation = None
      parameterNode.moldThickness = 1.5
      parameterNode.cutMoldTracheaFlat = True
      parameterNode.moldTracheaCutExtraDiameter = 10.0
      parameterNode.moldSurfaceSmoothingFactor = 0.1
      parameterNode.moldModel = None

  def runTest(self):
    tests = [
      self.test_InputValidation,
      self.test_CreateWall,
      self.test_CreateWallFromClosedSurface,
      self.test_OpenEnds,
      self.test_OpenEndsOblique,
      self.test_OpeningExtraDiameter,
      self.test_KeepLargestIsland,
      self.test_CenterlineDirection,
      self.test_CutTracheaFlat,
      self.test_ExtractCenterline,
      self.test_ExportModel,
      self.test_SurfaceSmoothing,
      self.test_CreateMold,
      self.test_ExportMoldModel,
    ]
    self.skippedTests = []
    for test in tests:
      self.setUp()
      test()
    if self.skippedTests:
      self.delayDisplay("Skipped tests: " + ", ".join(self.skippedTests))
    self.delayDisplay("All Endoluminal Phantom Builder tests passed")

  # -- Helpers
  @staticmethod
  def createTubeLumen(name="Lumen"):
    """Creates a segmentation with a lumen segment: a tube (radius 5 mm) along the S axis that fills the whole height of
    its 40 x 40 x 40 mm geometry and is close to the R side, like an airway cut off by the edges of a CT. A second
    segment is stored in the same labelmap, to check that only the lumen is used. Returns the segmentation node and the
    lumen segment id."""
    k, j, i = np.mgrid[0:40, 0:40, 0:40]
    labels = np.zeros((40, 40, 40), np.uint8)
    labels[(i - 6) ** 2 + (j - 20) ** 2 <= 25] = 1
    labels[(i - 30) ** 2 + (j - 20) ** 2 + (k - 20) ** 2 <= 16] = 2
    labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", name + "Labels")
    slicer.util.updateVolumeFromArray(labelmapVolume, labels)
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    segmentationNode.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolume, segmentationNode)
    slicer.mrmlScene.RemoveNode(labelmapVolume)
    segmentation = segmentationNode.GetSegmentation()
    segmentation.GetNthSegment(0).SetName("Lumen")
    segmentation.GetNthSegment(1).SetName("Other")
    return segmentationNode, segmentation.GetNthSegmentID(0)

  @staticmethod
  def getSegmentArray(segmentationNode, segmentId):
    """Returns the segment as a boolean array (k, j, i) and the extent of the array."""
    labelmap = slicer.vtkOrientedImageData()
    segmentationNode.GetBinaryLabelmapRepresentation(segmentId, labelmap)
    extent = labelmap.GetExtent()
    shape = (extent[5] - extent[4] + 1, extent[3] - extent[2] + 1, extent[1] - extent[0] + 1)
    array = numpy_support.vtk_to_numpy(labelmap.GetPointData().GetScalars()).reshape(shape) > 0
    return array, extent

  def setInputs(self, segmentationNode, segmentId, start=(6, 20, 0), end=(6, 20, 39)):
    """Sets the inputs, with a straight centerline (points 0.5 mm apart) from the start to the end, which are also the
    endpoints. The defaults are the axis of the tube from createTubeLumen."""
    parameterNode = self.logic.getParameterNode()
    centerlineModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Centerline")
    line = vtk.vtkLineSource()
    line.SetPoint1(*start)
    line.SetPoint2(*end)
    line.SetResolution(max(int(np.linalg.norm(np.subtract(end, start)) * 2), 1))
    line.Update()
    centerlineModel.SetAndObservePolyData(line.GetOutput())
    centerlineEndpoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Endpoints")
    centerlineEndpoints.AddControlPoint(*start)
    centerlineEndpoints.AddControlPoint(*end)
    with slicer.util.NodeModify(parameterNode):
      parameterNode.lumenSegmentation = segmentationNode
      parameterNode.lumenSegmentId = segmentId
      parameterNode.centerlineModel = centerlineModel
      parameterNode.centerlineEndpoints = centerlineEndpoints

  # -- Tests
  def test_InputValidation(self):
    self.delayDisplay("Testing input validation")
    parameterNode = self.logic.getParameterNode()
    self.assertEqual(self.logic.getInputErrors(), ["Select the lumen segmentation.", "Select the centerline model.",
                                                   "Select the centerline endpoints."])
    segmentationNode, segmentId = self.createTubeLumen()
    parameterNode.lumenSegmentation = segmentationNode
    self.assertEqual(self.logic.getLumenError(), "Select the lumen segment.")
    self.setInputs(segmentationNode, segmentId)
    self.assertEqual(self.logic.getInputErrors(), [])

    parameterNode.centerlineModel.SetAndObservePolyData(vtk.vtkPolyData())
    parameterNode.centerlineEndpoints.RemoveNthControlPoint(1)
    self.assertEqual(self.logic.getInputErrors(), ["The centerline model has no lines.",
                                                   "The centerline endpoints need at least 2 points."])

    parameterNode.wallSegmentation = segmentationNode
    self.assertIn("must be different", self.logic.getLumenError())
    parameterNode.wallSegmentation = None

    self.delayDisplay("Input validation test passed")

  def test_CreateWall(self):
    self.delayDisplay("Testing wall creation")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    segmentationNode.SetAndObserveTransformNodeID(transform.GetID())
    self.setInputs(segmentationNode, segmentId)
    lumen, lumenExtent = self.getSegmentArray(segmentationNode, segmentId)
    parameterNode.wallThickness = 3.0
    # Other modules (e.g. the Segment Editor) can have their own Segment Editor parameters in the scene
    segmentEditorNodeCount = len(slicer.util.getNodesByClass("vtkMRMLSegmentEditorNode"))

    # The hollowed wall, before it is opened
    wallSegmentationNode, wallSegmentId, _ = self.logic.createHollowWall()
    self.assertEqual(wallSegmentationNode.GetName(), "Lumen_wall")
    self.assertEqual(wallSegmentationNode.GetTransformNodeID(), transform.GetID())
    self.assertIs(parameterNode.wallSegmentation, wallSegmentationNode)
    wallSegmentation = wallSegmentationNode.GetSegmentation()
    self.assertEqual(wallSegmentation.GetNumberOfSegments(), 1)
    self.assertEqual(wallSegmentation.GetSegment(wallSegmentId).GetName(), "Lumen_wall")
    self.assertEqual(self.logic.getWallSegmentId(), wallSegmentId)

    # The lumen is unchanged
    lumenAfter, lumenAfterExtent = self.getSegmentArray(segmentationNode, segmentId)
    self.assertEqual(lumenAfterExtent, lumenExtent)
    self.assertTrue(np.array_equal(lumenAfter, lumen))

    # The wall goes past the edges of the original geometry (I = 0 and S = 0..39) instead of being cut off there,
    # including the end caps of the tube, which a later step opens
    wall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    self.assertLess(wallExtent[0], 0)
    self.assertLess(wallExtent[4], 0)
    self.assertGreater(wallExtent[5], 39)

    # Compare the wall with the lumen in the same index space: the wall surrounds the lumen without overlapping it,
    # and is about 3 mm (3 voxels) thick, so it covers the voxels 1 to 3 mm away from the lumen
    # (the lumen array can extend past the wall, as it has the extent of the labelmap shared with the other segment)
    lumenInWall = np.zeros_like(wall)
    for k in range(lumen.shape[0]):
      for j in range(lumen.shape[1]):
        for i in np.nonzero(lumen[k, j])[0]:
          lumenInWall[k + lumenExtent[4] - wallExtent[4], j + lumenExtent[2] - wallExtent[2], i + lumenExtent[0] - wallExtent[0]] = True
    self.assertEqual(lumenInWall.sum(), lumen.sum())
    self.assertFalse(np.any(wall & lumenInWall))
    middleSlice = wall.shape[0] // 2
    centerJ = 20 - wallExtent[2]
    centerI = 6 - wallExtent[0]
    # Lumen radius is 5 voxels, so the wall is from 6 to 8 voxels away from the center along each direction
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      profile = []
      for distance in range(0, 11):
        j = centerJ + direction[0] * distance
        i = centerI + direction[1] * distance
        # The array ends just outside the wall
        profile.append(bool(0 <= j < wall.shape[1] and 0 <= i < wall.shape[2] and wall[middleSlice, j, i]))
      self.assertEqual(profile, [False] * 6 + [True] * 3 + [False] * 2, f"Wall profile along {direction}")
    # The other segment stored with the lumen (centered at I = 30) is not copied, so it does not widen the wall
    self.assertLess(wallExtent[1], 25)

    # Creating the wall again replaces it in the same segmentation
    parameterNode.wallThickness = 2.0
    wallSegmentationNode2, wallSegmentId2, _ = self.logic.createHollowWall()
    self.assertIs(wallSegmentationNode2, wallSegmentationNode)
    self.assertEqual(wallSegmentation.GetNumberOfSegments(), 1)
    wall2, _ = self.getSegmentArray(wallSegmentationNode, wallSegmentId2)
    self.assertLess(wall2.sum(), wall.sum())
    # The Segment Editor parameters used to create the wall are removed
    self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLSegmentEditorNode")), segmentEditorNodeCount)
    self.delayDisplay("Wall creation test passed")

  def test_CreateWallFromClosedSurface(self):
    self.delayDisplay("Testing wall creation from a closed surface")
    # A lumen imported from a model has a closed surface source representation and a geometry that just fits it
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(5)
    cylinder.SetHeight(30)
    cylinder.SetResolution(48)
    cylinder.Update()
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SurfaceLumen")
    segmentationNode.CreateDefaultDisplayNodes()
    segmentation = segmentationNode.GetSegmentation()
    segmentation.SetSourceRepresentationName(slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName())
    geometry = slicer.vtkOrientedImageData()
    geometry.SetSpacing(0.5, 0.5, 0.5)
    geometry.SetOrigin(-5, -15, -5)
    geometry.SetExtent(0, 20, 0, 60, 0, 20)
    segmentation.SetConversionParameter(slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName(),
                                        slicer.vtkSegmentationConverter.SerializeImageGeometry(geometry))
    segmentId = segmentationNode.AddSegmentFromClosedSurfaceRepresentation(cylinder.GetOutput(), "Lumen")
    self.setInputs(segmentationNode, segmentId, (0, -15, 0), (0, 15, 0))
    self.logic.getParameterNode().wallThickness = 2.0

    # The hollowed wall, before it is opened, so the size of the whole wall can be checked
    wallSegmentationNode, wallSegmentId, _ = self.logic.createHollowWall()
    wallSegmentationNode.CreateClosedSurfaceRepresentation()
    # The lumen segmentation is unchanged
    self.assertEqual(segmentationNode.GetSegmentation().GetSourceRepresentationName(),
                     slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName())
    wallSurface = vtk.vtkPolyData()
    wallSegmentationNode.GetClosedSurfaceRepresentation(wallSegmentId, wallSurface)
    bounds = wallSurface.GetBounds()
    # The wall is about 2 mm outside the 10 mm wide, 30 mm tall cylinder on every side (cylinder axis is along A)
    self.assertAlmostEqual(bounds[1] - bounds[0], 14, delta=1.5)
    self.assertAlmostEqual(bounds[3] - bounds[2], 34, delta=1.5)
    self.assertAlmostEqual(bounds[5] - bounds[4], 14, delta=1.5)
    self.delayDisplay("Closed surface wall creation test passed")

  def test_OpenEnds(self):
    self.delayDisplay("Testing opening the ends of the wall")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    self.setInputs(segmentationNode, segmentId)
    # An endpoint away from the lumen is not opened
    parameterNode.centerlineEndpoints.AddControlPoint(30, 5, 20)
    parameterNode.wallThickness = 3.0

    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    openings = self.logic.lastOpenings
    self.assertEqual(len(openings), 3)
    # The lumen has a radius of 5 voxels (1 mm), so its surface is about 5 mm from the axis
    for opening in openings[:2]:
      self.assertIsNone(opening['error'])
      self.assertAlmostEqual(opening['radius'], 5.0, delta=0.75)
    self.assertIsNone(openings[2]['radius'])
    self.assertIn("not at the lumen", openings[2]['error'])

    wall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    def isWall(i, j, k):
      index = (k - wallExtent[4], j - wallExtent[2], i - wallExtent[0])
      return bool(all(0 <= index[axis] < wall.shape[axis] for axis in range(3)) and wall[index])
    # Past each end of the lumen (S = 0 and 39), the wall in front of the lumen is erased...
    for k in [-1, -2, -3, 40, 41, 42]:
      for j in range(15, 26):
        for i in range(1, 12):
          if (i - 6) ** 2 + (j - 20) ** 2 <= 25:
            self.assertFalse(isWall(i, j, k), f"Wall left in the opening at {(i, j, k)}")
    # ...but the wall along the sides is kept, except close to the openings
    for k in range(10, 30):
      self.assertTrue(isWall(6, 20 + 7, k), f"Wall missing at S = {k}")
      self.assertTrue(isWall(6, 20 - 7, k), f"Wall missing at S = {k}")
      self.assertTrue(isWall(6 - 7, 20, k), f"Wall missing at S = {k}")
    self.delayDisplay("Opening test passed")

  def test_OpenEndsOblique(self):
    self.delayDisplay("Testing opening the ends of an oblique tube")
    parameterNode = self.logic.getParameterNode()
    # A tube (radius 4 mm) along a diagonal, inside a 50 x 50 x 50 mm volume, with flat ends at its endpoints
    start = np.array([12.0, 15.0, 12.0])
    end = np.array([36.0, 33.0, 36.0])
    axis = (end - start) / np.linalg.norm(end - start)
    k, j, i = np.mgrid[0:50, 0:50, 0:50]
    relative = np.stack([i, j, k], axis=-1) - start
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[..., np.newaxis] * axis, axis=-1)
    labels = ((radial <= 4) & (axial >= 0) & (axial <= np.linalg.norm(end - start))).astype(np.uint8)
    labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.util.updateVolumeFromArray(labelmapVolume, labels)
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ObliqueLumen")
    segmentationNode.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolume, segmentationNode)
    slicer.mrmlScene.RemoveNode(labelmapVolume)
    segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)
    self.setInputs(segmentationNode, segmentId, start, end)
    parameterNode.wallThickness = 2.0

    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    for opening in self.logic.lastOpenings:
      self.assertIsNone(opening['error'])
      # Within a voxel, as the spheres are centered on voxels and the surface of the lumen is jagged along a diagonal
      self.assertAlmostEqual(opening['radius'], 4.0, delta=1.0)
    length = np.linalg.norm(end - start)
    def getWallVoxels(segmentId):
      """Returns the positions (i, j, k, which are also in mm) of the wall voxels, with their axial and radial
      positions relative to the tube."""
      wall, wallExtent = self.getSegmentArray(wallSegmentationNode, segmentId)
      k, j, i = np.nonzero(wall)
      voxels = np.stack([i + wallExtent[0], j + wallExtent[2], k + wallExtent[4]], axis=-1)
      axial = (voxels - start) @ axis
      radial = np.linalg.norm(voxels - start - axial[:, np.newaxis] * axis, axis=-1)
      return voxels, axial, radial
    voxels, axial, radial = getWallVoxels(wallSegmentId)
    # No wall in front of either end of the lumen, through where the caps were
    self.assertFalse(np.any((radial <= 3.5) & ((axial < 0) | (axial > length))))
    # The wall around the middle of the tube is the same as without openings
    _, closedWallSegmentId, _ = self.logic.createHollowWall()
    closedVoxels, closedAxial, _ = getWallVoxels(closedWallSegmentId)
    middleVoxels = {tuple(voxel) for voxel in voxels[(axial > 8) & (axial < length - 8)]}
    closedMiddleVoxels = {tuple(voxel) for voxel in closedVoxels[(closedAxial > 8) & (closedAxial < length - 8)]}
    self.assertGreater(len(closedMiddleVoxels), 0)
    self.assertEqual(middleVoxels, closedMiddleVoxels)
    self.delayDisplay("Oblique opening test passed")

  def test_OpeningExtraDiameter(self):
    self.delayDisplay("Testing the extra diameter of the openings")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    self.setInputs(segmentationNode, segmentId)
    parameterNode.wallThickness = 3.0
    # Both ends are opened with spheres
    parameterNode.cutTracheaFlat = False
    # The sphere at S = 0 reaches the outside of the wall (7 to 8 mm from the axis) about 4 mm back from the end of
    # the lumen without extra diameter (radius 4.6 + 3 mm), and further back with 6 mm of extra diameter (radius 3 mm more)
    parameterNode.openingExtraDiameter = 0.0
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    wall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    self.assertTrue(wall[6 - wallExtent[4], 20 - wallExtent[2], 6 - 7 - wallExtent[0]])
    parameterNode.openingExtraDiameter = 6.0
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    widerWall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    self.assertFalse(widerWall[6 - wallExtent[4], 20 - wallExtent[2], 6 - 7 - wallExtent[0]])
    self.assertLess(widerWall.sum(), wall.sum())
    self.delayDisplay("Opening extra diameter test passed")

  def test_KeepLargestIsland(self):
    self.delayDisplay("Testing that only the largest piece of the wall is kept")
    parameterNode = self.logic.getParameterNode()
    # The tube from createTubeLumen, plus a separate small lumen (the "Other" segment merged into the lumen), whose wall
    # is a separate piece
    segmentationNode, segmentId = self.createTubeLumen()
    otherSegmentId = segmentationNode.GetSegmentation().GetNthSegmentID(1)
    otherLabelmap = slicer.vtkOrientedImageData()
    segmentationNode.GetBinaryLabelmapRepresentation(otherSegmentId, otherLabelmap)
    slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
      otherLabelmap, segmentationNode, segmentId, slicer.vtkSlicerSegmentationsModuleLogic.MODE_MERGE_MAX)
    self.setInputs(segmentationNode, segmentId)
    parameterNode.wallThickness = 2.0

    hollowSegmentationNode, hollowSegmentId, _ = self.logic.createHollowWall()
    hollowWall, hollowExtent = self.getSegmentArray(hollowSegmentationNode, hollowSegmentId)
    # Hollowing gives a wall around the small lumen (centered at I = 30, S = 20)
    self.assertTrue(hollowWall[20 - hollowExtent[4], 20 - hollowExtent[2], 30 + 5 - hollowExtent[0]])

    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    wall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    # Only the wall around the tube is kept
    self.assertFalse(np.any(wall[:, :, max(20 - wallExtent[0], 0):]))
    self.assertTrue(wall[20 - wallExtent[4], 20 + 7 - wallExtent[2], 6 - wallExtent[0]])
    self.delayDisplay("Keep largest island test passed")

  def test_CenterlineDirection(self):
    self.delayDisplay("Testing the direction of the centerline at an endpoint")
    # An L-shaped centerline: 10 mm along R, then 10 mm along A, with points 1 mm apart
    lShape = np.array([[x, 0, 0] for x in range(11)] + [[10, y, 0] for y in range(1, 11)], float)
    otherLine = np.array([[0, 0, 30], [0, 0, 50]], float)
    lines = [lShape, otherLine]
    # At the end of the second leg, a 5 mm direction distance stays within the leg
    distance, direction = EndoluminalPhantomBuilderLogic.getCenterlineDirection(lines, np.array([10, 10.5, 0]), 5.0)
    self.assertAlmostEqual(distance, 0.5)
    np.testing.assert_allclose(direction, [0, 1, 0], atol=1e-9)
    # At the start, the direction points out of the start of the line
    _, direction = EndoluminalPhantomBuilderLogic.getCenterlineDirection(lines, np.array([0, 0, 0]), 5.0)
    np.testing.assert_allclose(direction, [-1, 0, 0], atol=1e-9)
    # A longer distance goes around the corner, which averages the two directions
    _, direction = EndoluminalPhantomBuilderLogic.getCenterlineDirection(lines, np.array([10, 10, 0]), 20.0)
    np.testing.assert_allclose(direction, np.array([1, 1, 0]) / np.sqrt(2), atol=1e-9)
    # The closest line is used, and a line shorter than the distance is followed to its other end
    _, direction = EndoluminalPhantomBuilderLogic.getCenterlineDirection(lines, np.array([0, 0, 52]), 50.0)
    np.testing.assert_allclose(direction, [0, 0, 1], atol=1e-9)
    self.delayDisplay("Centerline direction test passed")

  def test_CutTracheaFlat(self):
    self.delayDisplay("Testing the flat cut of the trachea opening")
    parameterNode = self.logic.getParameterNode()
    # A tube along S, wide at the top like the trachea (S = 25 to 45, an oval 9 mm along R and 6 mm along A) and
    # narrow at the bottom like a bronchus (radius 3 mm, S = 5 to 24)
    k, j, i = np.mgrid[0:50, 0:50, 0:50]
    inTrachea = ((i - 25) / 9.0) ** 2 + ((j - 25) / 6.0) ** 2 <= 1
    labels = ((inTrachea & (k >= 25) & (k <= 45)) | ((np.hypot(i - 25, j - 25) <= 3) & (k >= 5) & (k < 25))).astype(np.uint8)
    labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.util.updateVolumeFromArray(labelmapVolume, labels)
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "TracheaLumen")
    segmentationNode.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolume, segmentationNode)
    slicer.mrmlScene.RemoveNode(labelmapVolume)
    segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)
    self.setInputs(segmentationNode, segmentId, (25, 25, 5), (25, 25, 45))
    parameterNode.wallThickness = 2.0
    # Only the trachea cut extra diameter widens the cut
    parameterNode.openingExtraDiameter = 0.0

    # The cut needs the centerline model
    centerlineModel = parameterNode.centerlineModel
    parameterNode.centerlineModel = None
    self.assertIn("needed to cut the trachea flat", " ".join(self.logic.getWallErrors()))
    parameterNode.centerlineModel = centerlineModel

    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    bronchusOpening, tracheaOpening = self.logic.lastOpenings
    self.assertFalse(bronchusOpening['trachea'])
    self.assertTrue(tracheaOpening['trachea'])
    self.assertIsNone(tracheaOpening['error'])
    self.assertGreater(tracheaOpening['radius'], bronchusOpening['radius'])

    wall, wallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    def isWall(wall, wallExtent, i, j, k):
      index = (k - wallExtent[4], j - wallExtent[2], i - wallExtent[0])
      return bool(all(0 <= index[axis] < wall.shape[axis] for axis in range(3)) and wall[index])
    def isTracheaOpen(wall, wallExtent):
      """Returns whether there is no wall past the end of the trachea (S = 45), in front of the lumen."""
      for k in [46, 47, 48]:
        for j in range(19, 32):
          for i in range(16, 35):
            if ((i - 25) / 9.0) ** 2 + ((j - 25) / 6.0) ** 2 <= 1 and isWall(wall, wallExtent, i, j, k):
              return False
      return True
    # The trachea is open across its whole end...
    self.assertTrue(isTracheaOpen(wall, wallExtent))
    # ...and its side wall is kept right up to the cut across the endpoint (in the last layer of the lumen, S = 45),
    # where a sphere would have trimmed it back
    for k in [42, 43, 44]:
      self.assertTrue(isWall(wall, wallExtent, 25 + 10, 25, k), f"Trachea wall missing at S = {k}")
      self.assertTrue(isWall(wall, wallExtent, 25, 25 - 7, k), f"Trachea wall missing at S = {k}")

    # Without extra diameter, the cut is as wide as the narrowest width of the trachea (radius 6 mm) plus the wall,
    # which does not reach across the wide direction of the trachea (9 mm)
    parameterNode.tracheaCutExtraDiameter = 0.0
    narrowCutSegmentationNode, narrowCutSegmentId = self.logic.createWall()
    narrowCutWall, narrowCutWallExtent = self.getSegmentArray(narrowCutSegmentationNode, narrowCutSegmentId)
    self.assertFalse(isTracheaOpen(narrowCutWall, narrowCutWallExtent))
    # The bronchus is opened with a sphere
    for k in [2, 3, 4]:
      self.assertFalse(isWall(wall, wallExtent, 25, 25, k), f"Wall left in the bronchus opening at S = {k}")

    parameterNode.cutTracheaFlat = False
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    self.assertFalse(any(opening['trachea'] for opening in self.logic.lastOpenings))
    sphereWall, sphereWallExtent = self.getSegmentArray(wallSegmentationNode, wallSegmentId)
    self.assertFalse(isWall(sphereWall, sphereWallExtent, 25, 25 - 7, 44))
    self.delayDisplay("Flat trachea cut test passed")

  def test_ExtractCenterline(self):
    if not EndoluminalPhantomBuilderLogic.isExtractCenterlineAvailable():
      logging.info("test_ExtractCenterline skipped: it needs the SlicerVMTK extension")
      self.skippedTests.append("test_ExtractCenterline (needs SlicerVMTK)")
      return
    self.delayDisplay("Testing centerline extraction with SlicerVMTK")
    parameterNode = self.logic.getParameterNode()
    # A Y-shaped lumen: a wide trachea (radius 6 mm) along S that splits into two narrower bronchi (radius 4 mm)
    k, j, i = np.mgrid[0:70, 0:40, 0:70]
    trachea = (np.hypot(i - 35, j - 20) <= 6) & (k >= 35) & (k <= 62)
    labels = trachea.copy()
    for side in [-1, 1]:
      # Each bronchus goes 45 degrees down and to the side from the end of the trachea
      start = np.array([35, 20, 36])
      axis = np.array([side, 0, -1]) / np.sqrt(2)
      relative = np.stack([i, j, k], axis=-1) - start
      axial = relative @ axis
      radial = np.linalg.norm(relative - axial[..., np.newaxis] * axis, axis=-1)
      labels |= (radial <= 4) & (axial >= 0) & (axial <= 35)
    labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.util.updateVolumeFromArray(labelmapVolume, labels.astype(np.uint8))
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "AirwayLumen")
    segmentationNode.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolume, segmentationNode)
    slicer.mrmlScene.RemoveNode(labelmapVolume)
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    segmentationNode.SetAndObserveTransformNodeID(transform.GetID())
    with slicer.util.NodeModify(parameterNode):
      parameterNode.lumenSegmentation = segmentationNode
      parameterNode.lumenSegmentId = segmentationNode.GetSegmentation().GetNthSegmentID(0)

    centerlineModel, centerlineEndpoints = self.logic.extractCenterline()
    self.assertIs(parameterNode.centerlineModel, centerlineModel)
    self.assertIs(parameterNode.centerlineEndpoints, centerlineEndpoints)
    self.assertEqual(centerlineModel.GetTransformNodeID(), transform.GetID())
    self.assertEqual(centerlineEndpoints.GetTransformNodeID(), transform.GetID())
    self.assertEqual(self.logic.getInputErrors(), [])
    # The trachea and the two bronchi each have an end, and the trachea is the start point (first and unselected)
    self.assertEqual(centerlineEndpoints.GetNumberOfControlPoints(), 3)
    tracheaEnd = centerlineEndpoints.GetNthControlPointPosition(0)
    self.assertGreater(tracheaEnd[2], 55)
    self.assertFalse(centerlineEndpoints.GetNthControlPointSelected(0))
    self.assertGreater(centerlineModel.GetPolyData().GetNumberOfLines(), 0)

    # Extracting again replaces the extracted nodes instead of adding new ones
    modelCount = len(slicer.util.getNodesByClass("vtkMRMLModelNode"))
    centerlineModel2, centerlineEndpoints2 = self.logic.extractCenterline()
    self.assertIs(centerlineModel2, centerlineModel)
    self.assertIs(centerlineEndpoints2, centerlineEndpoints)
    self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLModelNode")), modelCount)

    # The wall can be made from the extracted centerline, with the trachea cut flat
    self.logic.createWall()
    self.assertEqual([opening['error'] for opening in self.logic.lastOpenings], [None] * 3)
    self.assertTrue(self.logic.lastOpenings[0]['trachea'])
    self.delayDisplay("Centerline extraction test passed")

  def test_ExportModel(self):
    self.delayDisplay("Testing the export of the wall to a model")
    parameterNode = self.logic.getParameterNode()
    self.assertEqual(self.logic.getCreatedWallError(), "Create the wall first.")
    segmentationNode, segmentId = self.createTubeLumen()
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    segmentationNode.SetAndObserveTransformNodeID(transform.GetID())
    self.setInputs(segmentationNode, segmentId)
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    self.assertIsNone(self.logic.getCreatedWallError())

    wallModel = self.logic.exportWallModel()
    self.assertIs(parameterNode.wallModel, wallModel)
    self.assertEqual(wallModel.GetName(), "Lumen_wall_model")
    self.assertEqual(wallModel.GetTransformNodeID(), transform.GetID())
    wallSurface = vtk.vtkPolyData()
    wallSegmentationNode.GetClosedSurfaceRepresentation(wallSegmentId, wallSurface)
    self.assertGreater(wallModel.GetPolyData().GetNumberOfPoints(), 0)
    np.testing.assert_allclose(wallModel.GetPolyData().GetBounds(), wallSurface.GetBounds(), atol=1e-3)

    # Exporting again replaces the model
    modelCount = len(slicer.util.getNodesByClass("vtkMRMLModelNode"))
    self.assertIs(self.logic.exportWallModel(), wallModel)
    self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLModelNode")), modelCount)

    # Saving as STL writes a file with the same surface
    import tempfile, shutil
    directory = tempfile.mkdtemp()
    try:
      filePath = os.path.join(directory, "wall.stl")
      self.logic.saveModel(wallModel, filePath)
      self.assertTrue(os.path.getsize(filePath) > 0)
      savedModel = slicer.util.loadModel(filePath)
      self.assertEqual(savedModel.GetPolyData().GetNumberOfCells(), wallModel.GetPolyData().GetNumberOfCells())
    finally:
      shutil.rmtree(directory, ignore_errors=True)
    self.delayDisplay("Model export test passed")

  def test_SurfaceSmoothing(self):
    self.delayDisplay("Testing the smoothing of the wall surface")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    self.setInputs(segmentationNode, segmentId)
    smoothingFactorName = slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName()
    def getSurfacePoints(wallSegmentationNode, wallSegmentId):
      surface = vtk.vtkPolyData()
      wallSegmentationNode.GetClosedSurfaceRepresentation(wallSegmentId, surface)
      return numpy_support.vtk_to_numpy(surface.GetPoints().GetData()).copy()

    # The wall is created with the smoothing factor
    parameterNode.surfaceSmoothingFactor = 0.0
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    self.assertEqual(float(wallSegmentationNode.GetSegmentation().GetConversionParameter(smoothingFactorName)), 0.0)
    unsmoothedPoints = getSurfacePoints(wallSegmentationNode, wallSegmentId)
    # Without smoothing, the surface follows the voxels (1 mm): its points are halfway between voxel centers, so every
    # coordinate is a multiple of 0.5 mm
    np.testing.assert_allclose(np.mod(unsmoothedPoints + 0.25, 0.5) - 0.25, 0.0, atol=1e-6)

    # Changing the smoothing factor updates the surface of the existing wall, and the exported model
    parameterNode.surfaceSmoothingFactor = 0.8
    self.logic.updateSurfaces()
    self.assertEqual(float(wallSegmentationNode.GetSegmentation().GetConversionParameter(smoothingFactorName)), 0.8)
    smoothedPoints = getSurfacePoints(wallSegmentationNode, wallSegmentId)
    self.assertGreater(np.abs(np.mod(smoothedPoints + 0.25, 0.5) - 0.25).max(), 0.01)
    wallModel = self.logic.exportWallModel()
    np.testing.assert_allclose(numpy_support.vtk_to_numpy(wallModel.GetPolyData().GetPoints().GetData()), smoothedPoints)
    self.delayDisplay("Surface smoothing test passed")

  def test_CreateMold(self):
    self.delayDisplay("Testing mold creation")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    segmentationNode.SetAndObserveTransformNodeID(transform.GetID())
    self.setInputs(segmentationNode, segmentId)
    parameterNode.wallThickness = 2.0
    parameterNode.moldThickness = 1.0
    self.assertIn("Create the wall first.", self.logic.getMoldErrors())
    wallSegmentationNode, wallSegmentId = self.logic.createWall()
    self.assertEqual(self.logic.getMoldErrors(), [])
    parameterNode.moldSegmentation = wallSegmentationNode
    self.assertIn("must be different", " ".join(self.logic.getMoldErrors()))
    parameterNode.moldSegmentation = None
    wallVoxelCount = self.getSegmentArray(wallSegmentationNode, wallSegmentId)[0].sum()

    moldSegmentationNode, moldSegmentId = self.logic.createMold()
    self.assertEqual(moldSegmentationNode.GetName(), "Lumen_mold")
    self.assertIs(parameterNode.moldSegmentation, moldSegmentationNode)
    self.assertEqual(moldSegmentationNode.GetTransformNodeID(), transform.GetID())
    self.assertEqual(moldSegmentationNode.GetSegmentation().GetNumberOfSegments(), 1)
    self.assertEqual(moldSegmentationNode.GetSegmentation().GetSegment(moldSegmentId).GetName(), "Lumen_mold")
    # The wall is unchanged
    self.assertEqual(self.getSegmentArray(wallSegmentationNode, wallSegmentId)[0].sum(), wallVoxelCount)

    def getVoxels(segmentationNode, segmentId):
      """Returns the voxels of the segment as a set of (i, j, k)."""
      array, extent = self.getSegmentArray(segmentationNode, segmentId)
      k, j, i = np.nonzero(array)
      return set(zip((i + extent[0]).tolist(), (j + extent[2]).tolist(), (k + extent[4]).tolist()))
    mold = getVoxels(moldSegmentationNode, moldSegmentId)
    wall = getVoxels(wallSegmentationNode, wallSegmentId)
    # The mold encloses the wall without overlapping it
    self.assertFalse(mold & wall)
    # Lumen radius is 5 voxels and the wall is 2 voxels thick (6 and 7 voxels from the center), so the 1 voxel thick
    # mold is 5 voxels from the center (lining the lumen) and 8 voxels from the center (around the outside)
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      profile = [(6 + direction[0] * distance, 20 + direction[1] * distance, 20) in mold for distance in range(0, 11)]
      self.assertEqual(profile, [False] * 5 + [True, False, False, True] + [False] * 2, f"Mold profile along {direction}")
    # The trachea end (S = 0, the first endpoint, as both ends are as wide) is cut open, so the end of the wall is not
    # covered there
    self.assertFalse(any(k <= 0 and math.hypot(i - 6, j - 20) <= 12 for i, j, k in mold))
    # The mold follows the wall, so the other end is open where the wall was opened
    self.assertNotIn((6, 20, 40), mold)

    # Without the trachea cut, the end of the wall at the trachea is covered by the mold
    parameterNode.cutMoldTracheaFlat = False
    moldSegmentationNode, moldSegmentId = self.logic.createMold()
    self.assertEqual(moldSegmentationNode.GetSegmentation().GetNumberOfSegments(), 1)
    uncutMold = getVoxels(moldSegmentationNode, moldSegmentId)
    self.assertIn((6, 20 + 6, 0), uncutMold)
    self.assertNotIn((6, 20 + 6, 0), mold)

    # The mold has its own surface smoothing
    smoothingFactorName = slicer.vtkBinaryLabelmapToClosedSurfaceConversionRule.GetSmoothingFactorParameterName()
    def getSurfaceArea(segmentationNode, segmentId):
      surface = vtk.vtkPolyData()
      segmentationNode.GetClosedSurfaceRepresentation(segmentId, surface)
      massProperties = vtk.vtkMassProperties()
      massProperties.SetInputData(surface)
      massProperties.Update()
      return massProperties.GetSurfaceArea()
    self.assertEqual(float(moldSegmentationNode.GetSegmentation().GetConversionParameter(smoothingFactorName)), 0.1)
    self.assertEqual(float(wallSegmentationNode.GetSegmentation().GetConversionParameter(smoothingFactorName)),
                     parameterNode.surfaceSmoothingFactor)
    smoothedArea = getSurfaceArea(moldSegmentationNode, moldSegmentId)
    parameterNode.moldSurfaceSmoothingFactor = 0.0
    self.logic.updateSurfaces()
    unsmoothedArea = getSurfaceArea(moldSegmentationNode, moldSegmentId)
    self.assertGreater(smoothedArea, 0.9 * unsmoothedArea)

    parameterNode.moldSurfaceSmoothingFactor = parameterNode.surfaceSmoothingFactor
    self.logic.updateSurfaces()
    self.assertLess(getSurfaceArea(moldSegmentationNode, moldSegmentId), 0.5 * unsmoothedArea)
    self.delayDisplay("Mold creation test passed")

  def test_ExportMoldModel(self):
    self.delayDisplay("Testing the export of the mold to a model")
    parameterNode = self.logic.getParameterNode()
    segmentationNode, segmentId = self.createTubeLumen()
    self.setInputs(segmentationNode, segmentId)
    parameterNode.wallThickness = 2.0
    parameterNode.moldThickness = 2.0
    self.logic.createWall()
    self.assertEqual(self.logic.getCreatedMoldError(), "Create the mold first.")
    moldSegmentationNode, moldSegmentId = self.logic.createMold()
    self.assertIsNone(self.logic.getCreatedMoldError())
    self.assertEqual(self.logic.getMoldSegmentId(), moldSegmentId)

    wallModel = self.logic.exportWallModel()
    moldModel = self.logic.exportMoldModel()
    self.assertIs(parameterNode.moldModel, moldModel)
    self.assertIsNot(moldModel, wallModel)
    self.assertEqual(moldModel.GetName(), "Lumen_mold_model")
    moldSurface = vtk.vtkPolyData()
    moldSegmentationNode.GetClosedSurfaceRepresentation(moldSegmentId, moldSurface)
    np.testing.assert_allclose(moldModel.GetPolyData().GetBounds(), moldSurface.GetBounds(), atol=1e-3)
    # Exporting again replaces the model
    self.assertIs(self.logic.exportMoldModel(), moldModel)
    self.delayDisplay("Mold model export test passed")
