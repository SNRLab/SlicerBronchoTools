# BronchoTools

3D Slicer extension containing tools for image-guided bronchoscopy.

**Documentation and tutorials: see the [BronchoTools wiki](https://github.com/SNRLab/SlicerBronchoTools/wiki).**

![BronchoTools: depth map and point cloud generated live from bronchoscope video](Images/BronchoTools.webp)

## Modules

- [Video Input](https://github.com/SNRLab/SlicerBronchoTools/wiki/Video-Input): streams a webcam, video file or image sequence into a volume.
- [Bronchoscopic Depth Map Generation](https://github.com/SNRLab/SlicerBronchoTools/wiki/Bronchoscopic-Depth-Map-Generation): estimates depth maps and point clouds from monoscopic bronchoscope images.
- [Endoluminal Phantom Builder](https://github.com/SNRLab/SlicerBronchoTools/wiki/Endoluminal-Phantom-Builder): builds a wall and a casting mold from a lumen segmentation to make a physical phantom.

## Dependencies

- **Bronchoscopic Depth Map Generation** requires PyTorch, installed with the Slicer [PyTorch extension](https://github.com/fepegar/SlicerPyTorch).
- **Endoluminal Phantom Builder** can generate centerlines automatically if [SlicerVMTK](https://github.com/vmtk/SlicerExtension-VMTK) is installed.

Sample data is included. See [Sample data](https://github.com/SNRLab/SlicerBronchoTools/wiki#sample-data).

## License

BSD 3-Clause. See [LICENSE.txt](LICENSE.txt).
