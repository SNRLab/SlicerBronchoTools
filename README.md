# BronchoTools

3D Slicer extension containing tools for image-guided bronchoscopy.

![BronchoTools: depth map and point cloud generated live from bronchoscope video](Images/BronchoTools.webp)

## Modules

- **Video Input**: Streams frames from a webcam, video file, or image sequence into a volume node.
- **Bronchoscopic Depth Map Generation**: Estimates a depth map from a 2D bronchoscopic image with a deep learning model,
  once or continuously on a live image (e.g. from Video Input). It is intended for monoscopic (single camera) bronchoscope
  footage. The depth map can be converted into a point cloud using
  the camera intrinsics, optionally placed at a camera pose given by a transform.

## Installation (development)

Add the module folders (e.g. `SlicerBronchoTools/VideoInput`) to 3D Slicer's Additional module paths
(Edit > Application Settings > Modules), then restart Slicer.

## Python dependencies

Installed on first use (after a confirmation prompt) into Slicer's Python environment:

- **Video Input**: `opencv-python-headless` (capture), `cv2_enumerate_cameras` (camera names on Windows, Linux, and macOS)
- **Bronchoscopic Depth Map Generation**: PyTorch, which the user installs with the
  [PyTorch extension](https://github.com/fepegar/SlicerPyTorch) (an extension dependency, so the Extensions Manager installs the extension automatically): open
  Utilities > PyTorch Utils and install PyTorch, which picks the CUDA or CPU build that matches the computer.
  Model weights are downloaded into Slicer's cache folder on first use.

## Sample data

A short porcine bronchoscopy recording is included with the Bronchoscopic Depth Map Generation module, in its
`Resources/SampleData` folder:

- `porcine.mkv`: the video
- `porcine_frames/`: its 90 frames as JPEG images

To try it, open Video Input and select the video (Video input) or the frames folder (Images input), start playback, and
select Video Input's output volume as the input of Bronchoscopic Depth Map Generation. A single frame can also be loaded
directly with File > Add Data.

When the extension is installed from the Extensions Manager, the module folder can be found from Slicer's Python
console:

```python
import os
print(os.path.join(os.path.dirname(slicer.modules.bronchoscopicdepthmapgeneration.path), "Resources", "SampleData"))
```

## Testing

Each module contains a self-test (e.g. `VideoInputTest` in `VideoInput/VideoInput.py`). In Slicer, enable developer mode
(Edit > Application Settings > Developer) and press **Reload and Test** in the module.

## License

BSD 3-Clause. See [LICENSE.txt](LICENSE.txt).
