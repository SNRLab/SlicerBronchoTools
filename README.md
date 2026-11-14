# BronchoTools

3D Slicer extension containing tools for image-guided bronchoscopy.

## Modules

- **Video Input**: Streams frames from a webcam, video file, or image sequence into a volume node.

## Installation (development)

Add the module folders (e.g. `SlicerBronchoTools/VideoInput`) to 3D Slicer's Additional module paths
(Edit > Application Settings > Modules), then restart Slicer.

## Python dependencies

Installed on first use (after a confirmation prompt) into Slicer's Python environment:

- **Video Input**: `opencv-python-headless` (capture), `cv2_enumerate_cameras` (camera names on Windows, Linux, and macOS)

## Testing

Each module contains a self-test (e.g. `VideoInputTest` in `VideoInput/VideoInput.py`). In Slicer, enable developer mode
(Edit > Application Settings > Developer) and press **Reload and Test** in the module. The tests generate their own
video and image files, so no test data needs to be downloaded. When the extension is built with `BUILD_TESTING`
enabled, the tests are also registered with CTest.

## License

BSD 3-Clause. See [LICENSE.txt](LICENSE.txt).
