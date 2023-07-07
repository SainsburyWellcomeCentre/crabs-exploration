# Collecting calibration data

## Overview and rationale?
...

## Recommendations

### 0. Arrangement of the cameras
There are very nice recommendations on the arrangement of the cameras in the [StereoMorph docs](https://aaronolsen.github.io/tutorials/stereomorph/arranging_cameras_photography.html)
- As much FOV overlap as possible? --- see StereoMorph
- Remember! We are calibrating the cameras for a specific relative position of the cameras, zoom and focus levels!
    - From the [choosing cameras section](https://aaronolsen.github.io/tutorials/stereomorph/choosing_cameras_general.html) > "the resulting calibration is specific to that particular arrangement of the cameras (their position and orientation). The cameras can move if they are rigidly attached to some support system and the support system itself is moved. However, the relative position and orientation of the cameras relative to one another must remain fixed to collect accurate 3D data. Additionally, the calibration is specific to the particular zoom and focus, so these must also remain fixed.
    - 
- Set the cameras to the smallest aperture

### 1. Design of the checkerboard
- The pattern should be on a flat and rigid surface 

- White border around it, of roughly the same width as one of the cells (this makes the detection more robust in various environments see [here](https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91))

- Min number of **inner corners**
    - *Why inner corners?*: "All chessboard related algorithm expects amount of inner corners as board width and height" [^2] - openCV parametrises the width and height based on number of inner corners
    - 



- Creating a calibration pattern
    - [OpenCV](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
    - [Calib.io](https://calib.io/pages/camera-calibration-pattern-generator)
    - [Mark Hedley Jones blog post](https://markhedleyjones.com/projects/calibration-checkerboard-collection) --- not sure if these print out with a white border!

### 2. Position and orientation of the checkerboard
For a calibration frame to be useful, the checkerboard must fulfill the following criteria:

- Keep the orientation of the checkboard within 90deg (conservatively 30 deg [^1]) of the "original"/default position
    - *Why?* The algorithm takes the bottom (right?) corner of the board as the origin, and then traverses the rest of corners from there. If in some frames the board is rotated beyond 30 deg, the origin may be detected at a different physical corner, and all the corners may be traversd in a different order.

- Cover several distances, and within each distance, cover (all parts of the image view)/(of the FOV overlap) [^1]
    - *Why?*
    - Note however that if at some distance the calibration pattern is too small, we may not be able to detect the corners reliably and the calibration may be poor. To fix this we may need a larger board (or move cameras closer to the area of interest?) 

- Use a checkerboard as big as possible, ideally with of at least 8x6 squares and a square size of at least 100 mm [^1]
    - *Why?*

- For a calibration frame to be useful, the checkerboard must be fully visible in both camers
    - *Why?* Since the algorithm first tries to search for the outer corners of the board, it will fail if the board is only partially visible. This may be avoidable using an alternative algorithm, but the approach [doesn't seem 100% robust](https://github.com/opencv/opencv/issues/15712#issuecomment-1493344373)

- Number of frames captured
    - From DLC, they claim 30-70 pairs of candidate calibration images should be sufficient, as after corner detection, some of the images might need to be discarded due to either incorrect corner detection or incorrect order of detected corners [^1]

- Keep the pattern steady when aiming to collect a calibration frame

## Timecode drift
- Use approach from Anipose as validation?

## Additional comments
- Acinoset paper may be a good reference on people doing stereo in the field, in a large area
- Should we try other patterns too (charuco boards, circles grid) ? Would it be more robust?

## Useful references
- [DeepLabCut 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#jump-in-with-direct-deeplabcut-2-camera-support)
- [Anipose]()
- [Aniposelib](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html)
- [OpenCV docs calib3d](https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91)


- [^1]: from [DeepLabCut 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#jump-in-with-direct-deeplabcut-2-camera-support), and [OpenPose](https://github.com/jrkwon/openpose/blob/master/doc/calibration_demo.md#general-quality-tips)

- [^2]: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

- [^3]: https://github.com/TemugeB/python_stereo_camera_calibrate#procedure

