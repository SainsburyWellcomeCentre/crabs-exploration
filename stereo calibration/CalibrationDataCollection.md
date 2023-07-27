# Collecting calibration data

## Overview and rationale

There is a nice overview on the objective of a typical calibration procedure [here](https://aaronolsen.github.io/tutorials/stereomorph/calibration_general.html).

There are three main requirements for good calibration data (see intro to [this paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-6-9040&id=407319))

- calibration target must be placed in the common FOV
- calibration target should cover at least 1/3 of the common FOV (this may be too strict for us at large distances)
- the features of the calibration target need to be in focus and sufficiently spatially resolved (!) to allow for centroid detection.

All of these become challenging with a wide FOV (> 1m width)

## Recommendations

### 0. Arrangement of the cameras

There are very nice recommendations on the arrangement of the cameras in the [StereoMorph docs](https://aaronolsen.github.io/tutorials/stereomorph/arranging_cameras_photography.html) (although they focus on taking pictures to reconstruct a 3D object)

- Remember! We are calibrating the cameras for a specific relative position of the cameras, but also zoom (focal length) and focus levels!
  - All must stay constant during calibration and data collection (see the [choosing cameras section](https://aaronolsen.github.io/tutorials/stereomorph/choosing_cameras_general.html)
- Re camera settings

  - They recommend to turn off auto-focus and auto-zoom modes, and vibration reduction (VR) if the lens has it.
  - They recommend setting the cameras to the smallest aperture
    > "The smaller the aperture, the greater the depth of field (i.e. the more things are in focus both close and far away from the camera). This is essential in a stereo camera setup because in order to digitize points accurately throughout the calibration volume they must be in focus."

- Re cameras' positions:

  - "Because the camera is often positioned half a meter or more away from the object, even a small shift of the camera can translate into a large shift in the image frame, causing large inaccuracies"
  - Leg stoppers ([this type](https://www.amazon.co.uk/Universal-Stainless-Photography-Accessories-Replacement/dp/B0B9NXFTWP/ref=sr_1_11?keywords=tripod+rubber+feet+replacement&qid=1689006351&sr=8-11) or [these with spikes for uneven surfaces](https://www.amazon.co.uk/Universal-Non-Slip-Replacement-Compatible-Kingjoy/dp/B0B33RDMTC/ref=sr_1_12?keywords=tripod+rubber+feet+replacement&qid=1689006351&sr=8-12)) could be useful to keep the tripods as fixed as possible against the wind and on top of uneven terrain
  - "It's best to use a remote (wireless or cord) to release the shutter so you minimizing touching the shutter button on the cameras as much as possible."

- In our case, we probably want as much FOV overlap as possible

### 1. Design of the checkerboard

- The pattern should be on a flat and rigid surface

  - From [StereoMorph](https://aaronolsen.github.io/tutorials/stereomorph/creating_a_checkerboard.html):
    "If you want an exceptionally flat surface, I recommend plexiglass (at least 0.22 inches thick)"

- It should have a white border around it, of roughly the same width as one of the cells (this makes the detection more robust in various environments see [here](https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91))

- Size of the checkerboard

  - From [StereoMorph docs](https://aaronolsen.github.io/tutorials/stereomorph/choose_checkerboard_size.html):
    - What volume is visible from the two (or more) cameras?
      - If we approximated it to a box, what is the shortest dimension (_n_)?
        - is it smaller than 0.5 m? Then the checkerboard side should be around half of the shortest dimension (_0.5n_)
          > For example, for a calibrated volume with approximate dimensions of 0.3 m x 0.35 m x 0.4 m, a checkerboard pattern measuring 0.15 m along one dimension should be ideal.
        - is it larger than 0.5 m? Then the checkerboard side should be around 0.5m

- Min number of **inner corners**

  - From [DLC 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#take-and-process-camera-calibration-images): at least 8x6 squares
  - _Why do we say inner corners?_: "All chessboard related algorithm expects amount of inner corners as board width and height" [^2] - openCV parametrises the width and height based on number of inner corners because this is what will actually be detected
  - The number of inner corners should be different along each dimension, to ensure they are returned in the same order when the chessboard changes orientation

- Min size of the cells

  - In [StereoMorph](https://aaronolsen.github.io/tutorials/stereomorph/checkerboard_size_dpi.html) they use 6.35 mm
  - In [OpenPose](https://github.com/jrkwon/openpose/blob/master/doc/calibration_demo.md#general-quality-tips) they recommend a square size of at least 100 mm

- Creating a calibration pattern

  - [OpenCV](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
  - [Calib.io](https://calib.io/pages/camera-calibration-pattern-generator)
  - [Mark Hedley Jones blog post](https://markhedleyjones.com/projects/calibration-checkerboard-collection) --- not sure if these print out with a white border!

- Consider printing it in cardstock paper (from [here](https://aaronolsen.github.io/tutorials/stereomorph/creating_a_checkerboard.html))
  - "Once taped or glued to a hard surface, cardstock is less likely to get bubbles from moisture over time."

### 2. Position and orientation of the checkerboard

For a calibration frame to be useful, the checkerboard must fulfill the following criteria:

- Keep the orientation of the checkboard within 90deg (conservatively 30 deg [^1]) of the "original"/default position

  - _Why?_ The algorithm takes the bottom (right?) corner of the board as the origin, and then traverses the rest of corners from there. If in some frames the board is rotated beyond 30 deg, the origin may be detected at a different physical corner, and all the corners may be traversed in a different order.

- Cover several distances, and within each distance, cover (all parts of the image view)/(of the FOV overlap) [^1]

  - We want to cover as much as possible of the calibration space (or the space where we want to collect data in)
  - _Why?_:
    > "If you only photograph the checkerboard in one area of the calibration volume, reconstruction errors could be relatively higher in other areas. Similarly, if you only photograph the checkerboard at a particular angle (e.g. 45 degrees), you won't have good sampling of points along each dimension of the space (since a checkerboard is a flat surface it can only sample two dimensions at any one time). This can causes reconstruction errors to be higher along particular dimensions than along others." (from [here](https://aaronolsen.github.io/tutorials/stereomorph/calibration_general.html))
  - Note however that if at some distance the calibration pattern is too small, we may not be able to detect the corners reliably and the calibration may be poor. To fix this we may need a larger board (or move cameras closer to the area of interest before calibrating)

- For a calibration frame to be useful, the checkerboard must be fully visible in both cameras

  - _Why?_ Since the algorithm first tries to search for the outer corners of the board, it will fail if the board is only partially visible. This may be avoidable using an alternative algorithm, but the approach [doesn't seem 100% robust](https://github.com/opencv/opencv/issues/15712#issuecomment-1493344373), so it's probably a good idea to be conservative here and keep it fully visible at all times.

- Number of frames captured

  - In [DeepLabCut 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#jump-in-with-direct-deeplabcut-2-camera-support) they claim 30-70 pairs of candidate calibration images should be sufficient, as after corner detection, some of the images might need to be discarded due to either incorrect corner detection or incorrect order of detected corners.

- Keep the pattern steady when aiming to collect a calibration frame

## Timecode drift

- The timecode option is sensitive to drift, because it relies on the clocks of the two cameras running perfectly in sync (right?)
  - In reality, one of the clocks may lag behind the other if the recording time is too long
- Some questions
  - what is "too long"? (I think several hours or days)
  - is the Free Run option more sensitive to drift?
  - can we resync the cameras' timecodes frequently?
- We could validate how good/bad the timecode drift is with the approach they use in the [Acinoset paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561338):
  - they flashed a rig of LEDs three times, just before and after data collection

## Additional comments

- [Acinoset paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561338) paper may be a good reference on people doing stereo in the field, in a large area
  - code [here](https://github.com/African-Robotics-Unit/AcinoSet#camera-calibration-and-3d-reconstruction) and notebook using opencv [here](https://github.com/African-Robotics-Unit/AcinoSet/blob/main/src/calib_with_gui.ipynb)
- [Texas instruments](https://software-dl.ti.com/jacinto7/esd/robotics-sdk/latest/docs/source/tools/stereo_camera/calibration/README.html) have nice sample images of a good calibration.
- Should we try other patterns too (charuco boards, circles grid)? Would it be more robust?

## Useful references

- [DeepLabCut 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#jump-in-with-direct-deeplabcut-2-camera-support)
- [Anipose](https://anipose.readthedocs.io/en/latest/)
- [Aniposelib](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html)
- [OpenCV docs calib3d](https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91)

- [^1]: from [DeepLabCut 3D](https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html#jump-in-with-direct-deeplabcut-2-camera-support), and [OpenPose](https://github.com/jrkwon/openpose/blob/master/doc/calibration_demo.md#general-quality-tips)

- [^2]: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html

- [^3]: https://github.com/TemugeB/python_stereo_camera_calibrate#procedure
