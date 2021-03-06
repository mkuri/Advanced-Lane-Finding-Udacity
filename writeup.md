## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[undist]: ./figures/undistorted/calibration1_undistorted.jpg "Undistorted"
[undist_test]: ./figures/undistorted/straight_lines1_undistorted.jpg "Undistorted test"
[binary]: ./figures/binary_images/test1_binarized.jpg "Binarized"
[warped]: ./figures/warped_images/test1_warped.jpg "Warped"
[lane]: ./output_images/window/test2_window.jpg "Found lanes"
[pipelined]: ./figures/pipelined/pipelined.png "Pipelined"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code fot this step is implemented as the calibrate_camera funciton in find_lane_advanced.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted][undist]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted test][undist_test]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. This step is implemented as the generate_binary function in find_lane_advanced.py  Here's an example of my output for this step.

Perform binaryization under the following OR conditions.

* X-gradient with a grayscaled image
* X-gradient with a S channel image

![Binarized][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transfrom is implemented as the get_transformation_matrix function and cv2.warpPerspective. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped][warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I found the initial position of the left and right lanes from the histogram of the binary image in the x direction. And I extracted the binary coordinates corresponding to the lane using the 200 x 80 window. Finally I found a quadratic polynomial approximated from the coordinate information of lanes contained in all the windows.

The code for identifying lane-line pixels and fit their positions with a polynomial is implemented as the window function in find_lane_advanced.py.

The following image is the result.

![alt text][lane]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


I calculated the radius of curvature of the lane using the coefficients of the quadratic polynomial.

The code for calculating th radius of curvature of the lane is implemented as the measure_the_curvature function in find_lane_advanced.py.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for calculating th radius of curvature of the lane is implemented as the draw_road_area function in find_lane_advanced.py.

Here is an example of my result on a test image:

![Pipelined][pipelined]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_lane_found.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The process of combining filters for recognizing white lines was difficult. I combined x direction edge detection of gray scale image and x direction edge detection of S channel of HLS image. However, a binary image including many edges other than white lines is generated. It causes wrong detection when it is a construction trace such as in a challenge movie or when a pedestrian crossing is on the road surface. Robustness is improved by removing edges that is not a white line using information such as the distance between white lines.
