# Advanced Lane Finding Project

This repository contains code for Advanced Lane Finding project for the Udacity Self Driving Car Nanodegree program.

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

[image1]: ./output/undistorted.png "Undistorted"
[image2]: ./output/road_undistorted.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This step is implemented in the class Camera by the method calibrate(). It takes a path to the folder containing chessboard images, a file name pattern of all the calibration images to use, and the number of grid cells in x and y directions (6 and 9 in this case). The method then prepares "object points", which are the 3-D coordinates of the chessboard corners in relative terms. Here we have 6x9 chessboards, so we define coordinate z=0, x in [0,5] and y in [0,8]. So the "object point" coordinates of the internal corners of each chessboard are [0,0,0], [0,1,0], [0,2,0], ..., [0,8,0] for the first row. Similarly, for second row they are [1,0,0], [1,1,0], ..., [1,8,0], and so on for subsequent rows.

OpenCV's method findChessboardCorners is used to find "image points", i.e. the actual x,y pixel coordinates of the corners for each image. The list of object points and detected image points is then used with cv2.calibrateCamera() method to get the Camera matrix. This is then stored in the Camera class along with the Distribution Coefficients for later use in undistorting images.

`cv2.undistort()` function is used (encapsulated by Camera.undistort() method) to correct distortion of images. Following is an example of this correction applied to a chessboard image.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of a distortion-corrected image of a road frame:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Binary thresholding is done in class RoadFrame in the method threshold(). It uses various methods of thresholding images defined in the file thresholds.py, but finally a combination of HLS channel thresholding with X gradient was used to produce the final binary image. Here is an example of a binary road frame:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The class RoadFrame contains the method getLane() that performs perspective transform on a frame of the road, using cv2.warpPerspective(). The coordinates are hand chosen in the method getTransformParameters() in the file identify_lane_lines.py. Here are the source and destination points:

```python
src = np.float32([[520,  500],
                  [200,  720],
                  [1100, 720],
                  [765,  500]])
                  
dst = np.float32([[300,  450],
                  [300,  720],
                  [1000, 720],
                  [1000, 450]]) 
```

Here is an example of a warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code to identify lane pixels is contained in the class RoadFrame in the method sliding_window() and search_around_poly(). The code to fit a polynomial to these pixels is contained in class Lane in Lanes.py. First the method RoadFrame.getLane() is called. It determines if a previous left or right fit is passed to it. If not, then it uses the sliding_window() method, otherwise it uses search_around_poly(). Both of these methods, after identifying left and right pixels, instantiate a Lane class and use Lane.fitPoly() method to fit a polynomial to them.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The Lane class includes this code in the calculateByFit() method. It calculates the radius of curvature of each lane line and stores it in the class.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in class RoadFrame.highlightLane() method.  An example of this can be seen below:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
