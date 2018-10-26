# Advanced Lane Finding Project

In this project, I implement Advanced Lane finding methods as part of the Udacity Self-driving car nano-degree program.

The project implements the following pipeline:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/road_undistorted.jpg "Road Transformed"
[image3]: ./output_images/thresholded.png "Binary Example"
[image4]: ./output_images/warped_example.png "Warp Example"
[image5]: ./output_images/lanes_sliding_window_1.png "Lanes"
[image5.1]: ./output_images/lanes_search_around_poly_1.png "Lanes"
[image5.2]: ./output_images/lanes_search_around_poly_2.png "Lanes 2"
[image5.3]: ./output_images/lanes_search_around_poly_3.png "Lanes 3"
[image6]: ./output_images/highlighted_lane.png "Highlighted Lanes 4"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

This step is implemented in the class `Camera` in the method calibrate(). It takes a path to the folder containing a set of chessboard images, a file name pattern of all the calibration images to use, and the number of grid cells in x and y directions (6 and 9 in this case). The method then prepares "object points", which are the 3-D coordinates of the chessboard corners in relative terms. Here we have 6x9 chessboards, so we define coordinate z=0, x in [0,5] and y in [0,8]. So the "object point" coordinates of the internal corners of each chessboard are [0,0,0], [0,1,0], [0,2,0], ..., [0,8,0] for the first row. Similarly, for second row they are [1,0,0], [1,1,0], ..., [1,8,0], and so on for subsequent rows.

OpenCV's method findChessboardCorners is used to find "image points", i.e. the actual x,y pixel coordinates of the corners for each image. The list of object points and detected image points is then used with cv2.calibrateCamera() method to get the Camera matrix. This is then stored in the Camera class along with the Distribution Coefficients for later use in undistorting images.

### Undistort Image

`cv2.undistort()` function is then used (encapsulated by `Camera.undistort()` method) to correct distortion of images. Following is an example of this correction applied to a chessboard image.

![alt text][image1]

Here is an example of a distortion-corrected image of a road frame:
![alt text][image2]

### Binary Thresholding

Binary thresholding is the first step to separate lane pixels from non-lane pixels. Basically it should convert most pixels identified as a lane line as 1, and most others as 0 (it is not perfect and that's why later steps build on this). This is implemented in class `RoadFrame` in the method `threshold()`. It uses various methods of thresholding images defined in the file thresholds.py, but finally a combination of HLS channel thresholding with X gradient was used to produce the final binary image. Here is an example of a binary road frame:

![alt text][image3]

### Perspective Transformation

First the transformation coordinates are hand chosen in `Perspective.getTransformParameters()` in the file `Perspective.py`. Here are the source and destination points:

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

These are then passed to `Camera.setPerspectiveTransform()`, which creates and stores a perspective matrix using `cv2.getPerspectiveTransform()` using these source and destination points. This matrix is then also used to initialize the class `RoadFrame` and is used in the method `RoadFrame.getLane()` to perform perspective transform on a frame of the road, using `cv2.warpPerspective()`.
Here is an example of a warped image:

![alt text][image4]

### Identify lane-line pixels and fit a polynomial through them

In `RoadFrame.getLane()` after warping the road frame and applying binary threshold, the methods sliding_window() or search_around_poly() are called to identify lane pixels, depending on whether it's the first frame or subsequent. These methods identify lane pixels and instantiate an object of class Lane (contained in `Lanes.py`). Then `Lane.fitPoly()` method is called to fit a polynomial to these identified line pixels.

Here is an example pipeline where lane lines are identified using sliding window:
![alt text][image5]

This example is of a subsequent frame where lane lines are identified using a window around previously identified polynomial:
![alt text][image5.1]

### Calculate Radius of Curvature of the lane and Position of the vehicle with respect to center

The Lane class includes this code in the calculateByFit() method. It calculates the radius of curvature of each lane line and stores it in the class.

### Results

The code to highlight the lane area is implemented in class `RoadFrame.highlightLane()` method.  An example of this can be seen below:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I abstracted out the whole pipeline into an object oriented design that uses the following classes:

1. Camera (Camera.py): Represents the camera object and contains methods such as calibrate(), setPerspectiveTransform(), getNextFrame(), undistort(), etc.
2. RoadFrame (RoadFrame.py): An object of RoadFrame is returned by Camera.getNextFrame(). It encapsulates methods such as getLane(), sliding_window(), search_around_poly(), highlightLane().
3. Lane (Lanes.py): Represents the identified lane. Contains methods like fitPoly(), calculateByCoeffs(), calculateByFit(). RoadFrame.getLane() method returns an object of this class.
4. LaneSmoother (Lanes.py): Utility class to help smooth lanes over several frames.
5. run_adv_lane_lines2.py: Main program that executes the pipeline.

The first major problem I faced was finding the right thresholds to generate a good binary image that identifies the lane lines. This is the most experimental and manually laborious part of the pipeline, and is key to getting the rest of the project right. Here the HLS channel proved useful, alongwith the X gradient. The quality of the polynomial fit to the lane lines can be improved by spending some more time here to identify good parameters. I want to experiment with some deep learning models here just to identify the polynomial fit - I think that has the potential for alleviating a lot of manual experimentation. 

Another trick that I want to use is to project the right lane line onto the left and vice versa to get better lines. This will help in those situations where one line is noisier or more obscure than the other. This is kind of what a human would do if one side is obscured but the other has good visibility.

There is also more opportunity to experiment with lane smoothing. So far I tried averaging over X pixels over multiple frames, and averaging polynomial coefficients over frames. Even then, sometimes the individual lane line polynomials move aggressively from one frame to the next. This type of averaging works well on a highway where the lanes move gradually, but in a city or hilly roads (such as the advanced challenge) this by itself will not be so robust. Another thing to try would be to use weighted averages, based on recency of frames and a score of good fit, so past frames don't sway the lane lines too much.

Another problem area is that the perspective transform parameters are statically and manually selected before the pipeline starts. It doesn't work robustly across different roads or where the camera perspective changes, e.g. if there is a incline or decline. In a real setting, there needs to be a way to initialize the transformation parameters dynamically.
