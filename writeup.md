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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in the file called `camera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][./output_images/test_undistorted.jpg]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][./output_images/undistorted.jpg]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 54 through 57 in `detector.py`).  Here's an example of my output for this step.

![alt text][./output_images/threshold.jpg]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 81 through 86 in the file `camera.py`, and a function called `calculateTransformMatrix()`, which takes care of calculating the transformation matrix during initialization. (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`), while the `calculateTransformMatrix()` function takes an image shape (`shape`). I separated the initialization of the transformation matrix from the warping because it only needs to happen once and the matrix can be used for the rest of the program execution. By doing it this way it does not need to be calculated for every image which speeds things up a bit. I chose the hardcode the source and destination points in the following manner:

```python
bottom_left = [190, shape[0]]
top_left = [(shape[1]/2)-105, (shape[0]/2)+125]
top_right = [(shape[1]/2)+110, (shape[0]/2)+125]
bottom_right = [shape[1]-155, shape[0]]

src_corners = np.float32([bottom_left, top_left, top_right, bottom_right])
dst_corners = np.float32([
    [200, shape[0]],
    [200, 0],
    [shape[1]-200, 0],
    [shape[1]-200, shape[0]]
])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 200, 720      |
| 520, 485      | 200, 0        |
| 740, 485      | 1080, 0       |
| 1125, 720     | 1080, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][./output_images/warped.jpg]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used the sliding window method to identify the lane lines in the image. To start, I produced a histogram of the bottom half of the image and used the positions of the two peaks on either side of the image as a base. Then, I found the nonzero pixels in the x and y directions within the window, and calculated the new position of the next window. This was done for each window all the way up the image, and then the indices of the nonzero points were used to fit a polynomial. Finally, I used the `Line` object to store the coefficients and averaged them over a moving window of 25 frames to smooth out the line.

![alt text][./output_images/polynomials.jpg]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 49 through 57 in my code in `line.py` where I calculated the radius of each line independently using the equation given in the lessons. This is one of the areas that needs work, both because I'm not really sure my x and y scales are correct, and because the radius fluctuates rapidly and does not seem very accurate. Then, in lines 294 and 295 of `detector.py`, I averaged the radius of each line to get the overall radius.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `detector.py` in the function `drawLane()`.  Here is an example of my result on a test image:

![alt text][./output_images/lane.jpg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

I really don't feel like this passes the expectations, but I was really having trouble fitting the lines and no matter what I tried nothing would really seem to change. See below for comments about this.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I had was with the shadows in the image, both on the side of the road and the shadows caused by the bridge in the middle of the video. The left line often would fit to the shadows caused by the barrier instead of the lane line, and it seemed like no amount of thresholding would give me the lane lines but not the shadows.

The shadow under the bridge unsurprisingly caused issues, which I tried to overcome by using smoothing techniques and eliminating the frames from the line calculations. However, I couldn't find a way to adequately eliminate frames due to the erratic radius calculations, and the smoothing always seemed to produce a line that didn't change enough and was stuck in on place, or changed the right amount but was then still affected by the shadow.

I'm really not sure what direction I should be going in from this point, I feel like I understand the material and have been working at it consistently for a week or two now, but I've become stuck. I would definitely appreciate some feedback or a nudge in the right direction, I think I'm actually very close but I've painted myself into a corner and a fresh perspective might help me get to where I need to be.

One way the pipeline could be made more robust is to try to do feature extraction on the lane markings instead of simply using thresholding. For instance, a lane should have a higher value for all its pixels than the road on either side of it. In the case of a large shadow or bright area on the side of the road, the area of higher intensity pixels would most likely just have a single edge and not be a strip of high intensity.

In the case of a shadow covering the lines, this might help, but it might be better to detect that the overall brightness of the image has decreased and adjust the color/edge thresholding values on the fly.