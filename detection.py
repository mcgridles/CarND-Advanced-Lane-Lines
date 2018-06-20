import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#imageio.plugins.ffmpeg.download()

from camera import Camera
from line import Line

class Processor:
    """
    Image processor to identify lanes

    Attributes:
        camera: (Camera) -> Camera object for undistorting images
    """

    COLOR_CONVERT = {
        'RGB': {'2GRAY': cv2.COLOR_RGB2GRAY, '2HLS': cv2.COLOR_RGB2HLS},
        'BGR': {'2GRAY': cv2.COLOR_BGR2GRAY, '2HLS': cv2.COLOR_BGR2HLS}
    }

    def __init__(self, c_space):
        self.camera = Camera()
        self.camera.calibrate()
        self.color_space = c_space
        self.color_convert = self.COLOR_CONVERT[c_space]

        # sliding window variables
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        self.frames_since_success = 0

    def processImage(self, img):
        # remove distortion and
        undistorted = self.camera.undistort(img)

        # apply color thresholding
        s_channel = self.sChannelThreshold(undistorted, thresh=(90, 255))
        r_channel = self.rChannelThreshold(undistorted, thresh=(90, 255))
        combined_channel = np.zeros_like(r_channel)
        combined_channel[(s_channel == 1) | (r_channel == 1)] = 1

        # warp for birds-eye view
        warped = self.camera.warp(combined_channel)

        # apply gradient thresholding
        grad_x = absSobelThresh(warped, orient='x', sobel_kernel=3, thresh=(20, 100))
        grad_y = absSobelThresh(warped, orient='y', sobel_kernel=3, thresh=(20, 100))
        mag_binary = magThresh(warped, sobel_kernel=9, thresh=(30, 100))
        dir_binary = dirThreshold(warped, sobel_kernel=3, thresh=(0.7, 1.3))
        combined_grad = np.zeros_like(dir_binary)
        combined_grad[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        self.slidingWindows(combined_grad)

    def sChannelThreshold(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, self.color_convert['2HLS'])
        s_channel = hls[:,:,2]

        binary_s = np.zeros_like(s_channel)
        binary_s[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_s

    def rChannelThreshold(self, img, thresh=(0, 255)):
        if self.color_space == 'RGB':
            r_channel = img[:,:,0]
        elif self.color_space == 'BGR':
            r_channel = img[:,:,2]

        binary_r = np.zeros_like(r_channel)
        binary_r[(r_channel > thresh[0]) & (r_channel <= thresh[1])] = 1
        return binary_r

    @staticmethod
    def absSobelThresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
        Expects binary image
        """
        if orient == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            print('Invalid option: {}'.format(orient))

        abs_sobel = np.abs(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thres[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    @staticmethod
    def magThresh(img, sobel_kernel=3, thresh=(0, 255)):
        """
        Expects binary image
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        mag_sobel = np.sqrt(sobel_x**2 + sobel_y**2)

        scale_factor = np.max(mag_sobel)/255
        mag_sobel = (mag_sobel/scale_factor).astype(np.uint8)
        mag_binary = np.zeros_like(mag_sobel)
        mag_binary[(mag_sobel >= thresh[0]) & (mag_sobel <= thresh[1])] = 1
        return mag_binary

    @staticmethod
    def dirThreshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Expects binary image
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel_x = np.abs(sobel_x)
        abs_sobel_y = np.abs(sobel_y)

        grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        return dir_binary

    def slidingWindows(self, img):
        """
        Expects binary warped image
        """
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        out_img = np.dstack((img, img, img))*255

        midpoint = np.int(histogram.shape[0]//2)
        left_x_base = np.argmax(histogram[:midpoint])
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int(img.shape[0]//self.nwindows)

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if > minpix pixels found, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # create lines

def main():
    processor = Processor('BGR')

    #clip = VideoFileClip('challenge_video.mp4')
    #processed_video = clip.fl_image(processor.processImage)
    #%time processed_video.write_videofile('challenge_processed.mp4', audio=False)

if __name__ == '__main__':
    main()
