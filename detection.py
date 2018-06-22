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
        r_channel = self.rChannelThreshold(undistorted, thresh=(110, 220))
        combined_color = np.zeros_like(r_channel)
        combined_color[(s_channel == 1) | (r_channel == 1)] = 1
        test = np.dstack((s_channel, s_channel, s_channel))*255

        # warp for birds-eye view
        warped = self.camera.warp(combined_color)

        # apply gradient thresholding
        # grad_x = self.absSobelThresh(warped, orient='x', sobel_kernel=9, thresh=(5, 100))
        # grad_y = self.absSobelThresh(warped, orient='y', sobel_kernel=9, thresh=(20, 100))
        # mag_binary = self.magThresh(warped, sobel_kernel=9, thresh=(30, 100))
        # dir_binary = self.dirThreshold(warped, sobel_kernel=3, thresh=(0.7, 1.3))
        # combined_grad = np.zeros_like(dir_binary)
        # combined_grad[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        self.slidingWindows(warped)

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
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
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
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_x_current = left_x_base
        right_x_current = right_x_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_x_left_low = left_x_current - self.margin
            win_x_left_high = left_x_current + self.margin
            win_x_right_low = right_x_current - self.margin
            win_x_right_high = right_x_current + self.margin

            # draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high), (0,255,0), 2)

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_left_low) &  (nonzero_x < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_right_low) &  (nonzero_x < win_x_right_high)).nonzero()[0]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if > minpix pixels found, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        # create lines

def main():
    processor = Processor('BGR')
    img = cv2.imread('test_images/test2.jpg')
    processor.processImage(img)

    #clip = VideoFileClip('challenge_video.mp4')
    #processed_video = clip.fl_image(processor.processImage)
    #%time processed_video.write_videofile('challenge_processed.mp4', audio=False)

if __name__ == '__main__':
    main()
