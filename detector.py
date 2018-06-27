import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from camera import Camera
from line import Line

class Processor:
    """
    Image processor to identify lanes

    Attributes:
        camera: (Camera) -> Camera object for undistorting images
        camera_pos: (float) -> the position of the camera in the image
        color_space: (string) -> the color space the images will start in
        color_convert: (dict{cv2.COLOR_}) -> color conversion flags
        nwindows: (int) -> the number of windows to use for fitting a line
        margin: (int) -> the width from the center of a window in pixels
        minpix: (int) -> the minimum number of pixels that a line requires
        left_line: (Line) -> Line object to store information about the left lane line
        right_line: (Line) -> Line object to store information about the right lane line
    """

    COLOR_CONVERT = {
        'RGB': {'2GRAY': cv2.COLOR_RGB2GRAY, '2HLS': cv2.COLOR_RGB2HLS},
        'BGR': {'2GRAY': cv2.COLOR_BGR2GRAY, '2HLS': cv2.COLOR_BGR2HLS}
    }

    def __init__(self, c_space, shape):
        self.camera = Camera(shape)
        self.camera_pos = shape[1] / 2
        self.color_space = c_space
        self.color_convert = self.COLOR_CONVERT[c_space]

        # sliding window variables
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        self.left_line = Line()
        self.right_line = Line()

    def processImage(self, img):
        # remove distortion and warp for birds-eye view
        undistorted = self.camera.undistort(img)
        gray = cv2.cvtColor(undistorted, self.color_convert['2GRAY'])

        # apply color thresholding
        s_channel = self.sChannelThreshold(undistorted, thresh=(130, 255))

        grad_x = self.absSobelThresh(gray, orient='x', sobel_kernel=9, thresh=(50, 255))
        grad_y = self.absSobelThresh(gray, orient='y', sobel_kernel=13, thresh=(75, 255))
        mag_binary = self.magThresh(gray, sobel_kernel=25, thresh=(40, 255))
        dir_binary = self.dirThresh(gray, sobel_kernel=25, thresh=(0.8, 1.2))
        combined_grad = np.zeros_like(mag_binary)
        combined_grad[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        combined_grad_color = np.zeros_like(combined_grad)
        combined_grad_color[(s_channel == 1) | (combined_grad == 1)] = 1

        warped = self.camera.warp(combined_grad_color)

        plot_y = self.detectLaneLines(warped)
        result = self.drawLane(combined_grad_color, undistorted, plot_y)
        result_with_text = self.displayStats(result)

        return result_with_text

    def sChannelThreshold(self, img, thresh=(0, 255)):
        """
        Expects 3 channel image
        """
        hls = cv2.cvtColor(img, self.color_convert['2HLS'])
        s_channel = hls[:,:,2]

        binary_s = np.zeros_like(s_channel)
        binary_s[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_s

    def rChannelThreshold(self, img, thresh=(0, 255)):
        """
        Expects 3 channel image
        """
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
    def dirThresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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

    def detectLaneLines(self, img):
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
        left_fit = None
        right_fit = None
        self.left_line.detected = False
        self.right_line.detected = False

        if not self.left_line.detected or self.right_line.detected:
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
        else:
            previous_fit_left = self.left_line.best_fit_all[-1]
            left_lane_inds = ((nonzero_x > (previous_fit_left[0]*(nonzero_y**2) + previous_fit_left[1]*nonzero_y +
                             previous_fit_left[2] - self.margin)) & (nonzero_x < (previous_fit_left[0]*(nonzero_y**2) +
                             previous_fit_left[1]*nonzero_y + previous_fit_left[2] + self.margin)))

            previous_fit_right = self.right_line.best_fit_all[-1]
            right_lane_inds = ((nonzero_x > (previous_fit_right[0]*(nonzero_y**2) + previous_fit_right[1]*nonzero_y +
                              previous_fit_right[2] - self.margin)) & (nonzero_x < (previous_fit_right[0]*(nonzero_y**2) +
                              previous_fit_right[1]*nonzero_y + previous_fit_right[2] + self.margin)))

        # Again, extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Generate x and y values for plotting
        plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
        right_fit_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

        # determine if lines have been found
        self.left_line.calculateRadius(plot_y, left_fit_x)
        if self.left_line.detected:
            self.left_line.best_fit_raw.append(left_fit)
            if len(self.left_line.best_fit_raw) > self.left_line.best_fit_window:
                self.left_line.best_fit_raw.pop(0)
            self.left_line.calculateBestFit()

            self.left_line.current_x_fit = self.left_line.best_fit[0]*plot_y**2 + self.left_line.best_fit[1]*plot_y + self.left_line.best_fit[2]

        self.right_line.calculateRadius(plot_y, right_fit_x)
        if self.right_line.detected:
            self.right_line.best_fit_raw.append(right_fit)
            if len(self.right_line.best_fit_raw) > self.right_line.best_fit_window:
                self.right_line.best_fit_raw.pop(0)
            self.right_line.calculateBestFit()

            self.right_line.current_x_fit = self.right_line.best_fit[0]*plot_y**2 + self.right_line.best_fit[1]*plot_y + self.right_line.best_fit[2]

        # apply color mask and plot lines
        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(self.left_line.current_x_fit, plot_y, color='yellow')
        # plt.plot(self.right_line.current_x_fit, plot_y, color='yellow')
        # plt.xlim(0, out_img.shape[1])
        # plt.ylim(out_img.shape[0], 0)
        # plt.show()

        return plot_y

    def drawLane(self, warped, undist, plot_y):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.current_x_fit, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.current_x_fit, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = self.camera.unwarp(color_warp)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
        # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # plt.show()

        return result

    def displayStats(self, img):
        x_scale = 3.7/700
        y_scale = 30/720

        radius = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2
        radius_msg = 'Radius of Curvature = {0}(m)'.format(int(radius))

        lane_center = (self.left_line.current_x_fit[719] + self.right_line.current_x_fit[719]) / 2
        line_offset_pixels = self.camera_pos - lane_center
        line_base_pos = line_offset_pixels * x_scale
        if line_base_pos > 0:
            lane_msg = 'Vehicle is {0}m right of center'.format(round(line_base_pos, 2))
        else:
            lane_msg = 'Vehicle is {0}m left of center'.format(round(abs(line_base_pos), 2))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,radius_msg,(10,75), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,lane_msg,(10,150), font, 2,(255,255,255),2,cv2.LINE_AA)

        return img

def main():
    processor = Processor('BGR', (720,1280))

    clip = VideoFileClip('challenge_video.mp4')
    processed_video = clip.fl_image(processor.processImage)
    processed_video.write_videofile('challenge_processed.mp4', audio=False)

if __name__ == '__main__':
    main()
