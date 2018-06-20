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
        ksize: (int) -> kernel size for thresholding operations
        camera: (Camera) -> Camera object for undistorting images
    """

    def __init__(self):
        self.ksize = 3

        self.camera = Camera()
        self.camera.calibrate()

    def processImage(self, img):
        undistorted = self.camera.undistort(img)
        unwarped = self.unwarp(undistorted)

        # Apply each of the thresholding functions
        grad_x = absSobelThresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
        grad_y = absSobelThresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
        mag_binary = magThresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
        dir_binary = dirThreshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

        combined = np.zeros_like(dir_binary)
        combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        # calculate histogram
        # calculate lines

    def unwarp(self, img):
        # trapezoid points
        bottom_left = [200, img.shape[0]]
        top_left = [(img.shape[1]/2)-30, (img.shape[0]/2)+80]
        top_right = [(img.shape[1]/2)+30, (img.shape[0]/2)+80]
        bottom_right = [img.shape[1]-165, img.shape[0]]
        src_corners = [bottom_left, top_left, top_right, bottom_right]
        dst_corners = [
            [325, img.shape[0]],
            [325, 0],
            [img.shape[1]-325, 0],
            [img.shape[1]-325, img.shape[0]]
        ]
        M = cv2.getPerspectiveTransorm(src_corners, dst_corners)
        return cv2.warpPerspective(img, M, img.shape[::-1])

    @staticmethod
    def absSobelThresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            print('Invalid option: {}'.format(orient))

        abs_sobel = np.abs(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thres[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    @staticmethod
    def magThresh(image, sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        mag_sobelx = np.sqrt(sobelx**2)
        mag_sobely = np.sqrt(sobely**2)
        mag_sobel = np.sqrt(sobelx**2 + sobely**2)

        scale_factor = np.max(mag_sobel)/255
        mag_sobel = (mag_sobel/scale_factor).astype(np.uint8)
        mag_binary = np.zeros_like(mag_sobel)
        mag_binary[(mag_sobel >= thresh[0]) & (mag_sobel <= thresh[1])] = 1
        return mag_binary

    @staticmethod
    def dirThreshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)

        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        return dir_binary

def main():
    processor = Processor()

    #clip = VideoFileClip('challenge_video.mp4')
    #processed_video = clip.fl_image(processor.processImage)
    #%time processed_video.write_videofile('challenge_processed.mp4', audio=False)

if __name__ == '__main__':
    main()
