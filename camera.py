import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

class Camera:
    """
    Wrapper for OpenCV camera operations

    Attributes:
        nx: (int) -> number of chessboard columns
        ny: (int) -> number of chessboard rows
        mtx: (numpy.ndarray) -> camera matrix
        dist: (numpy.ndarray) -> distortion coefficients
    """

    def __init__(self):
        self.nx = 9
        self.ny = 6
        self.mtx = None
        self.dist = None

    def calibrate(self):
        """
        Use calibration images to calculate the camera matrix and distortion
        coefficients
        """
        img_shape = None
        images = glob.glob('camera_cal/calibration*.jpg')

        for image_path in images:
            objpoints = []
            imgpoints = []

            objp = np.zeros((self.nx*self.ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not img_shape:
                img_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

    def undistort(self, img):
        """
        Remove distortion from an image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    @staticmethod
    def warp(img):
        """
        Warp the lane portion of an image to a birds-eye POV
        """
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