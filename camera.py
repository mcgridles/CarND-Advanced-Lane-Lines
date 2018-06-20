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
        Undistort an image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
