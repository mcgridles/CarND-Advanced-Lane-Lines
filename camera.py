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
        transform_matrix: (numpy.ndarray) -> perspective warp transformation matrix
    """

    def __init__(self, shape):
        self.nx = 9
        self.ny = 6
        self.mtx = None
        self.dist = None
        self.transform_matrix = None
        self.transform_matrix_inv = None

        # automatically calibrate all camera objects
        self.calibrate()
        self.calculateTransformMatrix(shape)

    def calibrate(self):
        """
        Use calibration images to calculate the camera matrix and distortion
        coefficients
        """
        images = glob.glob('camera_cal/calibration*.jpg')
        img_shape = None
        objpoints = []
        imgpoints = []

        for image_path in images:
            objp = np.zeros((self.nx*self.ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not img_shape:
                img_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    def calculateTransformMatrix(self, shape):
        """
        Calculates the transform matrix used for warping
        """
        # trapezoid points
        bottom_left = [125, shape[0]]
        top_left = [(shape[1]/2)-70, (shape[0]/2)+90]
        top_right = [(shape[1]/2)+75, (shape[0]/2)+90]
        bottom_right = [shape[1]-95, shape[0]]

        src_corners = np.float32([bottom_left, top_left, top_right, bottom_right])
        dst_corners = np.float32([
            [150, shape[0]],
            [150, 0],
            [shape[1]-150, 0],
            [shape[1]-150, shape[0]]
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        self.transform_matrix_inv = cv2.getPerspectiveTransform(dst_corners, src_corners)

    def undistort(self, img):
        """
        Remove distortion from an image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp(self, img):
        """
        Warp the lane portion of an image to a birds-eye POV
        """
        shape = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.transform_matrix, shape)

    def unwarp(self, img):
        """
        Unwarps an image back to the original perspective
        """
        shape = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.transform_matrix_inv, shape)
