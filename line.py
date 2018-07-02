import numpy as np

class Line():
    """
    Stores characteristics of each line detection

    Attributes:
        detected: (boolean) -> was the line detected in the last iteration?
        best_fit_window: (int) -> the number of frames to average for the best fit
        best_fit_raw: (list[float]) -> the coefficients from the last n frames
        best_fit: (numpy.ndarray) -> polynomial coefficients averaged over the last n iterations
        current_x_fit: (numpy.ndarray) -> current polyline coordinates
        radius_of_curvature: (float) -> radius of curvature of the line in meters
        x_scale: (float) -> the scale in pixels/meter in the x direction
        y_scale: (float) -> the scale in pixels/meter in the y direction
    """

    def __init__(self):
        self.best_fit_window = 3
        self.previous_frames = []
        self.best_fit = None
        self.current_x_fit = None

        self.x_scale = 3.7/980
        self.y_scale = 30/720

        self.radius_of_curvature = None

    def calculateBestFit(self):
        self.best_fit = np.mean(self.previous_frames, axis=0)

    def calculateRadius(self, plot_y, x_fit):
        """
        Fit new polynomials to x,y in world space and calculate the new radii of curvature
        """
        y_eval = np.max(plot_y)
        fit_corrected = np.polyfit(plot_y * self.y_scale, x_fit * self.x_scale, 2)
        a = (2 * fit_corrected[0] * y_eval * self.y_scale + fit_corrected[1])

        self.radius_of_curvature = ((1 + a**2)**1.5) / np.absolute(2 * fit_corrected[0])
