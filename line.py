import numpy as np

class Line():
    """
    Stores characteristics of each line detection

    Attributes:
        detected: (boolean) -> was the line detected in the last iteration?
        recent_xfitted: (list[float]) -> x values of the last n fits of the line
        bestx: (float) -> average x values of the fitted line over the last n
            iterations
        best_fit: (numpy.ndarray) -> polynomial coefficients averaged over the last n
            iterations
        current_fit: (numpy.ndarray) -> polynomial coefficients for the most recent fit
        radius_of_curvature: (float) -> radius of curvature of the line in some
            units
        line_base_pos: (float) -> distance in meters of vehicle center from the
            line
        diffs: (numpy.ndarray) -> difference in fit coefficients between last
            and new fits
        allx: (list[int]) -> x values for detected line pixels
        ally: (list[int]) -> y values for detected line pixels
    """

    def __init__(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.allx = None
        self.ally = None
