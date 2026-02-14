import numpy as np

class AngleEncoder:
    """
    Maps numerical features to [0, π] for quantum rotation encoding.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def encode(self, x):
        # Clip to avoid out-of-range values
        x = max(self.min_val, min(x, self.max_val))

        # Normalize into [0, π]
        return (x - self.min_val) / (self.max_val - self.min_val) * np.pi
