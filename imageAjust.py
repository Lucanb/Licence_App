import numpy as np
from PIL import Image

class CustomPixelManipulation(object):
    def __init__(self, add_value=50):
        self.add_value = add_value

    def __call__(self, img):
        """
        Apply the transformation: increase brightness by adding a constant to all pixels.
        """
        np_img = np.array(img)
        np_img = np.clip(np_img + self.add_value, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img) 