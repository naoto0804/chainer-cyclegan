"""write_image function."""
from PIL import Image
import numpy as np


def write_image(image, path):
    """Save an image to a file.

    Args:
        image (~numpy.ndarray): An image to be saved.
        path (str): The path of an image file.
    """
    if image.shape[0] == 1:
        image = image[0]
    else:
        image = image.transpose((1, 2, 0))

    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)
