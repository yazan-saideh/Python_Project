import os

from PIL import Image as PILImage
from copy import deepcopy
from typing import List, Union
import numpy as np

SingleChannelImage = List[List[int]]
ColoredImage = List[List[List[int]]]
Image = Union[ColoredImage, SingleChannelImage]
Kernel = List[List[float]]

GRAYSCALE_CODE = "L"
RGB_CODE = "RGB"


# if in future colored images are needed, just change this function and add grayscale2RGB function
def load_image(image_filename: str, mode: str = RGB_CODE) -> np.ndarray:
    """
    Loads the image stored at image_filename and returns it as a NumPy array.
    :param image_filename: Path to the image file.
    :param mode: Use GRAYSCALE_CODE = "L" for grayscale images.
    :return: A NumPy array representing the image (2D for grayscale, 3D for RGB).
    """
    if not os.path.exists(image_filename):
        raise FileNotFoundError(f"The file '{image_filename}' was not found.")

    try:
        img = PILImage.open(image_filename).convert(mode)
    except Exception as e:
        raise IOError(f"An error occurred while opening the image: {e}")

    # Convert to NumPy array
    image = np.array(img)
    if mode == GRAYSCALE_CODE:
        return image  # No need to convert if already grayscale
    return RGB2grayscale(image)


def show_image(image: np.ndarray) -> None:
    """
    Displays an image (NumPy array) using PIL.
    """
    pil_image = PILImage.fromarray(image)
    pil_image.show()


def save_image(image: np.ndarray, filename: str) -> None:
    """
    Saves a NumPy array (image) to a file in PNG format.
    """
    pil_image = PILImage.fromarray(image)
    pil_image.save(filename)


def __lists_from_pil_image(image: PILImage) -> Image:
    """
    Converts an Image object to an image represented as lists.
    :param image: a PIL Image object
    :return: the same image represented as multi-dimensional list.
    """
    width, height = image.size
    pixels = list(image.getdata())
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    if type(pixels[0][0]) == tuple:
        for i in range(height):
            for j in range(width):
                pixels[i][j] = list(pixels[i][j])
    return pixels


def __pil_image_from_lists(image_as_lists: Image) -> PILImage:
    """
    Creates an Image object out of an image represented as lists.
    :param image_as_lists: an image represented as multi-dimensional list.
    :return: the same image as a PIL Image object.
    """
    image_as_lists_copy = deepcopy(image_as_lists)
    height = len(image_as_lists_copy)
    width = len(image_as_lists_copy[0])

    if type(image_as_lists_copy[0][0]) == list:
        for i in range(height):
            for j in range(width):
                image_as_lists_copy[i][j] = tuple(image_as_lists_copy[i][j])
        im = PILImage.new(RGB_CODE, (width, height))
    else:
        im = PILImage.new(GRAYSCALE_CODE, (width, height))

    for i in range(width):
        for j in range(height):
            im.putpixel((i, j), image_as_lists_copy[j][i])
    return im


def RGB2grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image (3D NumPy array) to grayscale (2D NumPy array).
    Uses vectorized operations for performance.
    """
    if image.ndim == 2:  # Already grayscale, no conversion needed
        return image

    # RGB to Grayscale conversion using matrix multiplication for efficiency
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], weights)  # Only use RGB channels and apply weights
