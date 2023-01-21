import torch
import numpy as np
import cv2
from typing import Callable, Optional, Tuple, Union
from PIL import Image
from deeplle.utils import check_path_exists


def read_image(path: str, backend: str = "PIL") -> Image:
    """
    Read image from the specified path and return an image of class PIL.Image.

    The image file should either be saved as image format with extension ['.png', '.jpg', '.jpeg', '.bmp']
    or numpy array with extension '.npy'.
    """
    assert check_path_exists(path), "Image path: {} does not exist!".format(path)
    assert backend in ['PIL', 'cv2'], "backend should be one of 'PIL' | 'cv2'"

    if path.endswith(".npy"):
        image = np.load(path)
        image = Image.fromarray(image)
    else:
        if backend == 'PIL':
            image = Image.open(path)
        elif backend == 'cv2':
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
    
    return image


def save_image(save_path: str, image: torch.Tensor, post_process: Callable = None, input_image: Optional[torch.Tensor] = None):
    """
    Save image to the specified path. If input image is provided, construct image pair
    by concatenateing [input_image, image] horizontally, note these two images must have
    the same height. 

    Args:
        save_path (str): path to save the image.
        image (tensor): image tensor of shape (C, H, W).
        post_process (callable): post processing function to apply on the image.
        input_image (tensor): input image tensor of shape (C, H, W) for comparison with output image.
            If provided, the image pair will be saved. Otherwise, only the output image will be saved.
    
    NOTE: The directory to save must exist and the data range of the image tensor must be
    in [0, 1].
    """
    image = image.cpu().numpy()
    image = np.transpose(image, [1, 2, 0])
    if post_process:
        image = post_process(image)
    image = np.clip(image * 255, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # construct image pair for comparison
    if input_image is not None:
        input_image = input_image.cpu().numpy()
        input_image = np.transpose(input_image, [1, 2, 0])
        input_image = np.clip(input_image * 255, 0, 255)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        image = cv2.hconcat([input_image, image])

    cv2.imwrite(save_path, image)


def convert_to_image(input: torch.Tensor, data_range: Tuple = (0., 1.), input_order="CHW") -> np.ndarray:
    """
    Args:
        input (tensor): input image tensor.
        data_range (list): data range of the input image tensor. Default: (0, 1).
        input_order (str): order of the input image tensor. Default: "CHW".
    
    Returns:
        converted image in numpy array.
    """
    assert isinstance(input, torch.Tensor), "Expect input as Tensor."
    
    assert input_order in ["CHW", "HWC"], "input_order must be chosen from ['CHW', 'HWC']"
    if input_order == "CHW":
        input = torch.einsum("chw->hwc", input)
    
    input = input.detach().cpu().numpy()

    input = (input - data_range[0]) / (data_range[1] - data_range[0])
    input = np.clip(input * 255, 0, 255).astype(np.uint8)
    return input


def calculate_brightness(image: Union[Image.Image, np.ndarray]) -> float:
    """
    Calculate the brightness of the input image. The image should be either PIL.Image or numpy.ndarray.

    Args:
        image (np.ndarray or PIL.Image): input image.
    
    Returns:
        float: brightness in range [0, 1].
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale