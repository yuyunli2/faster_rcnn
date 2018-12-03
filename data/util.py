import numpy as np
import random
from PIL import Image


# Read image from a file
def read_image(path, dtype=np.float32, color=True):
    # Open the file
    f = Image.open(path)

    try:
        if(color):
            image = f.convert('RGB')
        else:
            image = f.convert('P')
        # Transform the image into an Numpy array
        image = np.asarray(image, dtype=dtype)
    finally:
        # Close the file
        if hasattr(f, 'close'):
            f.close()

    # Reshape the image array
    # (H, W, C) --> (C, H, W)
    if(image.ndim != 2):
        return image.transpose((2, 0, 1))
    # Expand a dimension, i.e. (H, W) --> (1, H, W)
    else:
        return image[np.newaxis]


# Random flip an image
def random_flip(img, y_random=False, x_random=False, return_param=False):
    y_flip, x_flip = False, False

    if(y_random):
        y_flip = random.choice([True, False])
    if(x_random):
        x_flip = random.choice([True, False])

    if(y_flip):
        img = img[:, ::-1, :]
    if(x_flip):
        img = img[:, :, ::-1]

    if(return_param):
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


# Resize a bounding box
def resize_bbox(bbox, in_size, out_size):
    # bbox's dimension is [num_of_bounding_boxes, 4]
    bbox = bbox.copy()

    # Compute the scales of x and y dimensions
    scale_x = float(out_size[1])/in_size[1]
    scale_y = float(out_size[0])/in_size[0]

    # Rescale the four attributes of a bounding box
    bbox[:, 1] = scale_x * bbox[:, 1]
    bbox[:, 3] = scale_x * bbox[:, 3]
    bbox[:, 0] = scale_y * bbox[:, 0]
    bbox[:, 2] = scale_y * bbox[:, 2]

    return bbox


# Flip a bounding box
def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    bbox = bbox.copy()
    H, W = size

    # Flip horizontally
    if(x_flip):
        # x_max = W - old_x_min
        # x_min = W - old_x_max
        # Assign them into the bounding box respectively
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    # Flip vertically
    if(y_flip):
        # y_max = H - old_y_min
        # y_min = H - old_y_max
        # Assign them into the bounding box respectively
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max

    return bbox


# Sub-function to be used by crop_bbox
def _slice_to_bounds(slice_):
    if(slice_ is None):
        return 0, np.inf

    if(slice_.start is None):
        lower_bound = 0
    else:
        lower_bound = slice_.start

    if(slice_.stop is None):
        upper_bound = np.inf
    else:
        upper_bound = slice_.stop

    return lower_bound, upper_bound


# Truncate a bounding box to fit with the cropped image
def crop_bbox(bbox, y_slice=None, x_slice=None, outside_center_allowed=True, return_param=False):
    # Initialization
    # top = y_min, bottom = y_max, left = x_min, right = x_right
    bbox = bbox.copy()
    top, bottom = _slice_to_bounds(y_slice)
    left, right = _slice_to_bounds(x_slice)
    crop_bb = np.array((top, left, bottom, right))

    if(outside_center_allowed):
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:])/2.0
        mask = np.logical_and(crop_bb[:2]<=center, center<crop_bb[2:]).all(axis=1)

    # Adjust top-left and bottom-right vertex's coordinates
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2]<bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if(return_param):
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox

