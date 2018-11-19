import numpy as np

def augmentation(x):
    """
        Method used for augmenting the instances of the dataset
    """

    pad_size = 4

    h, w, c = 32, 32, 3
    pad_h = h + 2 * pad_size
    pad_w = w + 2 * pad_size

    pad_img = np.zeros((pad_h, pad_w, c))
    pad_img[pad_size:h+pad_size, pad_size:w+pad_size, :] = x

    # Randomly crop and horizontal flip the image
    top = np.random.randint(0, pad_h - h + 1)
    left = np.random.randint(0, pad_w - w + 1)
    bottom = top + h
    right = left + w
    if np.random.randint(0, 2):
        pad_img = pad_img[:, ::-1, :]

    aug_data = pad_img[top:bottom, left:right, :]

    return aug_data



