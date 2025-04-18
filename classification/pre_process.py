import numpy as np
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomContrast(object):

    def __init__(self, contrast, scale):
        self.contrast = contrast
        self.scale = scale

    def __call__(self, img):
        if self.contrast:
            img = transforms.ColorJitter(contrast=self.scale)(img)
        return img


class RandomBright(object):

    def __init__(self, bright, scale):
        self.bright = bright
        self.scale = scale

    def __call__(self, img):
        if self.bright:
            img = transforms.ColorJitter(brightness=self.scale)(img)
        return img

class land_transform(object):
    def __init__(self, img_size, flip_reflect):
        self.img_size = img_size
        self.flip_reflect = flip_reflect.astype(int) - 1

    def __call__(self, land, flip, offset_x, offset_y):
        land[0:len(land):2] = land[0:len(land):2] - offset_x
        land[1:len(land):2] = land[1:len(land):2] - offset_y
        # change the landmark orders when flipping
        if flip:
            land[0:len(land):2] = self.img_size - 1 - land[0:len(land):2]
            land[0:len(land):2] = land[0:len(land):2][self.flip_reflect]
            land[1:len(land):2] = land[1:len(land):2][self.flip_reflect]

        return land


class image_train(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y, contrast, scale_c, bright, scale_b):
        normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                         IMAGENET_DEFAULT_STD)
        transform = transforms.Compose([
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            RandomContrast(contrast, scale_c),
            RandomBright(bright, scale_b),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def image_test(crop_size=176):
    normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                     IMAGENET_DEFAULT_STD)

    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])