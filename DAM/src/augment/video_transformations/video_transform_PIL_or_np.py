import numbers
import random
import numpy as np
import PIL
import skimage
import skimage.transform
import torchvision

from . import functional as F
from torchvision import transforms


class ColorDistortion(object):
    def __init__(self, s=1.0):
        self.s = s
        self.color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)

    def __call__(self, video):
        color_distort = transforms.Compose([self.rnd_color_jitter, self.rnd_gray])
        return color_distort(video)


class Compose(object):
    """Composes several transforms
    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        corner_imgs = list()
        for n in self.crop_positions:
            #print(n)
            if n == 'c':
                th, tw = (self.size, self.size)
                x1 = int(round((w- tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif n == 'tl':
                x1 = 0
                y1 = 0
                x2 = self.size
                y2 = self.size
            elif n == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = self.size
            elif n == 'bl':
                x1 = 0
                y1 = h - self.size
                x2 = self.size
                y2 = h
            elif n == 'br':
                x1 = w - self.size
                y1 = h - self.size
                x2 = w
                y2 = h
            corner_imgs.append(imgs[:, y1:y2, x1:x2, :])
        return corner_imgs

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return F.normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




def transform_data(data, scale_size=256, crop_size=224, random_crop=False, random_flip=False):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    if random_crop:
        x0 = random.randint(0, width - crop_size)
        y0 = random.randint(0, height - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    else:
        x0 = int((width-crop_size)/2)
        y0 = int((height-crop_size)/2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if random_flip and random.randint(0,1) == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return  data


def get_10_crop(data, scale_size=256, crop_size=224):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    top_left = [[0, 0],
                [width-crop_size, 0],
                [int((width-crop_size)/2), int((height-crop_size)/2)],
                [0, height-crop_size],
                [width-crop_size, height-crop_size]]
    crop_data = []
    for point in top_left:
        non_flip = []
        flip = []
        x_0 = point[0]
        y_0 = point[1]
        x_1 = x_0 + crop_size
        y_1 = y_0 + crop_size
        for img in data:
            tmp = img.crop((x_0, y_0, x_1, y_1))
            non_flip.append(tmp)
            flip.append(ImageOps.mirror(tmp))
        crop_data.append(non_flip)
        crop_data.append(flip)
    return  crop_data


def scale(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    if width >= height:
        h = scale_size
        w = round((width/height)*scale_size)
    else:
        w = scale_size
        h = round((height/width)*scale_size)
    for i, image in enumerate(data):
        data[i] = image.resize((w, h))
    return  data


def resize(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    for i, image in enumerate(data):
        data[i] = image.resize((scale_size, scale_size))
    return  data


def video_frames_resize(data, scale_size):
    t, h, w, c = data.shape

    if h >= scale_size and w >= scale_size:
        return data
    else:
        data2 = data.copy()
        data2.resize((t, scale_size, scale_size, c))
        return data2

