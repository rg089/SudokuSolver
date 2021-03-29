import cv2
import numpy as np
import imutils
from skimage.transform import rotate, AffineTransform, warp


class DigitAugmenter:
    def __init__(self, transforms=None, rescale=True):
        if transforms is None:
            transforms = {"rotation": [-15, 15], "horizontal_shift": [-0.2, 0.2], "vertical_shift": [-0.2, 0.2]}
        self.transforms = transforms
        self.rescale = rescale
        self.transform_keys = list(self.transforms)

    def deskew_centralize(self, img):
        img = imutils.deskew_digit(img, applyThreshold=False)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3, 3), iterations=2)
        img = imutils.centralize_digit(img, (28, 28))
        return img

    def random_transform(self, img):
        transformations = np.random.choice(self.transform_keys, np.random.randint(1, len(self.transform_keys)), replace=False)
        temp_img = img.copy()
        h, w = img.shape[:2]
        for transform in transformations:
            lower_lim, upper_lim = self.transforms[transform]
            if transform == "rotation":
                # print("Applied Rotation")
                deg = np.random.randint(lower_lim, upper_lim+1)
                temp_img = rotate(temp_img, deg)
                # imutils.showImage(temp_img)
            elif transform == "horizontal_shift":
                # print("Applied Horizontal Shifting")
                pix_shift = w*np.random.uniform(lower_lim, upper_lim)
                h_shift = AffineTransform(translation=(pix_shift, 0))
                temp_img = warp(temp_img, h_shift, mode="wrap")
                # imutils.showImage(temp_img)
            elif transform == "vertical_shift":
                # print("Applied Vertical Shifting")
                pix_shift = h*np.random.uniform(lower_lim, upper_lim)
                v_shift = AffineTransform(translation=(0, pix_shift))
                temp_img = warp(temp_img, v_shift, mode="wrap")
                # imutils.showImage(temp_img)
        if not self.rescale:
            temp_img = (temp_img * 255.0).astype("uint8")
        return temp_img

    def generate_batch(self, img, batch_size, process=True):
        imgs = []
        if process:
            img = self.deskew_centralize(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        first_img = img
        if self.rescale:
            first_img = first_img * 1./255
        imgs.append(first_img)
        for i in range(batch_size-1):
            # print(f"Image {i+1}")
            imgs.append(self.random_transform(img))
        return np.array(imgs)


