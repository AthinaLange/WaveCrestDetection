"""
Wave Crests Time Stack Dataset and Dataset Loader

MIT License
(C) 2021 Athina Lange
"""

import os
import glob
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WaveCrestsTimeStackDataset(Dataset):
    """ Wave Crests Time Stack Dataset Class

    Loads samples from images that are located in
    a user specified images directory and ground-truth directory.
    Images consist of wave time stack images in the images directory and
    their corresponding crest ground-truth annotation images in the ground-truth directory.
    The filenames of the images are used as image ids.
    The filenames of the ground-truth images need to be the same as the corresponding images,
    but can have an optional user specified suffix.
    A user defined number of samples are extracted from each image
    by cropping regions of a user defined size at random locations
    in the region of the image where crest ground-truth annotation data is available.

    Attributes:
        images_dir (str): Images directory.
        ground_truth_dir (str): Ground-truth directory.
        ground_truth_suffix (str): Optional ground-truth filename suffix.
        image_ids (list of str): Image ids.
        samples_per_image (int): Number of samples per image.
        sample_width (int): Sample size in x.
        sample_height (int): Sample size in y.
    """

    def __init__(self,
                 images_dir="data/train/images",
                 ground_truth_dir="data/train/ground_truth",
                 ground_truth_suffix="",
                 samples_per_image=100,
                 sample_width=100,
                 sample_height=100):
        """ Constructs a Wave Crests Time Stack Dataset Object.

        Stores the parameters and creates a list of image ids
        from the filenames discovered in the images directory.

        Parameters:
            images_dir (str): Images directory.
            ground_truth_dir (str): Ground-truth directory.
            ground_truth_suffix (str): Optional ground-truth filename suffix.
            samples_per_image (int): Number of samples per image.
            sample_width (int): Sample size in x.
            sample_height (int): Sample size in y.
        """

        # Seed random number generator to ensure repeatable behavior
        torch.manual_seed(42)

        # Store parameters
        self.images_dir = images_dir
        self.ground_truth_dir = ground_truth_dir
        self.ground_truth_suffix = ground_truth_suffix
        self.samples_per_image = samples_per_image
        self.sample_width = sample_width
        self.sample_height = sample_height

        # Creates a list of image ids from the filenames discovered in the images directory
        self.image_ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir)
                    if not file.startswith('.')]
        logging.info(f"Created dataset with {len(self.image_ids)} wave time stack images "
                     f"and corresponding crest ground-truth annotation images "
                     f"with the following ids: \n"
                     f"{self.image_ids}")

    def __len__(self):
        """ Provides the number of samples in the dataset.

        The number of samples is defined by the number of images
        times the number of samples per image.

        Return (int): Number of samples in the dataset.
        """
        return len(self.image_ids * self.samples_per_image)

    def __getitem__(self, idx):
        """ Provides the idx(th) sample from the dataset,
        consisting of a region in a wave time stack image,
        its corresponding region in a crest ground-truth annotation image
        and its meta data.

        Parameters:
            idx (int): Sample index in the dataset.

        Return (dict): "image" (tensor): region in a wave time stack image,
                       "ground-truth" (tensor): corresponding region in a crest ground-truth annotation image and
                       "meta" (dict): meta data with
                            "sample idx" (int): sample index,
                            "image id" (str): image id,
                            "x start (int): start x coordinate of region in image
                            "x end (int): end x coordinate,
                            "y start (int): start y coordinate and
                            "y end (int): end y coordinate.
        """

        # Identify files
        # --------------
        image_id = self.image_ids[idx // self.samples_per_image]

        os.path.join(self.images_dir, image_id)
        image_fns = glob.glob(os.path.join(self.images_dir, image_id + '.*'))
        ground_truth_fns = glob.glob(os.path.join(self.ground_truth_dir, image_id + self.ground_truth_suffix + '.*'))
        # Check files
        assert len(image_fns) == 1, \
            f"{len(image_fns)} images were found for ID {image_id}"
        assert len(ground_truth_fns) == 1, \
            f"{len(ground_truth_fns)} ground-truth images were found for ID {image_id}"

        # Read images
        # -----------
        image = cv2.imread(image_fns[0])
        # Pytorch color images are RGB, convert Opencv format BRG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ground_truth = cv2.imread(ground_truth_fns[0], cv2.IMREAD_GRAYSCALE)
        # Invert ground-truth image so that annotations are 1 and background 0
        ground_truth = ground_truth < 128

        # Transformations and Augmentations
        # =================================
        # Check out: https://pytorch.org/vision/stable/transforms.html

        # Random Crop
        # -----------

        # Determine the region in the ground-truth image where there are crest annotations
        x_min = 0
        x_max = image.shape[1]
        # Annotations are assumed to be 1 and background 0
        y_min = np.min(np.nonzero(ground_truth.sum(axis=1)))
        y_max = np.max(np.nonzero(ground_truth.sum(axis=1)))
        assert (y_min - y_max) < self.sample_height, \
            f"Crest ground truth annotations are from {y_min} to {y_max} in y," \
            f"too small for a sample height of {self.sample_height}"

        # Random position of the sample in that region
        rand_tensor = torch.rand(2)
        y = y_min + int(rand_tensor[1].item() * (y_max - y_min - self.sample_height))
        x = x_min + int(rand_tensor[0].item() * (x_max - x_min - self.sample_width))

        # Crop
        image = image[y:y+self.sample_height, x:x+self.sample_width, :]
        ground_truth = ground_truth[y:y+self.sample_height, x:x+self.sample_width]

        # pytorch image format is [C(hannel), H(eight), W(idth)] Float [0.0, 1.0)
        # -----------------------------------------------------------------------

        # Add C to the HW gray scale ground-truth image
        ground_truth = ground_truth[..., np.newaxis]

        # Transpose HWC of image to CHW for pytorch
        # Normalize 8-bit image data to [0.0, 1.0)
        tfs = transforms.Compose([
            transforms.ToTensor()
        ])
        image = tfs(image)
        ground_truth = tfs(ground_truth)

        # Meta data
        # ---------
        meta = {
            "sample idx": idx,
            "image id": image_id,
            "x start": x,
            "x end": x+self.sample_width,
            "y start": y,
            "y end": y+self.sample_height,
        }

        return {
            "image": image,
            "ground-truth": ground_truth,
            "meta": meta
        }
