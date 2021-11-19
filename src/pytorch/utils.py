import sys

sys.path.append("../")

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import pathlib
from utils.preprocessing import preprocess_image, preprocess_image_segmentation
from scipy.ndimage.measurements import label
import numpy as np
import Augmentor


def threshold(image, thres=0.1):
    return image > thres


def count_components(array):
    structure = np.ones((3, 3), dtype=np.int)
    components, count = label(array, structure)
    return components, count


def get_count(image, thres=0.1):
    return count_components(threshold(image[0], thres))[1]


class SegmentationDataset(Dataset):
    def __init__(self, path_to_dir, transform=None):
        self.path = path_to_dir
        self.image_id_list = list(pathlib.Path(path_to_dir).glob("*.jpg"))
        self.transform = transform

        # self.center_id_list = list(pathlib.Path("../data/").glob("*.jpg"))[0].stem

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.image_id_list[idx].stem
        #print("image: " + name)
        sample = preprocess_image_segmentation(self.path, name)

        if self.transform:
            sample = self.transform(sample[0]), sample[1]
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return input_image, torch.unsqueeze(torch.Tensor(target), 0)
        else:
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return torch.unsqueeze(torch.Tensor(input_image), 0), torch.unsqueeze(
                torch.Tensor(target), 0
            )


class RegressionDataset(Dataset):
    def __init__(self, path_to_dir, transform=None):
        self.path = path_to_dir
        self.image_id_list = list(pathlib.Path(path_to_dir).glob("*.jpg"))
        self.transform = transform

        # self.center_id_list = list(pathlib.Path("../data/").glob("*.jpg"))[0].stem

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.image_id_list[idx].stem
        sample = preprocess_image(self.path, name)

        if self.transform:
            sample = self.transform(sample[0]), sample[1]
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return input_image, torch.unsqueeze(torch.Tensor(target), 0)
        else:
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return torch.unsqueeze(torch.Tensor(input_image), 0), torch.unsqueeze(
                torch.Tensor(target), 0
            )


class ClassificationDataset(Dataset):
    def __init__(self, path_to_good, path_to_bad):
        self.good_image_id_list = list(
            map(str, pathlib.Path(path_to_good).glob("*.jpg"))
        )

        self.bad_image_id_list = list(map(str, pathlib.Path(path_to_bad).glob("*.jpg")))
        self.image_list = self.bad_image_id_list + self.good_image_id_list
        self.label = [1.0 for _ in range(len(self.bad_image_id_list))] + [
            0.0 for _ in range(len(self.good_image_id_list))
        ]

        # self.center_id_list = list(pathlib.Path("../data/").glob("*.jpg"))[0].stem

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.image_list[idx]
        name = path.__str__()
        orig_img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        # resize image
        resized_image = torch.Tensor(cv2.resize(orig_img, (256, 256))) / 255.0
        label = self.label[idx]

        return torch.unsqueeze(resized_image, 0), label


class AugmentorSegmentationDataset(Dataset):
    def __init__(self, path_to_dir):
        self.path = path_to_dir
        self.image_id_list = list(pathlib.Path(path_to_dir).glob("*.jpg"))

        # self.center_id_list = list(pathlib.Path("../data/").glob("*.jpg"))[0].stem

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.image_id_list[idx].__str__()
        image, label = preprocess_image_segmentation(self.path, name)
        p = Augmentor.DataPipeline([[image, label]])
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.skew(probability=0.5)
        # p.crop(probability=0.5)
        p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
        p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
        # p.random_erasing(probability=0.2, rectangle_area=0.2)
        img = p.sample(1)

        return img
