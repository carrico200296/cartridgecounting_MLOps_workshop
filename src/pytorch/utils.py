import sys

sys.path.append("../")

import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import pathlib
from data.preprocessing import preprocess_image, preprocess_image_segmentation
from scipy.ndimage.measurements import label
import numpy as np
import Augmentor
import time
from pytorch.model import UNet


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
            return torch.unsqueeze(torch.Tensor(input_image), 0), torch.unsqueeze(torch.Tensor(target), 0)

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
        name = self.image_id_list[idx].stem
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
        label = p.sample(1)

        return img, label

class ValSegmentationDataset(Dataset):
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
            return input_image, torch.unsqueeze(torch.Tensor(target), 0), name
        else:
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return torch.unsqueeze(torch.Tensor(input_image), 0), torch.unsqueeze(torch.Tensor(target), 0), name


class CartridgeCounting:
    """
    CartridgeCounting class: used to make the prediction for an image.
    """
    def __init__(self, model_name):
        # load model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(input_filters=1, filters=64, N=2).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_name, map_location=self.device))
        print(":: U-Net model loaded")
        print(":: Ready to count cartridges")

    def find_center(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy

    def draw_predicted_layer(self, img, num_cartridges, contours):

        centers = [self.find_center(contour) for contour in contours]
        prediction_layer = img.copy()
        highlight_color = (255,60,0)
        for c in contours:
            cv2.drawContours(prediction_layer, [c], -1, highlight_color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = str(num_cartridges)
        cv2.putText(img, text, (45, 55), font, 1.5, (0,0,0), 2)
        (w, h), _ = cv2.getTextSize(text, font, 1.5, 2)
        for center in centers:
            img = cv2.circle(img, center, 4, highlight_color, -1)
            prediction_layer = cv2.circle(prediction_layer, center, 20, highlight_color, -1)   
        cv2.rectangle(img, (35, 65), (35 + w + 15, 65 - h -15), highlight_color, 4)     
        predicted_img = cv2.addWeighted(img.copy(), 0.7, prediction_layer, 0.3, 0)

        return predicted_img
    
    def inference_image(self, img, threshold = 0.6):
        original_img = img.copy()
        gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(gray_img.copy(), (256,256))
        tensor_image = torch.unsqueeze(torch.unsqueeze(torch.Tensor(resized_image), 0),0)/255.0

        # Inference process
        inference_start = time.time()
        prediction_prob = self.model(tensor_image.to(self.device))[0,0].cpu().detach().numpy()
        prediction = prediction_prob > threshold
        inference_time = time.time() - inference_start

        # Find and count contours (cartridges)
        resized_prediction = cv2.resize(prediction.astype(np.uint8)*255, (1280,960))
        contours, hierarchy = cv2.findContours(resized_prediction, 1, 2)
        num_cartridges = len(contours)
        predicted_img = self.draw_predicted_layer(img.copy(), num_cartridges, contours)

        return num_cartridges, inference_time, predicted_img, original_img