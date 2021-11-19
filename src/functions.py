'''
Project: NN Cartridge Counting using U-Net instance segmentation
Script description: CartridgeCounting class -> used to make the prediction for an image.
'''

import sys
sys.path.append("../")
from pypylon import pylon
import torch
from pytorch.model import UNet
import cv2
import numpy as np
import time


class CartridgeCounting:
    def __init__(self):
        # load model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(input_filters=1, filters=64, N=2).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load("pytorch/trained_models/segmentation_mixed_augmented_UNet_1604044090.034343.pth", map_location=self.device))
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

        # Find and find contours (cartridges)
        resized_prediction = cv2.resize(prediction.astype(np.uint8)*255, (1280,960))
        contours, hierarchy = cv2.findContours(resized_prediction, 1, 2)
        num_cartridges = len(contours)
        predicted_img = self.draw_predicted_layer(img.copy(), num_cartridges, contours)

        return num_cartridges, inference_time, predicted_img, original_img