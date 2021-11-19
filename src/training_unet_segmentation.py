import sys
sys.path.append('../')

import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
import utils.preprocessing
import glob
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch.model import UNet
from pytorch.looper import Looper
import pytorch.utils as torch_utils
from pytorch.utils import RegressionDataset, get_count, SegmentationDataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def display_img(img):
    fig, ax = plt.subplots(figsize=(10, 20))
    plt.imshow(img, cmap="gray")


transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(
            brightness=0.5*np.random.rand(),
            contrast=0.5*np.random.rand(),
       #    saturation=0.1*np.random.rand(),
       #    hue=0.1*np.random.rand()
            ),
        transforms.PILToTensor()
    ])

dataset_name = "mixed_green_black_caps"
network_architecture = "unet"
ds = SegmentationDataset("../data/mixed_green_black_caps/")
#ds = AugmentorSegmentationDataset("../data/mixed_green_black_caps/")

lengths = ((int(len(ds)*0.8), int(len(ds)*0.2)) if len(ds)%2 != 0 else (int(len(ds)*0.8) + 1, int(len(ds)*0.2)))
print(lengths)
train, val = torch.utils.data.random_split(ds, lengths)

train_loader = DataLoader(train, batch_size=8,
                        shuffle=True, num_workers=0)
valid_loader = DataLoader(val, batch_size=8,
                        shuffle=True, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = UNet(input_filters=1, filters=64, N=2).to(device)
model = torch.nn.DataParallel(model)

# initialize loss, optimized and learning rate scheduler
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=20,
                                               gamma=0.1)
writer = SummaryWriter()

# create training and validation Loopers to handle a single epoch
train_looper = Looper(model, device, loss, optimizer,
                      train_loader, len(train), tensorboard_writer=writer)
valid_looper = Looper(model, device, loss, optimizer,
                      valid_loader, len(val), tensorboard_writer=writer,
                      validation=True)

# current best results (lowest mean absolute error on validation set)
current_best = np.infty
es_count = 0
for epoch in range(25):
        print(f"Epoch {epoch + 1}\n")

        # run training epoch and update learning rate
        train_looper.run()
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run()
            # Early Stoping Check
            if valid_looper.running_loss[-1] >= valid_looper.running_loss[-2]:
                es_count += 1
            

        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            torch.save(model.state_dict(),
                       f'{dataset_name}_{network_architecture}.pth')
            print(f"\nNew best result: {result}")
        print("\n", "-"*80, "\n", sep='')

        if es_count > 2:
            break
print(f"[Training done] Best result: {current_best}")