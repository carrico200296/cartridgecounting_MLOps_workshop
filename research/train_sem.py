import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.draw import polygon, line
import glob
import torch
import torch.nn as nn
from pytorch.model import UNet
from pytorch.looper import Looper
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from adabelief_pytorch import AdaBelief
import json
import albumentations as albu


DATA_DIR = "../data/black_caps"


transform = albu.Compose(
    [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.GaussNoise(),
    ]
)


def generate_mask(path: str):
    with open(path, "r") as f:
        file = json.load(f)

    # mask = np.zeros((file["width/height"][1], file["width/height"][0]))
    mask = np.zeros((1536, 2048))

    if file["objects"] == []:
        # return empy mask if no polygons
        return mask

    # iterate over polygons
    for i in range(len(file["objects"])):
        points = np.array(file["objects"][i])
        rr, cc = polygon(points[:, 1], points[:, 0])

        mask[rr, cc] = 1

    return mask


class SegmentationDataset(Dataset):
    def __init__(self, path_to_dir, transform=None):
        self.path = path_to_dir
        self.image_id_list = list(glob.glob(f"{DATA_DIR}/*.jpg"))
        self.mask_list = list(glob.glob(f"{DATA_DIR}/*.json"))
        self.transform = transform

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = list(
            map(
                lambda x: cv2.resize(x, (256, 256)),
                [
                    cv2.imread(self.image_id_list[idx], 0),
                    generate_mask(self.mask_list[idx]),
                ],
            )
        )

        if self.transform:
            img, target = np.array(sample[0]), np.array(sample[1])
            sample_1, sample_2 = self.transform(image=img, mask=target), self.transform(
                image=img, mask=target
            )
            input_image_1, target_1 = (
                sample_1["image"] / 255.0,
                sample_1["mask"] / 255.0,
            )
            input_image_2, target_2 = (
                sample_2["image"] / 255.0,
                sample_2["mask"] / 255.0,
            )
            # augmented_batch = [aug(**x) for x in repeat(x, 4)]
            return (
                torch.unsqueeze(torch.Tensor(input_image_1), 0),
                torch.unsqueeze(torch.Tensor(input_image_2), 0),
                (torch.Tensor(target_1).unsqueeze(0)),
                (torch.Tensor(target_2).unsqueeze(0)),
            )
        else:
            input_image, target = sample[0] / 255.0, sample[1] / 255.0
            return (
                torch.unsqueeze(torch.Tensor(input_image), 0),
                torch.Tensor(target).unsqueeze(0),
            )


ds = SegmentationDataset(DATA_DIR, transform=transform)

lengths = (int(len(ds) * 0.95), len(ds) - int(len(ds) * 0.95))
print(lengths)
train, val = torch.utils.data.random_split(ds, lengths)

train_loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
model = UNet(input_filters=1, out_channels=1, filters=64, N=2).to(device)

# loss = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
consistency_loss = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


optimizer = AdaBelief(
    model.parameters(),
    lr=1e-3,
    eps=1e-16,
    betas=(0.9, 0.999),
    weight_decouple=True,
    rectify=False,
)


# create training and validation Loopers to handle a single epoch
# train_looper = Looper(model, device, loss, optimizer, train_loader, len(train))
# valid_looper = Looper(
# model, device, loss, optimizer, valid_loader, len(val), validation=True
# )


current_best = np.infty
for epoch in range(31):
    supervised_loss_running = 0
    unsupervised_loss_running = 0
    valid_loss = 0
    print(f"Epoch {epoch + 1}")

    for x_1, x_2, y_1, y_2 in train_loader:
        model.train()
        x_1 = x_1.to(device)
        x_2 = x_2.to(device)
        y_1 = y_1.to(device)
        y_2 = y_2.to(device)

        optimizer.zero_grad()

        out_1 = model(x_1)
        out_2 = model(x_2)

        supervised_loss = criterion(out_1, y_1) + criterion(out_2, y_2)
        unsupervised_loss = torch.abs(
            consistency_loss(x_1, x_2) - 0.1 * consistency_loss(out_1, out_2)
        )

        loss = 0.7 * supervised_loss + 0.3 * unsupervised_loss
        loss.backward()
        optimizer.step()

        supervised_loss_running += supervised_loss.item()
        unsupervised_loss_running += unsupervised_loss.item()

    for x_1, _, y_1, _ in valid_loader:
        model.eval()
        x_1 = x_1.to(device)
        y_1 = y_1.to(device)

        optimizer.zero_grad()

        out_1 = model(x_1)

        loss = criterion(out_1, y_1)
        loss.backward()
        optimizer.step()

        valid_loss += supervised_loss.item()

    print(
        f"Supervised loss: {supervised_loss_running} \n Unsupervised loss: {unsupervised_loss_running} \n Validation Loss: {valid_loss}"
    )

    if epoch % 10 == 0:
        for x, _, y, _ in valid_loader:
            x = x.to(device)
            prediction = model(x)
            plt.subplot(1, 2, 1)
            plt.imshow(x[0, 0].detach().cpu())
            plt.imshow(y[0, 0].cpu().detach(), alpha=0.5)
            plt.subplot(1, 2, 2)
            plt.imshow(x[0, 0].detach().cpu())
            plt.imshow(prediction[0, 0].cpu().detach(), alpha=0.5)
            plt.show()

torch.save(
    model, f"pytorch/trained_models/unet_segmentation_{time.time()}.pth"
)
