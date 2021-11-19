import sys

sys.path.append("../")

from pypylon import pylon
import numpy as np
import cv2
import torch
from pytorch.model import UNet


COUNT_MODEL = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SIZE = (256, 256)
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1536

BOUNDARY_COLOR = (0, 255, 0)
CENTER_COLOR = (255, 255, 255)


if COUNT_MODEL:
    path = "../pytorch/trained_models/segmentation_mixed_augmented_UNet_1604044090.034343.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(input_filters=1, filters=64, N=2).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))


def detect_circles(image):
    # resize image
    resized_image = cv2.resize(image, SIZE)
    # to Tensor and unsqueeze batch dimension
    tensor_image = (
        torch.unsqueeze(torch.unsqueeze(torch.Tensor(resized_image), 0), 0) / 255
    )
    # predict with model
    prediction = model(tensor_image)[0, 0].cpu().detach().numpy() > 0.5
    # get centers from resized mask
    # resize prediction to original size

    resized_prediction = cv2.resize(
        prediction.astype(np.uint8) * 255, (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # find centers
    contours, hierarchy = cv2.findContours(resized_prediction, 1, 2)

    def find_center(contour):
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    circles = [find_center(contour) for contour in contours]

    return np.array(circles), len(circles)


def overlay_circles_on_top(img, circles):
    circles = np.uint16(np.around(circles))
    img_over_lay = np.zeros_like(img)
    for no, i in enumerate(circles):
        img_over_lay = cv2.circle(img_over_lay, (i[0], i[1]), 30, (100, 100, 100), -1)
        img_over_lay = cv2.circle(img_over_lay, (i[0], i[1]), 2, (0, 0, 255), 3)

    added_image = cv2.addWeighted(img, 0.2, img_over_lay, 0.5, 0)
    return added_image


def display_image(image, count):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    cv2.putText(
        image,
        str(count),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType,
    )

    cv2.imshow("image", image)


if __name__ == "__main__":

    nodeFile = "NodeMap.pfs"

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    camera.Open()

    # Load Camera Configuration
    # pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data.

            image = converter.Convert(grabResult)

            img = image.Array

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # save raw image
            # cv2.imwrite(f"image_{i}.jpg", img)
            if COUNT_MODEL:
                # detect circles
                circles, count = detect_circles(img)
                # generate image with overlayed circles
                overlayed_img = overlay_circles_on_top(img, circles)
                display_image(overlayed_img, count)

            # save overlayed image
            # cv2.imwrite(f"image_with_circles_{i}.jpg", new_image)

            # display image
            k = cv2.waitKey(1)
            # use ESC to exit
            if k == 27:
                break
        grabResult.Release()

    camera.StopGrabbing()
    cv2.destroyAllWindows()
