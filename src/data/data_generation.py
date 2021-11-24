# This is a script to help up capture images.
from pypylon import pylon
import cv2
import os

data_dir = "data_generation/black_caps"  # specify the folder to save the data
count = 211


def display_image(image, id):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.imwrite(os.path.join(data_dir, f"{id}.jpg"), image)
    cv2.waitKey(50)


if __name__ == "__main__":

    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()
    camera.MaxNumBuffer = 1
    numberOfImagesToGrab = 1

    while True:
        camera.StartGrabbingMax(numberOfImagesToGrab)

        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(
                2000, pylon.TimeoutHandling_ThrowException
            )

            if grabResult.GrabSucceeded():
                count += 1
                img = grabResult.Array
                cv2.imwrite(f"image_{count}.jpg", img)
                display_image(img, count)
                grabResult.Release()

        print(f"{count} images generated!")
        k = cv2.waitKey(0)
        if k == 27:  # Esc key to stop
            break

    cv2.destroyAllWindows()
    camera.Close()
