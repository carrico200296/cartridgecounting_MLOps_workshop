'''
Project: NN Cartridge Counting using U-Net instance segmentation
Script description: 1. The Basler camera is open using the pypylon library.
                    2. The image is processed using the CartridgeCounting class.
                    3. The real-time image and the predicted image are displayed.
                    4. Info in the terminal: number of cartridges, inference time, accumulated number of cartridges.
'''

import sys
sys.path.append("../")
from pypylon import pylon
import cv2
import time
from functions import CartridgeCounting


if __name__=='__main__':

    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Load Camera Configuration
    # nodeFile = "NodeMap.pfs"
    # pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Load U-Net model
    model = CartridgeCounting()

    cv2.namedWindow('Real-Time Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Image", 1280, 960)
    accu_cartridges = 0

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            cv2.imshow("Real-Time Image", img)

            # Press ESC to exit
            k = cv2.waitKey(100)
            if k == 27:
                cv2.destroyAllWindows()
                break

            num_cartridges, inference_time, predicted_img, original_img = model.inference_image(img, threshold=0.6)
            accu_cartridges += num_cartridges

            time.sleep(20)
            cv2.namedWindow("Predicted Image",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Predicted Image", 1280, 960)
            cv2.imshow("Predicted Image", predicted_img)
            print("------------------------------------------")
            print("   Number of cartridges detected: %d" %num_cartridges)
            print("   Inference Time: %.2f s" %inference_time)
            print("   Accumulated cartridges: %d" %accu_cartridges)
            
        grabResult.Release()
        
    # Releasing the resource    
    camera.StopGrabbing()

    cv2.destroyAllWindows()