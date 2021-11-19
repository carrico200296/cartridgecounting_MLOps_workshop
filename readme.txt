Project: NN Cartridge Counting

Folders:
>> /dataset: 
	- The dataset is grayscale images (.jpg) and their corresponding annotation files (.npy). 
	- The dataset includes images of black and green caps. Only a few images have been included in this folder.
	- Each annotation file is a numpy array that contains an element/raw for every label. Each cap within the image is annotated with a label.
	- Each label is defined with 3 parameters [x,y,diameter] in pixels.
	- An example of this annotation has been plotted in 'labelled_image_sample.png'

>> /trained_models:
	- This folder includes the trained models used for the inference.
	- File extension: python pickled object (.pth).

>> /src/pytorch:
	- This folder has the 'model.py' file which includes the U-Net model written in pytorch. This file is used to load the model during the inference process.

>> /src/backend_cartridgecounting/:
	- This folder has 2 files: 'functions.py' and 'inference_camera_image.py'
	- 'inference_camera_image.py' is used to get an image from the camera, make the prediction and visualize the cartridge counting results (raw and predicted images).
	- 'functions.py' file has the CartridgeCounting class, which is used in the 'inference_camera_image.py' file to load the trained model, inference image and draw the prediction layer.



