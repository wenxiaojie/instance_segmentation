# instance_segmentation
This code follows the implementation architecture of Detectron2, https://github.com/facebookresearch/detectron2.

# Goal
The goal of this project is to find which objects are in the images and also to detect its locations.

# Training
Run train.py

In train.py, register my own custom dataset which is in the coco format. Then, I applied Imagenet pretrained weights to train the model.

# Inference
Run inference.py

In inference.py, evaluate the training results on the weights that I trained. Additionally, visualize the training results on images and save them in the output_dir.
