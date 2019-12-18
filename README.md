# instance_segmentation
This code follows the implementation architecture of Detectron2, https://github.com/facebookresearch/detectron2.

# GOAL
The goal of this homework is that develop a neural network for instance segmentation.

# TRAINGING
Run train.py
In train.py, register my own custom dataset which is in the coco format. Then, I applied Imagenet pretrained weights to train the model.

# INFERENCE
Run inference.py
In inference.py, evaluate the training results on the weights that I trained. Additionally, visualize the training results on images and save them in the directory.
