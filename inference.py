import json
import cv2
import os
from utils import binary_mask_to_rle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer

cocoGt = COCO("test.json")
coco = COCO("pascal_train.json")  # load training annotations
coco_dt = []

from detectron2.data.datasets import register_coco_instances
register_coco_instances("mydataset_train", {}, "pascal_train.json", "train_images")
register_coco_instances("mydataset_test", {}, "test.json", "test_images")

cfg = get_cfg()
cfg.merge_from_file("/home/wenxiaojie/wenxiaojie_python/深度學習/深度學習HW4/"
                    "detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 20 classes 
cfg.DATASETS.TRAIN = ("mydataset_train",)
cfg.DATASETS.TEST = ("mydataset_test", )
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # sethe testing threshold for this model
predictor = DefaultPredictor(cfg)

for imgid in cocoGt.imgs:
    image = cv2.imread("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:, :, ::-1]  # load image
    outputs = predictor(image)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("myimage", v.get_image()[:, :, ::-1])
    cv2.imwrite('save_image/' + cocoGt.loadImgs(ids=imgid)[0]['file_name'], v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    n_instances = len(outputs["instances"].scores)    
    if len(outputs["instances"].pred_classes) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid  # this imgid must be same as the key of test.json
            pred['category_id'] = int(outputs["instances"].pred_classes[i]) + 1
            outputs["instances"].pred_masks[:, :, i].permute(0, 1)
            pred['segmentation'] = binary_mask_to_rle(outputs["instances"].pred_masks[i].to("cpu").numpy())
            # save binary mask to RLE, e.g. 512x512 -> rle
            pred['score'] = float(outputs["instances"].scores[i])
            coco_dt.append(pred)

with open("0856052.json", "w") as f:
    json.dump(coco_dt, f)
