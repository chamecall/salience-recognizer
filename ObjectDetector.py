import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class ObjectDetector:
    def __init__(self, cfg_path, weights_path, thresh=0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg['_BASE_'] = 'configs/object_detector_RFCN_LLA'
        self.predictor = DefaultPredictor(self.cfg)
        self.classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

    def forward(self, img):
        outputs = self.predictor(img)
        outputs = outputs['instances'].to('cpu')
        return outputs


    def draw_boxes(self, frame, object_detections):
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(object_detections)
        new_frame = v.get_image()[:, :, ::-1]
        return new_frame




