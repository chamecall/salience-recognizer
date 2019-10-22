from detectron2.utils.logger import setup_logger
import cv2
from Colors import Color

setup_logger()

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
        self.prev_detections = [[], []]

    def forward(self, img, get_prev=False):
        if not get_prev:
            outputs = self.predictor(img)
            outputs = outputs['instances'].to('cpu')
            self.prev_detections = outputs

            fields = outputs.get_fields()
            boxes, classes = fields['pred_boxes'].tensor.tolist(), fields['pred_classes'].tolist()
            self.prev_detections = [boxes, classes]
        return self.prev_detections

    def draw_boxes(self, frame, object_detections):
        # detections format: [[box, box...], [label_num, label_num...]]

        boxes, classes = object_detections

        for box, class_num in zip(boxes, classes):
            box = [int(num) for num in box]
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), Color.YELLOW, 2)
            cv2.putText(frame, f'{self.classes[class_num]}', (box[0], box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, Color.RED, 3)

        return frame

