import torchvision
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
import cv2

class BeautyEstimator:
    def __init__(self, model_path):
        use_cuda = torch.cuda.is_available()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        self.model.load_state_dict(torch.load(model_path))
        if use_cuda:
            self.model = self.model.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.prev_detections = []


    def estimate_beauty_by_face(self, frame, faces, get_prev=False):
        if not get_prev:
            frame_rgb = frame[:, :, ::-1]
            beauty_scores = []
            for face in faces:
                tl_x = face[0][0]
                tl_y = face[0][1]
                br_x = face[1][0]
                br_y = face[1][1]
                face = frame_rgb[tl_y: br_y, tl_x: br_x] / 255.
                face = resize(face, (224, 224, 3), mode='reflect')
                face = np.transpose(face, (2, 0, 1))
                face = torch.from_numpy(face).float().resize_(1, 3, 224, 224)
                face = face.type(self.FloatTensor)
                res = round(self.model(face).item(), 2)
                beauty_scores.append(res)
            self.prev_detections = beauty_scores
        return self.prev_detections
