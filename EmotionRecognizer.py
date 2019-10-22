import cv2
import numpy as np
from PIL import Image
import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
from cust_models import VGG

class EmotionRecognizer:
    NET_INPUT_SIZE = (48, 48)
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())
    print(torch.cuda.is_available())

    def __init__(self, prob_thresh):
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.net = VGG('VGG19')
        self.checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
        self.prob_thresh = prob_thresh
        self.prev_detections = []

    def detect_emotions_on_frame(self, frame, detected_faces, get_prev=False):
        # return list of items of the following format: ((lt_point: tuple, rb_point: tuple), (emotion: str, prob: int))
        emotions = []
        if not get_prev:
            for face_pos in detected_faces:
                (l, t), (r, b) = face_pos
                face_img = frame[t:b, l:r]
                emotion = self.recognize_emotion_by_face(face_img)
                if emotion:
                    emotions.append((face_pos, emotion))
            self.prev_detections = emotions
        else:
            emotions = self.prev_detections
        return emotions


    def recognize_emotion_by_face(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, self.NET_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        face_img = face_img[:, :, np.newaxis]
        face_img = np.concatenate((face_img, face_img, face_img), axis=2)

        face_img = Image.fromarray(face_img)

        inputs = transform_test(face_img)

        self.net.load_state_dict(self.checkpoint['net'])
        self.net.cuda()
        self.net.eval()

        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        prob = int(score[predicted.item()] * 100)
        most_probably_emotion = (str(self.class_names[int(predicted.cpu().numpy())]), prob)
        return most_probably_emotion if prob >= self.prob_thresh else None