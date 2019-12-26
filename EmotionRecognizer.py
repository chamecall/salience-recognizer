import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from cust_models import VGG
import imutils
import transforms

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


class EmotionRecognizer:
    NET_INPUT_SIZE = (48, 48)


    def __init__(self, prob_thresh):
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.net = VGG('VGG19')
        self.checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
        self.net.load_state_dict(self.checkpoint['net'])
        self.net.eval()
        # example = torch.rand(1, 3, 44, 44)
        # print(self.net)
        # traced_script_module = torch.jit.trace(self.net, example)
        # traced_script_module.save("emotion_recognition_vgg.pt")
        # print("It've done..")
        # return
        self.net.cuda()

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


        ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.cuda()
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)
        prob = int(score[predicted.item()] * 100)
        most_probably_emotion = (str(self.class_names[int(predicted.cpu().numpy())]), prob)
        return most_probably_emotion if prob >= self.prob_thresh else None





