import cv2

from Colors import Color
from DB import DB
from DetectionReader import DetectionReader
from Enums import CommandType, MediaType
from ImageOverlay import ImageOverlay
from ImageProcessing import generate_thought_balloon_by_text
from Media import Media
from TextOverlay import TextOverlay
from VideoOverlay import VideoOverlay
from VideoReader import VideoReader
from ObjectDetector import ObjectDetector
from FaceRecognizer import FaceRecognizer
from EmotionRecognizer import EmotionRecognizer
#from Captioner import Captioner
from SceneSegmentator import SceneSegmentator
from ClothesDetector import ClothesDetector
from tqdm import trange
from Captioner import Captioner

class SalienceRecognizer:
    EMOTION_PROB_THRESH = 0

    def __init__(self):
        self.output_video_file_name = '/home/algernon/samba/video_queue/SalienceRecognizer/output.mkv'
        self.emotion_recognizer = EmotionRecognizer(self.EMOTION_PROB_THRESH)
        #self.captioner = Captioner('/home/algernon/a-PyTorch-Tutorial-to-Image-Captioning/weights/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar',
        #                           '/home/algernon/a-PyTorch-Tutorial-to-Image-Captioning/weights/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
        self.segmentator = None
        self.clothes_detector = ClothesDetector("yolo/df2cfg/yolov3-df2.cfg", "yolo/weights/yolov3-df2_15000.weights", "yolo/df2cfg/df2.names")
        self.face_recognizer = FaceRecognizer()
        self.video_file_name = '/home/algernon/Downloads/BestActors.mp4'
        self.captioner = Captioner()
        self.video_reader = VideoReader(self.video_file_name)

        self.video_writer = cv2.VideoWriter(self.output_video_file_name, cv2.VideoWriter_fourcc(*"XVID"),
                                            self.video_reader.fps,
                                            (self.video_reader.width, self.video_reader.height))
        self.segmentator = SceneSegmentator(self.video_reader.fps * 5)

        self.object_detector = ObjectDetector('./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                                                'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl')


    def launch(self):
        for _ in trange(self.video_reader.frame_count):
            frame = self.video_reader.get_next_frame()
            #cur_frame_num = self.video_reader.cur_frame_num

            # emotions
            #emotion_detections = self.detect_emotions_on_frame(frame)
            emotion_detections = []
            emotions_per_frame = []
            for emotion_pos, emotion in emotion_detections:
                emotions_per_frame.append((emotion_pos, emotion))
                self.draw_emotion_box(frame, emotion_pos, emotion)

            self.segmentator.push_frame(frame)

            # clothes
            #clothes_detections = self.clothes_detector.detect_clothes(frame)
            clothes_detections = []

            self.draw_clothes(frame, clothes_detections)

            # objects
            object_detections_per_frame = self.object_detector.forward(frame)

            frame = self.object_detector.draw_boxes(frame, object_detections_per_frame)

            #labels_per_frame = [detection[0] for detection in object_detections_per_frame]

            self.show_caption(frame)
            cv2.imshow('frame', frame)
            self.video_writer.write(frame)
            cv2.waitKey(1)


    def show_caption(self, frame):
        most_clear_img = self.segmentator.get_most_clear_frame()
        caption = self.captioner.caption_img(most_clear_img)
        cv2.putText(frame, caption, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, Color.BLUE, 2)

    def draw_clothes(self, frame, clothes_detections):
        # clothes_detections: [[label: str, ((x1, y1), (x2, y2)), prob], ..]
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = Color.BLACK
        for label, ((x1, y1), (x2, y2)), prob in clothes_detections:
            text = f'{label} ({prob}%)'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1 - 2, y1 - 25), (x1 + 8.5 * len(text), y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def detect_emotions_on_frame(self, frame):
        #return list of items of the following format: ((lt_point: tuple, rb_point: tuple), (emotion: str, prob: int))
        detected_faces = self.face_recognizer.recognize_faces_on_image(frame)
        emotions = []
        for face_pos in detected_faces:
            (l, t), (r, b) = face_pos
            face_img = frame[t:b, l:r]
            emotion = self.emotion_recognizer.recognize_emotion_by_face(face_img)
            if emotion:
                emotions.append((face_pos, emotion))
        return emotions

    def draw_emotion_box(self, frame, emotion_pos, emotion: list):

        cv2.rectangle(frame, *emotion_pos, Color.GOLD, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{emotion[0]} - {emotion[1]}', (emotion_pos[0][0], emotion_pos[0][1] - 5), font, 1,
                    Color.YELLOW, 3)






    def is_rect_inside_rect(self, in_rect: tuple, out_rect: tuple):
        lt_in_box_point_inside_out_box = all([out_rect[0][i] <= in_rect[0][i] <= out_rect[1][i] for i in range(2)])
        rb_in_box_point_inside_out_box = all([out_rect[0][i] <= in_rect[0][i] <= out_rect[1][i] for i in range(2)])
        return lt_in_box_point_inside_out_box and rb_in_box_point_inside_out_box



    # def generate_video_overlay(self, command: Command, coords: tuple):
    #     video_cap = cv2.VideoCapture(command.media.file_name)
    #     duration = command.media.duration if command.duration == 0 else command.duration
    #     return VideoOverlay(video_cap, duration, coords, self.video_reader.one_frame_duration)
    #
    # def generate_image_overlay_object(self, command: Command, coords: tuple):
    #     image = cv2.imread(command.media.file_name)
    #     return ImageOverlay(image, command.duration, coords, self.video_reader.one_frame_duration)
    #
    # def generate_text_overlay_object(self, command: Command, coords: tuple):
    #     texts = self.read_text_from_file(command.media.file_name)
    #     ellipse, text_rect = generate_thought_balloon_by_text(texts)
    #     return TextOverlay((ellipse, text_rect), command.duration, coords, self.video_reader.one_frame_duration)

    # def read_text_from_file(self, txt_file):
    #     with open(txt_file) as txt:
    #         texts = txt.readlines()
    #     return texts

    def close(self):
        if self.video_reader:
            self.video_reader.close()
        if self.video_writer:
            self.video_writer.release()


salience_recognizer = SalienceRecognizer()
salience_recognizer.launch()
salience_recognizer.close()
