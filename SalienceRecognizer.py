import cv2

from Colors import Color
import string
from VideoReader import VideoReader
from ObjectDetector import ObjectDetector
from FaceRecognizer import FaceRecognizer
from EmotionRecognizer import EmotionRecognizer
from Captioner import Captioner
from SceneSegmentator import SceneSegmentator
from ClothesDetector import ClothesDetector
from tqdm import trange
from JokePicker import JokePicker
from DetectionReader import DetectionReader
from Captioner import Captioner
from AgeGenderPredictor import AgeGenderPredictor
from BeautyEstimator import BeautyEstimator

class SalienceRecognizer:
    EMOTION_PROB_THRESH = 0



    def __init__(self):
        self.output_video_file_name = 'output.mkv'
        self.emotion_recognizer = EmotionRecognizer(self.EMOTION_PROB_THRESH)

        self.segmentator = None
        #self.clothes_detector = ClothesDetector("yolo/df2cfg/yolov3-df2.cfg", "yolo/weights/yolov3-df2_15000.weights", "yolo/df2cfg/df2.names")
        self.video_file_name = '/home/algernon/Downloads/BestActors.mp4'
        self.captioner = Captioner()
        self.video_reader = VideoReader(self.video_file_name)
        self.joke_picker = JokePicker('joke_picker/shortjokes.csv', 'joke_picker/joke_picker.fse')
        self.video_writer = cv2.VideoWriter(self.output_video_file_name, cv2.VideoWriter_fourcc(*"XVID"),
                                            self.video_reader.fps,
                                            (self.video_reader.width, self.video_reader.height))
        self.segmentator = SceneSegmentator(self.video_reader.fps * 5)
        self.object_detection_reader = DetectionReader('detections.json')
        self.object_detector = ObjectDetector('./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                                                'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl')
        self.age_gender_predictor = AgeGenderPredictor()
        self.beauty_estimator = BeautyEstimator('/home/algernon/isBeauty/weights/epoch_50.pkl')

    def launch(self):
        for _ in trange(self.video_reader.frame_count):
            frame = self.video_reader.get_next_frame()
            age_gender_prediction = []
            beauty_score = []
            cur_frame_num = self.video_reader.cur_frame_num

            # faces
            detected_faces = FaceRecognizer.recognize_faces_on_image(frame)
            # emotions
            # emotion_detections = self.detect_emotions_on_frame(frame, detected_faces)
            # emotions_per_frame = []
            # for emotion_pos, emotion in emotion_detections:
            #     emotions_per_frame.append((emotion_pos, emotion))
            #     self.draw_emotion_box(frame, emotion_pos, emotion)

            # age gender
            if detected_faces:
                age_gender_predictions = self.age_gender_predictor.detect_age_dender_by_faces(frame, detected_faces)
                main_person_index = self.get_main_person_by_face_size(detected_faces)
                age_gender_prediction.append(age_gender_predictions[main_person_index])

                beauty_score = self.beauty_estimator.estimate_beauty_by_face(detected_faces, frame)
                self.apply_beauty_scores_on_frame(frame, detected_faces, beauty_score)

            self.segmentator.push_frame(frame)

            # clothes
            #clothes_detections = self.clothes_detector.detect_clothes(frame)
            clothes_detections = []

            self.draw_clothes(frame, clothes_detections)
            caption, caption_changed = self.get_caption(frame)
            self.show_caption(frame, caption)
            # objects
            #object_detections_per_frame = self.object_detector.forward(frame)
            object_detections = self.object_detection_reader.get_detections_per_specified_frame(cur_frame_num)
            frame = self.object_detector.draw_boxes(frame, object_detections)
            #labels_per_frame = [detection[0] for detection in object_detections_per_frame]
            if caption_changed:
                context = self.generate_context(object_detections, caption, age_gender_prediction, beauty_score)
                print(context)
                jokes = self.joke_picker.pick_jokes_by_context(context)
            else:
                jokes = self.joke_picker.prev_jokes

            self.apply_jokes_on_frame(jokes, frame)

            cv2.imshow('frame', frame)
            self.video_writer.write(frame)
            cv2.waitKey(1)

    def apply_beauty_scores_on_frame(self, frame, faces, beauty_scores):
        for i, face in enumerate(faces):
            font = cv2.FONT_HERSHEY_SIMPLEX
            tl_x, tl_y = face[0]
            cv2.putText(frame, 'Beauty:' + str(beauty_scores[i]), (tl_x - 5, tl_y + 30), font, 1, (0, 0, 255), 2)

    def apply_jokes_on_frame(self, jokes, frame):
        height = frame.shape[0]
        joke_height = 40
        joke_y = height - joke_height * len(jokes)
        for joke in jokes:
            cv2.putText(frame, joke, (0, joke_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.GOLD, 2)
            joke_y += joke_height

    def generate_context(self, detections, caption: str, age_gender: list, beaty_score: list):
        # detections format: [[box, box...], [label_num, label_num...]]
        context = self.exclude_side_phrases_from_caption(caption).rstrip()
        # exclude punctuations
        context = context.translate(str.maketrans('', '', string.punctuation))
        objects_info = self.get_objects_info(context, detections).lstrip()
        beauty_info = self.get_beauty_info(beaty_score)
        age_gender_info = self.get_age_gender_info(age_gender)
        context = f'{beauty_info} {age_gender_info} {objects_info} {context} '
        return context

    def get_beauty_info(self, beauty_score):
        result = ''
        for score in beauty_score:
            result = self.beauty_score2desc(score)
        return result

    def get_main_person_by_face_size(self, faces):
        main_person = max(enumerate(faces), key=lambda face: (face[1][1][0] - face[1][0][0]) * (face[1][1][1] - face[1][0][1]))
        return main_person[0]

    def get_age_gender_info(self, age_gender_detection):
        result = ''
        for age_gender in age_gender_detection:
            result = f'{age_gender[0]} {self.age2desc(age_gender[1])}'
        return result

    def beauty_score2desc(self, beauty_score: float):
        if beauty_score < 4.1: return 'unattractive'
        elif 4.1 <= beauty_score <= 4.2: return 'ordinary'
        elif 4.2 <= beauty_score <= 4.5: return 'attractive'
        else: return 'beautiful'

    def age2desc(self, age: int):
        if age <= 12: return 'child'
        elif 13 <= age <= 18: return 'teen'
        elif 19 <= age <= 25: return 'young'
        elif 26 <= age <= 60: return 'mature'
        else: return 'old'

    def get_objects_info(self, caption, detections):
        classes = self.object_detector.classes
        labels = set(classes[class_num] for class_num in detections[1])
        additional_info = ' '.join([label for label in labels if label not in caption])
        return additional_info

    def exclude_side_phrases_from_caption(self, caption):
        out_of_place_phrases = ["I think it's", ]
        for phrase in out_of_place_phrases:
            if caption.startswith(phrase):
                caption = caption[len(phrase):]
        return caption

    def get_caption(self, frame):
        most_clear_img = self.segmentator.get_most_clear_frame()


        caption, changed = self.captioner.caption_img(most_clear_img)
        return caption, changed

    def show_caption(self, frame, caption):
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

    def detect_emotions_on_frame(self, frame, detected_faces):
        #return list of items of the following format: ((lt_point: tuple, rb_point: tuple), (emotion: str, prob: int))
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
