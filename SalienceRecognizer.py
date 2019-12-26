import cv2
from tqdm import trange

from AgeGenderPredictor import AgeGenderPredictor
from BeautyEstimator import BeautyEstimator
from Captioner import Captioner
from Colors import Color
from ContextGenerator import ContextGenerator
from DetectionReader import DetectionReader
from EmotionRecognizer import EmotionRecognizer
from FaceRecognizer import FaceRecognizer
from JokePicker import JokePicker
from MainPersonDefiner import MainPersonDefiner
from ObjectDetector import ObjectDetector
from SceneSegmentator import SceneSegmentator
from VideoReader import VideoReader
import moviepy.editor as mpe


class SalienceRecognizer:
    EMOTION_PROB_THRESH = 0
    DELAY_TO_DETECT_IN_SECS = 5
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.output_video_file_name = '/home/algernon/samba/video_queue/SalienceRecognizer/videos/processed/output2.mkv'
        self.emotion_recognizer = EmotionRecognizer(self.EMOTION_PROB_THRESH)

        self.segmentator = None
        # self.clothes_detector = ClothesDetector("yolo/df2cfg/yolov3-df2.cfg", "yolo/weights/yolov3-df2_15000.weights", "yolo/df2cfg/df2.names")
        self.video_file_name = '/home/algernon/Videos/source_videos/interview_anna.webm'
        self.captioner = Captioner()
        self.video_reader = VideoReader(self.video_file_name)
        self.joke_picker = JokePicker('joke_picker/shortjokes.csv', 'joke_picker/joke_picker.fse')
        self.video_writer = cv2.VideoWriter(self.output_video_file_name, cv2.VideoWriter_fourcc(*"XVID"),
                                            self.video_reader.fps,
                                            (self.video_reader.width, self.video_reader.height))
        self.face_recognizer = FaceRecognizer()

        


        self.segmentator = SceneSegmentator(self.video_reader.fps * self.DELAY_TO_DETECT_IN_SECS)
        self.object_detection_reader = DetectionReader('detections.json')
        self.object_detector = ObjectDetector('./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                                              'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl')
        self.age_gender_predictor = AgeGenderPredictor()
        self.beauty_estimator = BeautyEstimator('/home/algernon/DNNS/isBeauty/weights/epoch_50.pkl')
        self.main_person_definer = MainPersonDefiner()
        self.context_generator = ContextGenerator(self.object_detector.classes)






    def launch(self):
        for _ in trange(self.video_reader.frame_count):
            frame = self.video_reader.get_next_frame()
            cur_frame_num = self.video_reader.cur_frame_num

            frame_for_detection = self.is_there_needing_to_update(frame)
            use_prev_detects = True if frame_for_detection is None else False

            if not use_prev_detects:
                # faces
                detected_faces = self.face_recognizer.recognize_faces_on_image(frame_for_detection, get_prev=use_prev_detects)



            # main person
            main_person_index = self.main_person_definer.get_main_person_by_face_size(detected_faces, get_prev=use_prev_detects)

            # emotions
            emotion_detections = self.emotion_recognizer.detect_emotions_on_frame(frame_for_detection, detected_faces, get_prev=use_prev_detects)

            emotion = [emotion_detections[main_person_index]] if main_person_index is not None else []

            # age gender
            age_gender_predictions = self.age_gender_predictor.detect_age_dender_by_faces(frame_for_detection, detected_faces, get_prev=use_prev_detects)
            age_gender_prediction = [age_gender_predictions[main_person_index]] if main_person_index is not None else []
            # beauty
            beauty_scores = self.beauty_estimator.estimate_beauty_by_face(frame_for_detection, detected_faces, get_prev=use_prev_detects)
            beauty_score = [beauty_scores[main_person_index]] if main_person_index is not None else []
            # clothes
            # clothes_detections = self.clothes_detector.detect_clothes(frame)
            # clothes_detections = []
            # self.draw_clothes(frame, clothes_detections)

            # caption
            #caption = self.captioner.caption_img(frame_for_detection, get_prev=use_prev_detects)

            # objects
            object_detections = self.object_detector.forward(frame_for_detection, get_prev=use_prev_detects)
            #object_detections = self.object_detection_reader.get_detections_per_specified_frame(cur_frame_num)
            # context = self.context_generator.generate_context(object_detections, caption, emotion, age_gender_prediction,
            #                                                   beauty_score, get_prev=use_prev_detects)
            #cv2.putText(frame, 'Context:', (0, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.GOLD, 2)
            # cv2.putText(frame, context, (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.GOLD, 2)
            # jokes = self.joke_picker.pick_jokes_by_context(context, get_prev=use_prev_detects)

            # self.apply_jokes_on_frame(jokes, frame)
            self.apply_emotions_on_frame(frame, emotion_detections)
            self.apply_beauty_scores_on_frame(frame, detected_faces, beauty_scores)
            self.apply_age_gender_on_frame(frame, detected_faces, age_gender_predictions)
            # self.apply_caption_on_frame(frame, caption)
            frame = self.object_detector.draw_boxes(frame, object_detections)

            cv2.imshow('frame', frame)
            self.video_writer.write(frame)
            cv2.waitKey(1)

    def apply_age_gender_on_frame(self, frame, faces, age_gender_predictions):
        for i, face in enumerate(faces):
            tl_x, tl_y = face[i]
            cv2.putText(frame, f'{age_gender_predictions[i]}', (tl_x - 5, tl_y + 60), self.FONT, 1, Color.BLACK, 2)


    def apply_beauty_scores_on_frame(self, frame, faces, beauty_scores):
        for i, face in enumerate(faces):
            tl_x, tl_y = face[i]
            cv2.putText(frame, f'{ContextGenerator.beauty_score2desc(beauty_scores[i])} ({beauty_scores[i]})', (tl_x - 5, tl_y + 30), self.FONT, 1, Color.BLACK, 2)

    def apply_jokes_on_frame(self, jokes, frame):
        height = frame.shape[0]
        joke_height = 40
        joke_y = height - joke_height * len(jokes)
        for joke in jokes:
            cv2.putText(frame, joke, (0, joke_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Color.GOLD, 2)
            joke_y += joke_height

    def apply_emotions_on_frame(self, frame, emotion_detections):
        for emotion_pos, emotion in emotion_detections:
            self.draw_emotion_box(frame, emotion_pos, emotion)

    def is_there_needing_to_update(self, frame):
        self.segmentator.push_frame(frame)
        most_clear_img = self.segmentator.get_most_clear_frame()

        return most_clear_img

    def apply_caption_on_frame(self, frame, caption):
        cv2.putText(frame, caption, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, Color.BLUE, 2)

    def draw_clothes(self, frame, clothes_detections):
        # clothes_detections: [[label: str, ((x1, y1), (x2, y2)), prob], ..]
        color = Color.BLACK
        for label, ((x1, y1), (x2, y2)), prob in clothes_detections:
            text = f'{label} ({prob}%)'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1 - 2, y1 - 25), (x1 + 8.5 * len(text), y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), self.FONT, 0.5, color.WHITE, 1, cv2.LINE_AA)

    def draw_emotion_box(self, frame, emotion_pos, emotion: list):

        cv2.rectangle(frame, *emotion_pos, Color.GOLD, 2)
        cv2.putText(frame, f'{emotion[0]} - {emotion[1]}', (emotion_pos[0][0], emotion_pos[0][1] - 5), self.FONT, 1,
                    Color.BLACK, 3)

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
