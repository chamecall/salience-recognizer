import face_recognition
import cv2

class FaceRecognizer:
    def __init__(self):
        self.prev_detections = []

    def recognize_faces_on_image(self, image, get_prev=False):

        # return in format [((l, t), (r, b)), ...]
        if not get_prev:
            rev_image = image[:, :, ::-1]
            face_locations = face_recognition.face_locations(rev_image, number_of_times_to_upsample=0, model='cnn')
            ret_face_locations = []
            for top, right, bottom, left in face_locations:
                face_location = (left, top), (right, bottom)
                ret_face_locations.append(face_location)
            self.prev_detections = ret_face_locations
        return self.prev_detections


