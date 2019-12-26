import string

class ContextGenerator:
    def __init__(self, classes):
        self.classes = classes
        self.prev_context = ''

    def generate_context(self, detections, caption: str, emotion: list, age_gender: list, beaty_score: list, get_prev=False):
        if not get_prev:
            # detections format: [[box, box...], [label_num, label_num...]]
            context = self.exclude_side_phrases_from_caption(caption).rstrip()
            # exclude punctuations
            context = context.translate(str.maketrans('', '', string.punctuation)).strip()
            objects_info = self.get_objects_info(context, detections).strip()
            emotion_info = self.get_emotion_info(emotion).lower().strip()
            beauty_info = self.get_beauty_info(beaty_score).strip()
            age_gender_info = self.get_age_gender_info(age_gender).strip()
            context = f'{emotion_info} {beauty_info} {age_gender_info} {objects_info} {context} '
            context = context.strip()
            self.prev_context = context
        return self.prev_context

    def get_emotion_info(self, emotion):
        result = ''
        for em in emotion:
            result = em[1][0]
        return result

    def get_beauty_info(self, beauty_score):
        result = ''
        for score in beauty_score:
            result = self.beauty_score2desc(score)
        return result

    def get_age_gender_info(self, age_gender_detection):
        result = ''
        for age_gender in age_gender_detection:
            result = f'{age_gender[0]} {self.age2desc(age_gender[1])}'
        return result

    @staticmethod
    def beauty_score2desc(beauty_score: float):
        if beauty_score < 4.1:
            return 'unattractive'
        elif 4.1 <= beauty_score <= 4.2:
            return 'ordinary'
        elif 4.2 <= beauty_score <= 4.5:
            return 'attractive'
        else:
            return 'beautiful'

    def age2desc(self, age: int):
        if age <= 12:
            return 'child'
        elif 13 <= age <= 18:
            return 'teen'
        elif 19 <= age <= 25:
            return 'young'
        elif 26 <= age <= 60:
            return 'mature'
        else:
            return 'old'

    def get_objects_info(self, caption, detections):
        labels = set(self.classes[class_num] for class_num in detections[1])
        additional_info = ' '.join([label for label in labels if label not in caption])
        return additional_info

    def exclude_side_phrases_from_caption(self, caption):
        out_of_place_phrases = ["I think it's", "I am not really confident but I think its",
                                "I am not really confident, but I think it's"]
        for phrase in out_of_place_phrases:
            if caption.startswith(phrase):
                caption = caption[len(phrase):]
        return caption