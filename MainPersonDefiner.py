
class MainPersonDefiner:
    def __init__(self):
        self.prev_main_person_index = None


    def get_main_person_by_face_size(self, faces, get_prev=False):
        if not get_prev and faces:
            main_person = max(enumerate(faces),
                              key=lambda face: (face[1][1][0] - face[1][0][0]) * (face[1][1][1] - face[1][0][1]))
            self.prev_main_person_index = main_person[0]
        return self.prev_main_person_index