
class Timer:
    def __init__(self, frame_duration, step_duration):
        self.remaining_frames_amount = 0
        self.frames_duration = frame_duration
        self.step_duration = step_duration
        self.reset()

    def reset(self):
        self.remaining_frames_amount = self.frames_duration

    def ticktock(self):
        self.frames_duration -= self.step_duration
        return self.frames_duration > 0
