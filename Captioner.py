from captionbot import CaptionBot
import cv2

class Captioner:
    def __init__(self):
        self.caption_bot = CaptionBot()
        self.prev_caption = ''


    def caption_img(self, frame):
        is_frame_new = False
        if frame is not None:
            img_name = 'to_caption.jpg'
            cv2.imwrite(img_name, frame)
            caption = self.caption_bot.file_caption(img_name)
            self.prev_caption = caption
            is_frame_new = True
        return self.prev_caption, is_frame_new