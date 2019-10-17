from captionbot import CaptionBot
import cv2

class Captioner:
    def __init__(self):
        self.caption_bot = CaptionBot()
        self.prev_caption = ''


    def caption_img(self, frame):
        if frame is not None:
            img_name = 'to_caption.jpg'
            cv2.imwrite(img_name, frame)
            caption = self.caption_bot.file_caption(img_name)
            self.prev_caption = caption
        return self.prev_caption