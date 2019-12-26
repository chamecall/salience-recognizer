import moviepy as mp
import speech_recognition as sr
import moviepy.editor as mpe
import webrtcvad


class SpeechRecognizer:
    FRAME_DURATION_IN_MS = 30
    MAX_DELAY_IN_MS = 300
    MAX_SPEECH_IN_S = 10
    MAX_SILENCE_PADDING_IN_FRAMES = 10
    vad = webrtcvad.Vad(3)
    max_speech_in_frames = int(MAX_SPEECH_IN_S / (FRAME_DURATION_IN_MS / 1000))
    max_delay_in_frames = MAX_DELAY_IN_MS // FRAME_DURATION_IN_MS

    def __init__(self, video_filename):
        self.wav_name = self.extract_audio_from_video(video_filename)

        self.recognizer = sr.Recognizer()


    def recognize_speech(self, from_, to_):
        pass


    def extract_audio_from_video(self, video_path):
        audio = mpe.AudioFileClip(video_path)
        audio_path = f'~{video_path.split("/")[-1].split(".")[0]}.wav'
        audio.write_audiofile(audio_path, fps=32000, nbytes=2, ffmpeg_params=['-ac', '1'])
        return audio_path