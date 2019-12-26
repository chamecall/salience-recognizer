import moviepy as mp
import speech_recognition as sr
import moviepy.editor as mpe
import webrtcvad
import contextlib
import wave


class AudioFrame(object):
    def __init__(self, audio_bytes, timestamp):
        self.bytes = audio_bytes
        self.timestamp = timestamp
        self.is_speech = False


class SpeechRecognizer:
    FRAME_DURATION_IN_SEC = 0.03
    MAX_DELAY_IN_SEC = 0.3
    MAX_SPEECH_IN_SEC = 10
    MAX_SILENCE_PADDING_IN_FRAMES = 10
    vad = webrtcvad.Vad(3)
    MAX_SPEECH_IN_FRAMES = round(MAX_SPEECH_IN_SEC / FRAME_DURATION_IN_SEC)
    MAX_DELAY_IN_FRAMES = round(MAX_DELAY_IN_SEC / FRAME_DURATION_IN_SEC)
    TEMP_WAV_NAME = 'speech.wav'

    def __init__(self, video_filename):
        wav_name = self.extract_audio_from_video(video_filename)
        audio, self.sample_rate = self.read_wave(wav_name)
        self.frames = self.frame_generator(self.FRAME_DURATION_IN_SEC, audio, self.sample_rate)
        self.recognizer = sr.Recognizer()
        self.prev_recognized_text = ''
        self.speech_frames = []
        self.cur_delay_in_frames = 0
        self.cur_start_silence_pad_in_frames = None
        self.first_frame_num = 0

    @staticmethod
    def read_wave(path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    @staticmethod
    def write_wave(path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    @staticmethod
    def frame_generator(frame_duration_sec, audio, sample_rate):
        n = int(sample_rate * frame_duration_sec * 2)
        offset = 0
        timestamp = 0
        duration = (float(n) / sample_rate) / 2
        while offset + n < len(audio):
            yield AudioFrame(audio[offset:offset + n], timestamp)
            timestamp += duration
            offset += n

    def rewind_audio(self, to):
        last_frame_num = to / 1000 % self.FRAME_DURATION_IN_SEC
        frames_count_to_process = last_frame_num - self.first_frame_num
        self.first_frame_num = last_frame_num

        for _ in range(frames_count_to_process):
            frame = next(self.frames)
            is_speech = self.vad.is_speech(frame.bytes, self.sample_rate)
            speech_len = len(self.speech_frames)

            if is_speech:
                frame.is_speech = True
                if self.cur_start_silence_pad_in_frames is None:
                    self.cur_start_silence_pad_in_frames = self.cur_delay_in_frames
                self.speech_frames.append(frame)
                self.cur_delay_in_frames = 0
            elif not is_speech and self.cur_delay_in_frames == self.MAX_DELAY_IN_FRAMES or \
                    speech_len == self.MAX_SPEECH_IN_FRAMES:
                # if there are frames with speech
                if speech_len > self.cur_delay_in_frames:
                    start_pad_frame_num = self.cur_start_silence_pad_in_frames - self.MAX_SILENCE_PADDING_IN_FRAMES
                    end_pad_diff = self.cur_delay_in_frames - self.MAX_SILENCE_PADDING_IN_FRAMES
                    # avoid negative end_pad_diff
                    end_pad_diff = end_pad_diff if end_pad_diff > 0 else 0
                    # avoid negative start_pad_frame_num
                    start_pad_frame_num = start_pad_frame_num if start_pad_frame_num > 0 else 0
                    end_pad_diff_frame_num = speech_len - end_pad_diff
                    self.prev_recognized_text = self.transcribe_voice(
                        self.speech_frames[start_pad_frame_num:end_pad_diff_frame_num + 1])
                    self.speech_frames.clear()
                    self.cur_delay_in_frames = 0
                    self.cur_start_silence_pad_in_frames = None
                    return self.prev_recognized_text
            elif not is_speech:
                self.speech_frames.append(frame)
                self.cur_delay_in_frames += 1
        return self.prev_recognized_text

    def transcribe_voice(self, voice_frames):
        audio = b''.join([frame.bytes for frame in voice_frames])
        self.write_wave(self.TEMP_WAV_NAME, audio, self.sample_rate)

        with sr.AudioFile(self.TEMP_WAV_NAME) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = ''

        return text if text else ''

    @staticmethod
    def extract_audio_from_video(video_path):
        audio = mpe.AudioFileClip(video_path)
        audio_path = f'~{video_path.split("/")[-1].split(".")[0]}.wav'
        audio.write_audiofile(audio_path, fps=32000, nbytes=2, ffmpeg_params=['-ac', '1'])
        return audio_path
