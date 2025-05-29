import pandas as pd
from gtts import gTTS
from pydub import AudioSegment, effects
import tempfile
import subprocess
import os
import io
import time

def speak(text, volume_gain=0, speed=1.0, wait=0.3):
    """텍스트를 TTS로 변환하여 재생"""
    try:
        # TTS 생성 및 메모리 저장
        tts = gTTS(text=text, lang='ko')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # AudioSegment 변환
        audio = AudioSegment.from_file(fp, format="mp3")

        # 볼륨 증폭 및 속도 조절
        if volume_gain != 0:
            audio += volume_gain
        if speed != 1.0:
            audio = effects.speedup(audio, playback_speed=speed)

        # 임시 wav로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            temp_path = tmp_wav.name

        # ffplay로 재생 (화면 없이, 자동 종료)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", temp_path])

        os.remove(temp_path)
        time.sleep(wait)

    except Exception as e:
        print(f"TTS 재생 중 오류: {e}")


