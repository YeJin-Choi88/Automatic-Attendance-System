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

def main():
    # CSV 파일에서 이름 읽기
    df = pd.read_csv("name.csv", encoding="utf-8-sig")
    name_list = df.iloc[:, 0].dropna().tolist()[1:]  # 헤더 제외

    # 출석 시작 멘트
    start_text = "지금부터 개신프론티어과목 출석체크를 시작하겠습니다."
    print("TTS:", start_text)
    speak(start_text, volume_gain=8, wait=3)

    # 이름 하나씩 부르기
    for name in name_list:
        print(f"TTS: {name}")
        speak(name, volume_gain=10, speed=1.2)

        check = input("확인 여부 (Y/N): ").strip().lower()
        if check == "y":
            txt = f"확인 되었습니다."
            speak(txt, volume_gain=10, speed=1.2)
            print(f"TTS: {txt}\n")
        else:
            txt = f"결석처리 되었습니다.."
            speak(txt, volume_gain=10, speed=1.2)
            print(f"TTS: {txt}\n")
        time.sleep(1)

if __name__ == "__main__":
    main()
