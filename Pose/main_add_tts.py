import cv2, os, time, torch, io, tempfile, threading
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# 음성 출력 중복 방지를 위한 상태 변수
speaking = threading.Event()

# gTTS 기반 비동기 TTS 함수 (영상은 실시간으로 유지되며 음성은 순차 실행)
def speak(text, volume_gain=0, speed=1.0, wait=0):
    def _play():
        speaking.set()
        try:
            tts = gTTS(text=text, lang='ko')
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                tts.save(fp.name)
                sound = AudioSegment.from_file(fp.name, format="mp3")
                sound += volume_gain
                play(sound)
                if wait > 0:
                    time.sleep(wait)
        finally:
            speaking.clear()

    while speaking.is_set():
        time.sleep(0.1)
    threading.Thread(target=_play, daemon=True).start()

# 설정값
THRESHOLD = 0.3
DURATION = 4.0  # 출석 인식 대기 시간 (초)
vis_thr = 0.7

# YOLO 로드
yolo = YOLO("yolo11n-pose.pt")
yolo.fuse()

# 인덱스 (포즈 인식용)
idx = dict(r_sh=6, r_wr=10, l_sh=5, l_wr=9)

# 디바이스 및 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
resnet = InceptionResnetV1(classify=False).eval().to(device)

# 파인튜닝된 모델 가중치 로딩
ckpt_path = 'facenet_supcon_best_v5.pt'
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt['backbone'] if 'backbone' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
resnet.load_state_dict(state_dict, strict=False)

# 얼굴 텐서를 OpenCV 이미지로 변환
def tensor_to_cv2(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)
    return img_np  # 변환하지 말고 그대로 사용 (RGB로 표시하고 싶으면 matplotlib 등에서 처리)

# 손 들었는지 판단
def is_hand_raised(xy, cf):
    if cf[idx['r_sh']] > vis_thr and cf[idx['r_wr']] > vis_thr:
        shx, shy = xy[idx['r_sh']]
        wrx, wry = xy[idx['r_wr']]
        if wry < shy and wrx < shx:
            return True
    if cf[idx['l_sh']] > vis_thr and cf[idx['l_wr']] > vis_thr:
        shx, shy = xy[idx['l_sh']]
        wrx, wry = xy[idx['l_wr']]
        if wry < shy and wrx > shx:
            return True
    return False

# 메인
def main():
    cap = None
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"카메라 {i}번 사용")
            break
    if not cap or not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 이름 리스트 불러오기
    df = pd.read_csv("name.csv", encoding="utf-8-sig")
    name_list = df.iloc[:, 0].dropna().tolist()[1:]

    # 출석 시작 TTS
    

    current_name = None
    check_emb = None
    sim_list = []
    start_time = None
    start = True
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        res = yolo.predict(frame, imgsz=640, conf=0.25, half=True, device=0, verbose=False)[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis_frame = res.plot() if res.boxes and res.boxes.xyxy.numel() > 0 else frame.copy()
        cv2.imshow("Pose", vis_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if start :
            speak("지금부터 개신프론티어 과목 출석체크를 시작하겠습니다.", volume_gain=8)
            start = False
        if current_name is None and name_list:
            current_name = name_list.pop(0)
            speak(current_name, volume_gain=10, speed=1.2)
            check_img_path = f"input/{current_name}.png"

            if not os.path.exists(check_img_path):
                print(f"[경고] {current_name}에 대한 참조 이미지가 존재하지 않음.")
                current_name = None
                continue

            check_img = Image.open(check_img_path).convert("RGB")
            check_face = mtcnn(check_img)
            if check_face is None:
                print(f"[경고] {current_name} 얼굴 추출 실패")
                current_name = None
                continue

            check_emb = F.normalize(resnet(check_face.unsqueeze(0).to(device)), dim=1)
            sim_list = []
            start_time = time.time()

        if current_name:
            for i, (xy, cf) in enumerate(zip(res.keypoints.xy, res.keypoints.conf)):
                if not is_hand_raised(xy, cf): continue

                x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])
                x1, y1 = max(0, x1), max(0, y1)
                crop = frame[y1:y2, x1:x2]
                input_face = mtcnn(crop)
                if input_face is None: continue

                input_emb = F.normalize(resnet(input_face.unsqueeze(0).to(device)), dim=1)
                sim = torch.matmul(check_emb, input_emb.T).item()
                sim_list.append(sim)
                cv2.imshow("Detected", tensor_to_cv2(input_face))

            if time.time() - start_time >= DURATION:
                mean_sim = np.mean(sim_list) if sim_list else 0
                if mean_sim >= THRESHOLD:
                    result = "출석 확인"
                else:
                    result = "결석 처리"
                speak(result, volume_gain=10, speed=1.1)
                print(f"[{current_name}] {result} (평균 유사도: {mean_sim:.2f})")
                current_name = None
                sim_list = []
                check_emb = None
                time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
