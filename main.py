import cv2
from ultralytics import YOLO
import torch, os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import mediapipe as mp

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

# YOLO pose model 로딩
yolo = YOLO("yolo11n-pose.pt")
yolo.fuse()

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# 손가락 개수 판별 함수
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

# 얼굴 이미지 텐서를 OpenCV로
def tensor_to_cv2(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# 출석 기준 얼굴 불러오기
check_img = Image.open("/home/a/zzang/Automatic-Attendance-System-yj/dong.png").convert("RGB")

# 설정값
THRESHOLD = 0.3
vis_thr = 0.7
idx = dict(r_sh=6, r_wr=10, l_sh=5, l_wr=9)

# 얼굴 인식 모델 로딩
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
resnet = InceptionResnetV1(classify=False).eval().to(device)

ckpt_path = '/home/a/zzang/Automatic-Attendance-System-yj/facenet_supcon_best_v5.pt'
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt['backbone'] if 'backbone' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
resnet.load_state_dict(state_dict, strict=False)

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    res = yolo.predict(frame, imgsz=640, conf=0.25, half=True, device=0, verbose=False)[0]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(Image.fromarray(rgb))

    for i, (xy, cf) in enumerate(zip(res.keypoints.xy, res.keypoints.conf)):
        hand_up = False
        hand_label = ""

        # 오른손 판단 (손목이 어깨보다 위)
        if cf[idx['r_sh']] > vis_thr and cf[idx['r_wr']] > vis_thr:
            shx, shy = xy[idx['r_sh']]
            wrx, wry = xy[idx['r_wr']]
            if wry < shy:
                hand_up = True
                hand_label = "왼손"

        # 왼손 판단
        if cf[idx['l_sh']] > vis_thr and cf[idx['l_wr']] > vis_thr:
            shx, shy = xy[idx['l_sh']]
            wrx, wry = xy[idx['l_wr']]
            if wry < shy:
                hand_up = True
                hand_label = "오른손"

        if not hand_up:
            continue

        print(f"손들음 ({hand_label})")

        # YOLO 박스 crop
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = hands.process(crop_rgb)

        hand_fingers_open = False
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if count_fingers(hand_landmarks) == 5:
                    hand_fingers_open = True
                mp_drawing.draw_landmarks(
                    crop,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

        cv2.imshow("HandCrop", crop)  # 항상 crop 시각화

        if not hand_fingers_open:
            print("손가락 5개 아님 → 출석 제외")
            continue

        input_face = mtcnn(crop)
        check_face = mtcnn(check_img)
        if input_face is not None:
            pil_face = tensor_to_cv2(input_face)
            pil_face_resized = cv2.resize(pil_face, (320, 320))
            cv2.imshow("DetectedFace", pil_face_resized)

        if input_face is not None and check_face is not None:
            imput_emb = F.normalize(resnet(input_face.unsqueeze(0).to(device)), dim=1)
            check_emb = F.normalize(resnet(check_face.unsqueeze(0).to(device)), dim=1)
            sim = torch.matmul(check_emb, imput_emb.T).item()
            if sim >= THRESHOLD:
                print("✅ 출석 확인")
            else:
                print("❌ 결석 처리")
        else:
            print("얼굴 인식 실패")

    cv2.imshow("Pose", res.plot())
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
