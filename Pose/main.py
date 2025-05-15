import cv2
from ultralytics import YOLO
import os, time, cv2, torch
from PIL import Image
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch.nn.functional as F
import numpy as np

yolo = YOLO("yolo11n-pose.pt")  # ① YOLO 객체
yolo.fuse()                     # ② 성능 최적화 (in-place)
THRESHOLD = 0.3
DURATION = 3.0  # 평균 측정을 위한 지속 시간 (초)
FPS = 10        # 프레임 간격 기준
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vis_thr = 0.7
check_img = Image.open("input/최예진.png").convert("RGB")

idx = dict(r_sh=6, r_wr=10, l_sh=5, l_wr=9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
resnet = InceptionResnetV1(classify=False).eval().to(device)

# 파인튜닝 모델 불러오기지
ckpt_path = 'facenet_supcon_best_v5.pt'
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt['backbone'] if 'backbone' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
resnet.load_state_dict(state_dict, strict=False)

# 얼굴 텐서를 numpy로 변환 → OpenCV용 이미지
def tensor_to_cv2(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)    # [-1,1] → [0,255]
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    res   = yolo.predict(frame, imgsz=640, conf=0.25,
                         half=True, device=0, verbose=False)[0]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(Image.fromarray(rgb))

    for i, (xy, cf) in enumerate(zip(res.keypoints.xy, res.keypoints.conf)):
        hand_up = False
        if cf[idx['r_sh']] > vis_thr and cf[idx['r_wr']] > vis_thr:
            shx, shy = xy[idx['r_sh']]
            wrx, wry = xy[idx['r_wr']]
            if wry < shy and wrx < shx:
                print("손들음(왼손)")
                hand_up =True

        if cf[idx['l_sh']] > vis_thr and cf[idx['l_wr']] > vis_thr:
            shx, shy = xy[idx['l_sh']]
            wrx, wry = xy[idx['l_wr']]
            if wry < shy and wrx > shx:
                print("손들음(오른손)")
                hand_up =True
        if not hand_up:
            continue
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])   # bbox 좌표
        x1, y1 = max(0, x1), max(0, y1)                # 경계 보호
        crop = frame[y1:y2, x1:x2]
        input_face = mtcnn(crop)
        check_face = mtcnn(check_img)
        
        if input_face is not None and check_face is not None:
            imput_emb = F.normalize(resnet(input_face.unsqueeze(0).to(device)), dim=1)
            check_emb = F.normalize(resnet(check_face.unsqueeze(0).to(device)), dim=1)

            sim = torch.matmul(check_emb, imput_emb.T).item()
            pil_face = tensor_to_cv2(input_face)
            face1_rgb = cv2.cvtColor(pil_face, cv2.COLOR_BGR2RGB)
            if face1_rgb.size:
                cv2.imshow(f"RaisedHand", face1_rgb)
            if sim >= THRESHOLD:
                print("출석 확인")
            else:
                print("결석 처리")
        else:
            print("얼굴 검출 실패 (input_face 또는 check_face가 None)")

    cv2.imshow("Pose", res.plot())
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
