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

cap = cv2.VideoCapture(0)
vis_thr = 0.7
idx = dict(r_sh=6, r_wr=10, l_sh=5, l_wr=9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
resnet = InceptionResnetV1(classify=False).eval().to(device)

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
        face = mtcnn(crop)
        if face is not None:
            pil_face = tensor_to_cv2(face)
            face1_rgb = cv2.cvtColor(pil_face, cv2.COLOR_BGR2RGB)
            if face1_rgb.size:
                cv2.imshow(f"RaisedHand", face1_rgb)

    cv2.imshow("Pose", res.plot())
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
