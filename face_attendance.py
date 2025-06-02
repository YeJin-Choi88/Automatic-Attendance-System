import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import torch.nn.functional as F
import numpy as np
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

class FaceAttendanceSystem:
    def __init__(self, model_ckpt_path='facenet_supcon_best_v5.pt'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.yolo = YOLO("yolo11n-pose.pt")
        self.yolo.fuse()

        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device, post_process=True)
        self.resnet = InceptionResnetV1(classify=False).eval().to(self.device)

        # Load face recognition model checkpoint
        ckpt = torch.load(model_ckpt_path, map_location=self.device)
        state_dict = ckpt['backbone'] if 'backbone' in ckpt else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.resnet.load_state_dict(state_dict, strict=False)

        self.THRESHOLD = 0.3
        self.VIS_THR = 0.7
        self.idx = dict(r_sh=6, r_wr=10, l_sh=5, l_wr=9)

    def load_check_face(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            return self.mtcnn(img)
        except Exception as e:
            print(f"[ERROR] 기준 얼굴 이미지 로딩 실패: {e}")
            return None

    def recognize_and_check_attendance(self, frame, check_face):
        if check_face is None:
            return None, None, None

        frame = cv2.flip(frame, 1)

        # YOLO 포즈 예측
        res = self.yolo.predict(frame, imgsz=640, conf=0.25, half=True, device=0, verbose=False)[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for i, (xy, cf) in enumerate(zip(res.keypoints.xy, res.keypoints.conf)):
            hand_up = False
            if cf[self.idx['r_sh']] > self.VIS_THR and cf[self.idx['r_wr']] > self.VIS_THR:
                shx, shy = xy[self.idx['r_sh']]
                wrx, wry = xy[self.idx['r_wr']]
                if wry < shy and wrx < shx:
                    hand_up = True

            if cf[self.idx['l_sh']] > self.VIS_THR and cf[self.idx['l_wr']] > self.VIS_THR:
                shx, shy = xy[self.idx['l_sh']]
                wrx, wry = xy[self.idx['l_wr']]
                if wry < shy and wrx > shx:
                    hand_up = True

            if not hand_up:
                continue

            x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]

            input_face = self.mtcnn(crop)

            if input_face is not None:
                input_emb = F.normalize(self.resnet(input_face.unsqueeze(0).to(self.device)), dim=1)
                check_emb = F.normalize(self.resnet(check_face.unsqueeze(0).to(self.device)), dim=1)

                sim = torch.matmul(check_emb, input_emb.T).item()
                if sim >= self.THRESHOLD:
                    return True, sim, crop  # 출석 인정
                else:
                    return False, sim, crop  # 출석 실패 (유사도 낮음)

        return None, None, None  # 손 들지 않음 또는 얼굴 없음
