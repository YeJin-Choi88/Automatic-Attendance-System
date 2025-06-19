#5k_HRW_yolo_Dataset_jp  hands-up을 검출
import os
import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolo11n-pose.pt")

# hands-up 기준: 손(wrist)이 어깨보다 위(y좌표 작음)
def is_hands_up(keypoints):
    # keypoints: [x0, y0, conf0, x1, y1, conf1, ...]
    try:
        left_shoulder_y = keypoints[5 * 3 + 1]
        right_shoulder_y = keypoints[6 * 3 + 1]
        left_wrist_y = keypoints[9 * 3 + 1]
        right_wrist_y = keypoints[10 * 3 + 1]

        return (left_wrist_y < left_shoulder_y) or (right_wrist_y < right_shoulder_y)
    except IndexError:
        return False

# 경로 설정
base_dir = "images"
sub_dirs = ["train", "val"]
output_dir = "handsup_detected"
os.makedirs(output_dir, exist_ok=True)

# 검출 수 카운트
total = 0
detected = 0

# 이미지 순회
for sub in sub_dirs:
    folder = os.path.join(base_dir, sub)
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        total += 1

        results = model.predict(img, conf=0.5, task="pose")
        if not results or results[0].keypoints is None:
            continue

        for person in results[0].keypoints.xy:
            if person is None or len(person) == 0:
                continue

            keypoints = person.cpu().numpy().flatten()
            if len(keypoints) < 33:  # 최소 11개 keypoints * 3(x, y, conf)
                continue

            if is_hands_up(keypoints):
                print(f"[✔] HANDS UP DETECTED: {img_name}")
                save_path = os.path.join(output_dir, img_name)
                cv2.imwrite(save_path, img)
                detected += 1
                break  # 하나만 감지돼도 저장 후 다음 이미지로

print(f"\n🎯 전체 이미지: {total}장 중 hands-up 이미지: {detected}장 저장됨.")
