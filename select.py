import os
import shutil
import numpy as np
from joblib import load

# 경로 설정
txt_dir = "runs/pose_t"
img_dir = "runs/pose_output"
output_dir = "runs/select"
os.makedirs(output_dir, exist_ok=True)

# 학습된 분류기 불러오기
clf = load("handup_classifier.pkl")

# 복사된 파일 수 카운트
selected_count = 0

# keypoint 파일 반복
for txt_file in os.listdir(txt_dir):
    if not txt_file.endswith("_keypoints.txt"):
        continue

    txt_path = os.path.join(txt_dir, txt_file)
    with open(txt_path, "r") as f:
        lines = f.readlines()

    people = []
    current_kp = []

    for line in lines:
        line = line.strip()
        if line.startswith("# 사람"):
            if current_kp:
                people.append(current_kp)
                current_kp = []
        elif ":" in line:
            try:
                _, coords = line.split(":")
                x, y = map(float, coords.strip().split())
                current_kp.append([x, y])
            except:
                continue

    if current_kp:
        people.append(current_kp)

    # 분류기로 손 든 사람 여부 판단
    found_handup = False
    for person_kp in people:
        if len(person_kp) != 17:
            continue  # 키포인트가 부족한 사람은 제외
        flat_kp = np.array(person_kp).flatten().reshape(1, -1)
        pred = clf.predict(flat_kp)[0]
        if pred == 1:
            found_handup = True
            break

    if found_handup:
        base = txt_file.replace("_keypoints.txt", "")
        src_img_path = os.path.join(img_dir, f"{base}_pose.jpg")
        dst_img_path = os.path.join(output_dir, f"{base}_pose.jpg")
        dst_txt_path = os.path.join(output_dir, txt_file)

        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(txt_path, dst_txt_path)
        print(f"✅ 복사 완료: {base}")
        selected_count += 1

print(f"\n📦 총 복사된 샘플 수: {selected_count}개 (jpg + txt)")

