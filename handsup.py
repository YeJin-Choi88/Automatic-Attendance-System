import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 📁 키포인트 txt 파일 경로
keypoint_dir = "runs/pose_t"
data = []

# 🙋‍♀️ 손 든 상태 판별 함수
def is_hand_raised(kp):
    try:
        L_shoulder = kp[5]
        R_shoulder = kp[6]
        L_wrist = kp[7]
        R_wrist = kp[8]
        return int(L_wrist[1] < L_shoulder[1] or R_wrist[1] < R_shoulder[1])
    except IndexError:
        return 0

# 📥 데이터셋 만들기
for fname in os.listdir(keypoint_dir):
    if not fname.endswith(".txt"):
        continue

    with open(os.path.join(keypoint_dir, fname), "r") as f:
        lines = f.readlines()
    
    person_kps = []
    cur_kp = []

    for line in lines:
        line = line.strip()
        if line.startswith("# 사람"):
            if cur_kp:
                person_kps.append(np.array(cur_kp))
                cur_kp = []
        elif line:
            try:
                _, xy = line.split(":")
                x, y = map(float, xy.strip().split())
                cur_kp.append([x, y])
            except:
                continue

    if cur_kp:  # 마지막 사람
        person_kps.append(np.array(cur_kp))

    for kp in person_kps:
        if kp.shape == (17, 2):
            x = kp.flatten()  # 17*2 = 34차원
            y = is_hand_raised(kp)
            data.append((x, y))

# ✅ NumPy 배열로 변환
X = np.array([d[0] for d in data])
y = np.array([d[1] for d in data])

print(f"📊 총 샘플 수: {len(X)} | 손 든 샘플: {np.sum(y)}")

# 🔍 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🎯 분류기 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 📈 평가
y_pred = clf.predict(X_test)
print("\n📋 분류 성능 보고서:")
print(classification_report(y_test, y_pred))

# 💾 모델 저장
joblib.dump(clf, "handup_classifier.pkl")
print("✅ 모델 저장 완료: handup_classifier.pkl")

