#평가메트릭
import os
from sklearn.metrics import classification_report

# 1. GT 이미지 리스트 만들기
def get_gt_images(label_root):
    gt_images = set()
    for sub in ["train", "val"]:
        path = os.path.join(label_root, sub)
        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                with open(os.path.join(path, fname), "r") as f:
                    for line in f:
                        if len(line.strip()) == 0:
                            continue
                        class_id = line.strip().split()[0]
                        if class_id == "0":  # Raise Hand 클래스
                            img_name = fname.replace(".txt", ".jpg")
                            gt_images.add(img_name)
                            break
    return gt_images

# 2. 예측 이미지 리스트
def get_predicted_images(pred_dir):
    return set([f for f in os.listdir(pred_dir) if f.endswith(".jpg")])

# 3. 경로 설정 (수정 가능)
label_root = "/home/a/SCB5-Handrise-Read-write-2024-9-17-20250605T083809Z-1-001/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17/labels"
pred_dir = "/home/a/5k_HRW_yolo_Dataset_jpg/handsup_detected"

# 4. GT 및 예측 이미지 리스트 생성
gt_images = get_gt_images(label_root)
pred_images = get_predicted_images(pred_dir)

# 5. 전체 비교 대상 이미지 리스트 생성
all_images = sorted(gt_images.union(pred_images))
y_true = [1 if img in gt_images else 0 for img in all_images]
y_pred = [1 if img in pred_images else 0 for img in all_images]

# 6. 라벨 이름
target_names = ["손 안 듬", "손 듬"]

# 7. classification_report 딕셔너리로 받기
report = classification_report(y_true, y_pred, target_names=target_names, digits=2, output_dict=True)

# 8. "손 듬" 클래스만 출력
handup = report["손 듬"]

print("🙋‍♂️ 손 듬 이미지에 대한 성능:\n")
print(f"🔹 Precision: {handup['precision']:.2f}")
print(f"🔹 Recall:    {handup['recall']:.2f}")
print(f"🔹 F1-score:  {handup['f1-score']:.2f}")
print(f"🔹 Support:   {handup['support']:.0f}")

# 9. 요약 출력
print(f"\n📊 총 GT(손 듬): {len(gt_images)}장 | 예측된 손 듬: {len(pred_images)}장 | 총 비교 이미지 수: {len(all_images)}장")
