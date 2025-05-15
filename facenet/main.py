import os, time, cv2, torch
from PIL import Image
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch.nn.functional as F

# ── 설정 ─────────────────────────────────────
INPUT_DIR = "input"
THRESHOLD = 0.3
DURATION = 3.0  # 평균 측정을 위한 지속 시간 (초)
FPS = 10        # 프레임 간격 기준

# ── 모델 로드 ───────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device, post_process=True)
resnet = InceptionResnetV1(classify=False).eval().to(device)

# 파인튜닝 모델 불러오기
ckpt_path = 'fine_tuning_model/facenet_supcon_best_v5.pt'
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt['backbone'] if 'backbone' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
resnet.load_state_dict(state_dict, strict=False)

# ── 참조 임베딩 ──────────────────────────────
def get_reference_embeddings():
    embeddings = {}
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        name = os.path.splitext(file)[0]
        path = os.path.join(INPUT_DIR, file)
        img = Image.open(path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            emb = F.normalize(resnet(face.unsqueeze(0).to(device)), dim=1)
            embeddings[name] = emb
    return embeddings

# ── 출석 판단 ───────────────────────────────
def verify_attendance(ref_emb):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    similarities = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(Image.fromarray(rgb))
        if face is not None:
            live_emb = F.normalize(resnet(face.unsqueeze(0).to(device)), dim=1)
            sim = torch.matmul(ref_emb, live_emb.T).item()
            similarities.append(sim)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

        if time.time() - start_time > DURATION:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(similarities) == 0:
        return False, 0.0

    avg_sim = sum(similarities) / len(similarities)
    return avg_sim >= THRESHOLD, avg_sim

# ── 실행 ─────────────────────────────────────
if __name__ == "__main__":
    refs = get_reference_embeddings()
    print("[INFO] 참조 임베딩 완료. 출석 시작...")

    for name, ref_emb in refs.items():
        print(f"[INFO] {name} 학생 얼굴 확인 중...")
        result, avg_sim = verify_attendance(ref_emb)
        if result:
            print(f"✅ {name} 출석 완료 (평균 유사도: {avg_sim:.3f})")
        else:
            print(f"❌ {name} 결석 처리 (평균 유사도: {avg_sim:.3f})")
