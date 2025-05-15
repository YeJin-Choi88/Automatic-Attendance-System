import cv2
import numpy as np
from models.mtcnn import MTCNN, extract_face
from models.inception_resnet_v1 import InceptionResnetV1
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# device 설정 및 MTCNN 초기화
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device:', device)
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

    
# 이미지 로드
img = Image.open('val_img/ningning.png').convert('RGB')
img2 = Image.open('val_img/karina2.png').convert('RGB')


resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

ckpt = torch.load('fine_tuning_model/facenet_supcon_best_v3.pt', map_location=device)

# 1) 훈련 때 저장한 그대로 꺼내기
backbone_sd = ckpt['backbone']        # <-- 핵심!

# 2) multi-GPU(DataParallel)로 저장했다면 'module.' 접두어 제거
backbone_sd = {k.replace('module.', ''): v for k, v in backbone_sd.items()}

# 3) 로드
missing, unexpected = resnet.load_state_dict(backbone_sd, strict=False)
print(f"loaded,  missing={len(missing)}  unexpected={len(unexpected)}")


# 얼굴 검출(박스, 확률, 랜드마크)
boxes, probs, points = mtcnn.detect(img, landmarks=True)
boxes2, probs2, points2 = mtcnn.detect(img2, landmarks=True)

face1 = mtcnn(img)
face2 = mtcnn(img2)

# 원본 이미지 복사 및 그리기 객체 생성
img_draw = img.copy()
img2_draw = img2.copy()
draw = ImageDraw.Draw(img_draw)
draw2 = ImageDraw.Draw(img2_draw)


# 얼굴 박스와 랜드마크 그리기
for i, (box, pts) in enumerate(zip(boxes, points)):
    # 얼굴 박스 그리기 (빨간색)
    draw.rectangle(box.tolist(), outline="red", width=2)
    # 각 랜드마크 점에 작은 파란 원 그리기
    for pt in pts:
        r = 2  # 반지름
        x, y = pt[0], pt[1]
        draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="blue")

for i, (box, pts) in enumerate(zip(boxes2, points2)):
    # 얼굴 박스 그리기 (빨간색)
    draw2.rectangle(box.tolist(), outline="red", width=2)
    # 각 랜드마크 점에 작은 파란 원 그리기
    for pt in pts:
        r = 2  # 반지름
        x, y = pt[0], pt[1]
        draw2.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="blue")

        
pre = InceptionResnetV1(pretrained='casia-webface', classify=False).eval().to(device)
emb_pre1 = F.normalize(pre(face1.unsqueeze(0).to(device)), dim=1)
emb_pre2 = F.normalize(pre(face2.unsqueeze(0).to(device)), dim=1)
print("pre-train sim :", F.cosine_similarity(emb_pre1, emb_pre2).item())

emb1 = F.normalize(resnet(face1.unsqueeze(0).to(device)), dim=1)
emb2 = F.normalize(resnet(face2.unsqueeze(0).to(device)), dim=1)
cos_sim = F.cosine_similarity(emb1, emb2)
print("supcon   sim :", cos_sim.item())

# 얼굴 텐서를 numpy로 변환 → OpenCV용 이미지
def tensor_to_cv2(tensor_img):
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)    # [-1,1] → [0,255]
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

face1_cv = tensor_to_cv2(face1)
face2_cv = tensor_to_cv2(face2)

face1_rgb = cv2.cvtColor(face1_cv, cv2.COLOR_BGR2RGB)
face2_rgb = cv2.cvtColor(face2_cv, cv2.COLOR_BGR2RGB)

# Figure와 Axes 생성
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# 첫 번째 얼굴
axes[0].imshow(face1_rgb)
axes[0].axis('off')
axes[0].set_title('Face 1')

# 두 번째 얼굴
axes[1].imshow(face2_rgb)
axes[1].axis('off')
axes[1].set_title('Face 2')

# 전체 타이틀에 유사도 표시
sim_score = cos_sim.item()
fig.suptitle(f'Cosine Similarity: {sim_score:.4f}', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


