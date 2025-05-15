#!/usr/bin/env python3
"""
SupCon Face Fine‑tuning Script  ‑  같은 사람(양성) 인식률 향상 버전
================================================================
"""

import argparse, gc, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import classification_report
import random, os
import wandb
from PIL import Image
import matplotlib.pyplot as plt

# ---------- SupCon Dataset (Two views per image) ----------
class SupConDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        for label, person in enumerate(sorted(os.listdir(root_dir))):
            person_path = os.path.join(root_dir, person)
            for img_file in os.listdir(person_path):
                self.samples.append((os.path.join(person_path, img_file), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        x1, x2 = self.transform(img), self.transform(img)
        return x1, x2, label

# ---------- SupCon Loss ----------
class SupConLoss(nn.Module):
    def __init__(self, T=0.1):
        super().__init__(); self.T=T
    def forward(self, feats, labels):
        B,N,D = feats.shape; feats = F.normalize(feats.view(B*N,D),dim=1)
        labels = labels.unsqueeze(1).repeat(1,N).view(-1)
        mask = torch.eq(labels[:,None],labels[None,:]).float().to(feats.device)
        logits = feats @ feats.T / self.T
        logits_mask = torch.ones_like(mask) - torch.eye(B*N,device=feats.device)
        mask*=logits_mask; logits = logits*logits_mask + (-1e9)*(1-logits_mask)
        log_prob = logits - torch.logsumexp(logits,1,keepdim=True)
        mean_pos = (mask*log_prob).sum(1)/mask.sum(1)
        return -mean_pos.mean()

class ProjectionHead(nn.Module):
    def __init__(self,in_dim=512,hid_dim=512,out_dim=128):
        super().__init__(); self.net=nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=False), nn.ReLU(inplace=True),
            nn.Linear(hid_dim,out_dim,bias=False))
    def forward(self,x): return F.normalize(self.net(x),dim=1)

# ---------- Train / Val ----------
def train_epoch(backbone, head, loader, crit, opt, dev, ep):
    backbone.train(); head.train(); tot=0
    for x1,x2,y in loader:
        x1,x2,y = [t.to(dev) for t in (x1,x2,y)]
        f1,f2 = head(backbone(x1)), head(backbone(x2))
        feats = torch.stack([f1,f2],dim=1)
        loss = crit(feats,y); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        wandb.log({'batch_train_loss':loss.item(),'epoch':ep})
    return tot/len(loader)

def val_epoch(backbone, head, loader, crit, dev):
    backbone.eval(); head.eval(); tot=0
    with torch.no_grad():
        for x1,x2,y in loader:
            x1,x2,y = [t.to(dev) for t in (x1,x2,y)]
            f1,f2 = head(backbone(x1)), head(backbone(x2))
            feats = torch.stack([f1,f2],dim=1)
            loss = crit(feats,y); tot+=loss.item()
    return tot/len(loader)

# ---------- Evaluation (all pairwise embedding comparison) ----------
def evaluate(backbone,data_dir,device):
    tf=transforms.Compose([transforms.Resize((160,160)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
    folder=ImageFolder(data_dir,tf)
    class_embs={}
    with torch.no_grad():
        for img,label in folder:
            emb=F.normalize(backbone(img.unsqueeze(0).to(device)),dim=1).cpu()
            class_embs.setdefault(label,[]).append(emb)

    thresholds = [round(x,2) for x in torch.arange(0.3, 0.91, 0.05).tolist()]
    print("\n🔍 Threshold sweep evaluation:")
    f1_scores = []
    for thr in thresholds:
        true,pred=[],[]
        labels=list(class_embs.keys())
        for i in labels:
            for j in labels:
                if i>=j: continue
                for emb_i in class_embs[i]:
                    for emb_j in class_embs[j]:
                        cos=F.cosine_similarity(emb_i,emb_j).item()
                        true.append(0); pred.append(1 if cos>thr else 0)
        for k in labels:
            for m in range(len(class_embs[k])):
                for n in range(m+1,len(class_embs[k])):
                    cos=F.cosine_similarity(class_embs[k][m],class_embs[k][n]).item()
                    true.append(1); pred.append(1 if cos>thr else 0)
        report = classification_report(true, pred, target_names=["다른사람", "같은사람"], output_dict=True)
        print(f"\n[Threshold = {thr:.2f}]")
        print(classification_report(true,pred,target_names=["다른사람","같은사람"]))
        f1_scores.append((thr, report['같은사람']['f1-score']))

    # 시각화
    import matplotlib.pyplot as plt
    thresholds, scores = zip(*f1_scores)
    plt.figure(figsize=(8,4))
    plt.plot(thresholds, scores, marker='o')
    plt.title('F1-score for 같은사람 by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.grid(True)
    plt.show()

# ---------- Main ----------
def main(args):
    wandb.init(project="facenet-supcon", config=vars(args))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

    ds = SupConDataset(args.data_dir, tf)
    val_len = int(len(ds)*args.val_ratio); train_len = len(ds)-val_len
    train_ds,val_ds=torch.utils.data.random_split(ds,[train_len,val_len])
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    train_ld=DataLoader(train_ds, shuffle=True, **dl_kw)
    val_ld  =DataLoader(val_ds, shuffle=False, **dl_kw)

    backbone=InceptionResnetV1(pretrained='casia-webface', classify=False)
    head=ProjectionHead(out_dim=args.proj_dim)
    if args.multi_gpu and torch.cuda.device_count()>1:
        backbone, head = [nn.DataParallel(m) for m in (backbone, head)]
    backbone, head = backbone.to(dev), head.to(dev)

    crit = SupConLoss(args.temperature)
    opt = torch.optim.SGD([
        {'params': backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': head.parameters(),     'lr': args.lr}], momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best=1e9; save_dir=Path('models'); save_dir.mkdir(exist_ok=True)
    for ep in range(1,args.epochs+1):
        tr=train_epoch(backbone,head,train_ld,crit,opt,dev,ep)
        vl=val_epoch(backbone,head,val_ld,crit,dev)
        sch.step(); wandb.log({"epoch_train_loss":tr,"epoch_val_loss":vl,"epoch":ep})
        print(f"[Ep {ep}/{args.epochs}] Train {tr:.4f} | Val {vl:.4f}")
        if vl<best:
            best=vl; torch.save({"backbone":backbone.state_dict()}, save_dir/'facenet_supcon_best_v6.pt')
            print("  🔥 New best checkpoint saved")
    print(f"\n✅ Training done. Best val {best:.4f}")
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    if args.eval_device=='cpu': backbone=backbone.to('cpu'); torch.cuda.empty_cache()
    evaluate(backbone,args.data_dir,torch.device(args.eval_device))

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-2)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--eval_device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()

if __name__=='__main__':
    main(parse_args())
