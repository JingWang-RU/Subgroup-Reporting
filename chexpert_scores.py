"""Retrain the CheXpert predictor and write frame predictions to CHEXPERT_SCORES (GPU required).

Requires CheXpert-v1.0-small under CHEXPERT_ROOT (Stanford AIMI research use agreement). The
frame and fitting split are those of data.chexpert_frame(); 60,000 fitting images are sampled
from the fitting patients (seed 0). DenseNet-121 from ImageNet weights, two epochs, AdamW
(lr 1e-4, weight decay 1e-4), batch 128, 224x224 inputs with random horizontal flips.
GPU nondeterminism means a retrained model will not reproduce the released scores bit for bit;
data.chexpert() asserts that the frame order matches whatever scores file it reads.
"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

EPOCHS, BATCH, N_FIT = 2, 128, 60000
torch.manual_seed(0)
fit, frame = data.chexpert_frame()
fit = fit.sample(N_FIT, random_state=0).reset_index(drop=True)
norm = tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_FIT = tv.transforms.Compose([tv.transforms.Resize((224, 224)), tv.transforms.RandomHorizontalFlip(),
                                tv.transforms.ToTensor(), norm])
TF_EVAL = tv.transforms.Compose([tv.transforms.Resize((224, 224)), tv.transforms.ToTensor(), norm])


class Images(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = df.local.values, df.Y.values.astype(np.float32), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(os.path.join(data.CHEXPERT_ROOT, self.paths[i])).convert("RGB")
        return self.tf(im), self.y[i]


dev = "cuda:0"
model = tv.models.densenet121(weights=tv.models.DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(1024, 1)
model = model.to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")
loss_fn = nn.BCEWithLogitsLoss()
loader = DataLoader(Images(fit, TF_FIT), batch_size=BATCH, shuffle=True, num_workers=24,
                    pin_memory=True, drop_last=True, persistent_workers=True)
for ep in range(EPOCHS):
    model.train(); t0 = time.time(); total = 0.0
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last); yb = yb.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(xb).squeeze(1), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total += loss.item()
    print(f"epoch {ep} loss {total / len(loader):.4f} ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def score(df):
    model.eval(); out = []
    for xb, _ in DataLoader(Images(df, TF_EVAL), batch_size=256, num_workers=24, pin_memory=True):
        xb = xb.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(xb).squeeze(1)).float().cpu().numpy())
    return np.concatenate(out)


np.savez(data.CHEXPERT_SCORES, f_pool=score(frame), pool_local=frame.local.values)
print("wrote", data.CHEXPERT_SCORES)
