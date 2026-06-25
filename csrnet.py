#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# In[3]:


class ShanghaiDataset(Dataset):
    def __init__(self, img_paths, gt_paths):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.size = 512  # 🔥 prevents OOM

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat = sio.loadmat(self.gt_paths[idx])
        points = mat["image_info"][0,0][0,0][0]

        h0, w0 = img.shape[:2]

        # 🔥 resize image
        img = cv2.resize(img, (self.size, self.size))

        scale_x = self.size / w0
        scale_y = self.size / h0

        density = np.zeros((self.size, self.size), dtype=np.float32)

        # create point map
        for p in points:
            x = int(p[0] * scale_x)
            y = int(p[1] * scale_y)
            x = min(self.size-1, max(0, x))
            y = min(self.size-1, max(0, y))
            density[y, x] += 1

        # gaussian
        density = cv2.GaussianBlur(density, (15,15), 4)

        # 🔥 downsample to match CSRNet output
        density = cv2.resize(density, (self.size//8, self.size//8))
        density *= 64

        # to tensor
        img = transforms.ToTensor()(img)
        density = torch.tensor(density).unsqueeze(0)

        return img, density


# In[4]:


root = "ShanghaiTech/part_B"

train_imgs = sorted(glob.glob(os.path.join(root, "train_data/images/*.jpg")))
train_gts  = sorted(glob.glob(os.path.join(root, "train_data/ground-truth/*.mat")))

test_imgs = sorted(glob.glob(os.path.join(root, "test_data/images/*.jpg")))
test_gts  = sorted(glob.glob(os.path.join(root, "test_data/ground-truth/*.mat")))

train_dataset = ShanghaiDataset(train_imgs, train_gts)
test_dataset  = ShanghaiDataset(test_imgs, test_gts)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)


# In[5]:


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(pretrained=True)

        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scaler = torch.cuda.amp.GradScaler()  # 


# In[9]:


num_epochs = 50
best_mae = float("inf")

train_losses = []
val_mae_list = []

for epoch in range(num_epochs):
    # ===== TRAIN =====
    model.train()
    total_loss = 0

    for imgs, dmaps in train_loader:
        imgs = imgs.to(device)
        dmaps = dmaps.to(device)

        with torch.cuda.amp.autocast():
            preds = model(imgs)
            loss = criterion(preds, dmaps)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ===== VALIDATION =====
    model.eval()
    mae = 0

    with torch.no_grad():
        for imgs, dmaps in test_loader:
            imgs = imgs.to(device)
            dmaps = dmaps.to(device)

            preds = model(imgs)

            pred_count = preds.sum().item()
            gt_count = dmaps.sum().item()

            mae += abs(pred_count - gt_count)

    mae /= len(test_loader)
    val_mae_list.append(mae)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | MAE: {mae:.2f}")

    # ===== SAVE BEST MODEL =====
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "csrnet_best.pth")
        print("✅ Saved Best Model")


# In[10]:


model.load_state_dict(torch.load("csrnet_best.pth"))
model.eval()


# In[11]:


model.eval()

mae = 0
rmse = 0

with torch.no_grad():
    for imgs, dmaps in test_loader:
        imgs = imgs.to(device)
        dmaps = dmaps.to(device)

        preds = model(imgs)

        pred_count = preds.sum().item()
        gt_count = dmaps.sum().item()

        mae += abs(pred_count - gt_count)
        rmse += (pred_count - gt_count) ** 2

mae /= len(test_loader)
rmse = (rmse / len(test_loader)) ** 0.5

print("Final MAE:", mae)
print("Final RMSE:", rmse)


# In[12]:


model.eval()

img, gt = test_dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device))

pred_count = pred.sum().item()
gt_count = gt.sum().item()

print("Predicted Count:", pred_count)
print("Ground Truth:", gt_count)


# In[13]:


pred_map = pred.cpu().squeeze().numpy()
gt_map = gt.squeeze().numpy()

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.title(f"GT Count: {gt_count:.1f}")
plt.imshow(gt_map, cmap='jet')
plt.colorbar()

plt.subplot(1,2,2)
plt.title(f"Pred Count: {pred_count:.1f}")
plt.imshow(pred_map, cmap='jet')
plt.colorbar()

plt.show()


# In[14]:


model.eval()

for i in range(3):
    img, gt = test_dataset[i]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))

    pred_count = pred.sum().item()
    gt_count = gt.sum().item()

    print(f"Sample {i}")
    print(f"GT: {gt_count:.1f} | Pred: {pred_count:.1f}")

    plt.imshow(pred.cpu().squeeze(), cmap='jet')
    plt.title(f"Pred Count: {pred_count:.1f}")
    plt.show()


# In[15]:


plt.figure(figsize=(12,5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(train_mae_losses, label='Training MAE')
plt.plot(val_mae_losses, label='Validation MAE')
plt.title('MAE Curves')
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.legend()

plt.show()


# In[16]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for imgs, dmaps in test_loader:
        imgs = imgs.to(device)
        dmaps = dmaps.to(device)

        preds = model(imgs)

        pred_count = preds.sum().item()
        gt_count = dmaps.sum().item()

        y_true.append(gt_count)
        y_pred.append(pred_count)

# 🔥 Convert to bins (classification-style)
def to_class(x):
    if x < 50:
        return 0
    elif x < 150:
        return 1
    elif x < 300:
        return 2
    else:
        return 3

y_true_cls = [to_class(x) for x in y_true]
y_pred_cls = [to_class(x) for x in y_pred]

acc = accuracy_score(y_true_cls, y_pred_cls)
prec = precision_score(y_true_cls, y_pred_cls, average='weighted')
rec = recall_score(y_true_cls, y_pred_cls, average='weighted')
f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")


# In[17]:


import matplotlib.pyplot as plt

plt.figure()

plt.scatter(y_true, y_pred)
plt.plot([0, max(y_true)], [0, max(y_true)])  # ideal line

plt.xlabel("Ground Truth Count")
plt.ylabel("Predicted Count")
plt.title("Predicted vs Ground Truth")

plt.show()


# In[18]:


errors = np.array(y_pred) - np.array(y_true)

plt.figure()

plt.hist(errors, bins=30)
plt.title("Error Distribution (Pred - GT)")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.show()


# In[19]:


abs_errors = np.abs(errors)

plt.figure()

plt.scatter(y_true, abs_errors)

plt.xlabel("Ground Truth Count")
plt.ylabel("Absolute Error")
plt.title("Error vs Crowd Size")

plt.show()


# In[20]:


plt.figure()

plt.plot(y_true, label="Ground Truth")
plt.plot(y_pred, label="Prediction")

plt.title("Prediction vs Ground Truth (Sequence)")
plt.xlabel("Sample Index")
plt.ylabel("Count")

plt.legend()
plt.show()

